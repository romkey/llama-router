"""Sign-in (password and single sign-on) and sign-out."""

from __future__ import annotations

import asyncio
import logging
import math

from fastapi import APIRouter, Form, HTTPException, Request
from fastapi.responses import (
    HTMLResponse,
    RedirectResponse,
)

from ...client_address import client_ip, forwarded_for_ignored
from ...config import settings
from .. import deps, login_throttle, oidc
from ..auth_core import (
    AUTH_COOKIE,
    _cookie_secure,
    _external_origin,
    _safe_next_url,
    _set_session_cookie,
    burn_password_check,
    normalize_username,
    verify_password,
)
from ..middleware import invalidate_dashboard_user_count_cache
from ..pages import templates

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/login", response_class=HTMLResponse)
async def login_page(request: Request):
    db = deps.get_db()
    cfg = oidc.current_config()
    users_exist = await db.count_dashboard_users() > 0
    # With no accounts only an admin-group SSO sign-in can create the first.
    if not users_exist and not (cfg.usable and cfg.admin_group):
        return RedirectResponse(url="/", status_code=302)
    err = request.query_params.get("error")
    next_url = _safe_next_url(request.query_params.get("next"))
    return templates.TemplateResponse(
        request,
        "login.html",
        {
            "error": err,
            "next_url": next_url,
            "dash_user": None,
            "show_users_tab": False,
            "oidc_enabled": cfg.usable,
            "oidc_display_name": cfg.display_name,
            # With no accounts yet only SSO can sign anyone in.
            "password_login": cfg.password_login_enabled and users_exist,
            "dashboard_origin": settings.dashboard_webauthn_origin,
        },
    )


_warned_untrusted_forwarded_for = False


def _warn_untrusted_forwarded_for(request: Request) -> None:
    """Log once per process that a proxy's X-Forwarded-For is being ignored.

    Behind an unlisted proxy every sign-in shares the proxy's address, so
    one client's failures throttle everyone.
    """
    global _warned_untrusted_forwarded_for
    if _warned_untrusted_forwarded_for or not forwarded_for_ignored(request):
        return
    _warned_untrusted_forwarded_for = True
    logger.warning(
        "Ignoring X-Forwarded-For from %s, which is not in "
        "LLAMA_ROUTER_TRUSTED_PROXIES; sign-in throttling uses the proxy's "
        "address for every client. List the reverse proxy there.",
        request.client.host if request.client else "unknown",
    )


@router.post("/login")
async def login_submit(
    request: Request,
    username: str = Form(...),
    password: str = Form(...),
    next: str = Form("/"),
):
    if not oidc.current_config().password_login_enabled:
        raise HTTPException(status_code=404, detail="Not Found")
    db = deps.get_db()
    if await db.count_dashboard_users() == 0:
        return RedirectResponse(url="/", status_code=302)
    # X-Forwarded-For counts only from LLAMA_ROUTER_TRUSTED_PROXIES; a client
    # can't rotate it to dodge the per-address limit.
    _warn_untrusted_forwarded_for(request)
    ip = client_ip(request) or "unknown"
    try:
        un = normalize_username(username)
    except ValueError:
        un = None
    keys = login_throttle.keys_for(ip, un) if un else [f"ip:{ip}"]
    wait = login_throttle.throttle.retry_after(keys)
    if wait > 0:
        logger.warning("Password sign-in for %r from %s throttled", un, ip)
        return _login_throttled(request, next, wait)
    # Counted as a failure before the password is checked and cleared on
    # success, so parallel attempts can't all slip in before any is recorded.
    login_throttle.throttle.failure(keys)
    if un is None:
        return RedirectResponse(url="/login?error=invalid", status_code=303)
    row = await db.get_dashboard_user_by_username(un)
    if row and row.get("auth_source", "local") == "local":
        # scrypt takes tens of milliseconds; keep it off the event loop.
        ok = await asyncio.to_thread(verify_password, password, row["password_hash"])
    else:
        await asyncio.to_thread(burn_password_check, password)
        ok = False
    if not ok or row is None:
        return RedirectResponse(url="/login?error=invalid", status_code=303)
    login_throttle.throttle.success(keys)
    from ..mfa import password_login_response

    return await password_login_response(
        request,
        row["id"],
        _safe_next_url(next),
        session_version=row["session_version"],
    )


def _login_throttled(request: Request, next_url: str, wait: float) -> HTMLResponse:
    seconds = max(1, math.ceil(wait))
    return templates.TemplateResponse(
        request,
        "login.html",
        {
            "error": "throttled",
            "retry_after": seconds,
            "next_url": _safe_next_url(next_url),
            "dash_user": None,
            "show_users_tab": False,
            "oidc_enabled": oidc.current_config().usable,
            "oidc_display_name": oidc.current_config().display_name,
            "password_login": True,
        },
        status_code=429,
        headers={"Retry-After": str(seconds)},
    )


@router.get("/login/oidc")
async def login_oidc_start(request: Request):
    cfg = oidc.current_config()
    if not cfg.usable:
        raise HTTPException(status_code=404, detail="Not Found")
    try:
        meta = await oidc.provider_metadata(cfg)
    except oidc.OIDCError as exc:
        logger.warning("OIDC sign-in unavailable: %s", exc)
        return RedirectResponse(url="/login?error=oidc_provider", status_code=303)
    scheme, host = _external_origin(request)
    url, state = oidc.begin_login(
        cfg,
        meta,
        callback_url=oidc.redirect_uri(cfg, scheme, host),
        next_url=_safe_next_url(request.query_params.get("next")),
        secret=await deps.get_dashboard_session_secret(),
    )
    resp = RedirectResponse(url=url, status_code=302)
    resp.set_cookie(
        oidc.STATE_COOKIE,
        state,
        httponly=True,
        samesite="lax",
        max_age=oidc.STATE_MAX_AGE,
        path="/login/oidc",
        secure=_cookie_secure(request),
    )
    return resp


@router.get(oidc.CALLBACK_PATH)
async def login_oidc_callback(request: Request):
    cfg = oidc.current_config()
    if not cfg.usable:
        raise HTTPException(status_code=404, detail="Not Found")
    db = deps.get_db()
    secret = await deps.get_dashboard_session_secret()
    try:
        claims, next_url = await oidc.complete_login(
            cfg,
            request.query_params,
            request.cookies.get(oidc.STATE_COOKIE),
            secret,
        )
        user = await oidc.sign_in(db, cfg, claims)
    except oidc.OIDCError as exc:
        level = logging.WARNING if exc.code in ("provider", "token") else logging.INFO
        logger.log(level, "OIDC sign-in failed: %s", exc)
        resp = RedirectResponse(url=f"/login?error=oidc_{exc.code}", status_code=303)
        _clear_cookie(request, resp, oidc.STATE_COOKIE, path="/login/oidc")
        return resp
    invalidate_dashboard_user_count_cache()
    resp = RedirectResponse(url=_safe_next_url(next_url), status_code=303)
    _clear_cookie(request, resp, oidc.STATE_COOKIE, path="/login/oidc")
    _set_session_cookie(request, resp, user, secret)
    return resp


def _clear_cookie(
    request: Request, resp: RedirectResponse, name: str, path: str = "/"
) -> None:
    """Expire a cookie with the attributes it was set with.

    Browsers may ignore a deletion that lacks ``Secure`` for a cookie that
    was set Secure, which would leave the session in place.
    """
    resp.delete_cookie(
        name, path=path, secure=_cookie_secure(request), httponly=True, samesite="lax"
    )


@router.post("/logout")
async def logout_post(request: Request):
    dest = "/login"
    user = getattr(request.state, "dashboard_user", None)
    cfg = oidc.current_config()
    if user is not None and user.is_sso and cfg.usable and cfg.logout_at_idp:
        try:
            meta = await oidc.provider_metadata(cfg)
            scheme, host = _external_origin(request)
            base = cfg.redirect_base_url or f"{scheme}://{host}"
            dest = oidc.logout_url(cfg, meta, base + "/login") or dest
        except oidc.OIDCError as exc:
            logger.warning("OIDC logout at the provider skipped: %s", exc)
    resp = RedirectResponse(url=dest, status_code=303)
    _clear_cookie(request, resp, AUTH_COOKIE)
    _clear_cookie(request, resp, "lr_second_factor")
    return resp


@router.post("/logout/everywhere")
async def logout_everywhere(request: Request):
    """Sign the current user out of every browser, this one included."""
    user = getattr(request.state, "dashboard_user", None)
    if user is not None and user.id > 0:
        await deps.get_db().bump_dashboard_user_session_version(user.id)
        logger.info("Dashboard user %s signed out everywhere", user.username)
    return await logout_post(request)
