"""Users tab and the signed-in user's own password."""

from __future__ import annotations

import asyncio
import logging
import math
import time
from datetime import UTC, datetime
from typing import Any

from fastapi import APIRouter, Form, HTTPException, Request
from fastapi.responses import (
    HTMLResponse,
    JSONResponse,
    RedirectResponse,
)

from ...auth import invalidate_auth_cache
from ...client_address import client_ip
from ...config import settings
from ...key_lifecycle import REASON_UNAPPROVED, hold_user_keys
from ...quotas import (
    QUOTA_FIELDS,
    UserLimits,
    default_limits,
    effective_limits,
    set_default_limits,
)
from .. import deps, login_throttle, management_keys, oidc
from ..auth_core import (
    AUTH_COOKIE,
    _external_origin,
    _set_session_cookie,
    hash_password,
    normalize_username,
    unsign_session,
    verify_password,
)
from ..form_values import _positive_int, _usd, _utc_day_start
from ..middleware import invalidate_dashboard_user_count_cache
from ..pages import _require_admin, dash_page_ctx, templates
from .login import _clear_cookie

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/users", response_class=HTMLResponse)
async def users_page(request: Request):
    db = deps.get_db()
    user = getattr(request.state, "dashboard_user", None)
    if not user or not user.is_admin:
        raise HTTPException(status_code=403, detail="Forbidden")
    users = await db.list_dashboard_users()
    defaults = await default_limits(db)
    usage = await db.usage_by_since(_utc_day_start(), "user_id")
    for u in users:
        u["own_limits"] = {f: u.get(f) for f in QUOTA_FIELDS}
        u["limits"] = effective_limits(u, defaults).to_dict()
        cost, tokens = usage.get(int(u["id"]), (0.0, 0))
        u["usage"] = {"cost": cost, "tokens": tokens}
    mgmt_keys = await db.list_management_keys()
    now = time.time()
    for k in mgmt_keys:
        exp = k["expires_at"]
        k["expires_at_dt"] = None if exp is None else datetime.fromtimestamp(exp, UTC)
        k["expired"] = exp is not None and exp <= now
    cfg = oidc.current_config()
    scheme, host = _external_origin(request)
    return templates.TemplateResponse(
        request,
        "users.html",
        await dash_page_ctx(
            request,
            {
                "dashboard_users": users,
                "quota_defaults": defaults.to_dict(),
                "current_user_id": int(user.id),
                "mfa_required": settings.dashboard_require_second_factor,
                "users_exist": bool(users),
                "management_keys": mgmt_keys,
                "management_privileges": management_keys.PRIVILEGES,
                "oidc": cfg if cfg.enabled else None,
                "oidc_problems": cfg.problems(),
                "oidc_warnings": cfg.warnings() if cfg.usable else [],
                "oidc_callback": oidc.redirect_uri(cfg, scheme, host),
                "query_error": request.query_params.get("error"),
                "query_created": request.query_params.get("created"),
                "query_updated": request.query_params.get("updated"),
                "query_approved": request.query_params.get("approved"),
                "query_signed_out": request.query_params.get("signed_out"),
            },
        ),
    )


@router.post("/users/add")
async def users_add(
    request: Request,
    username: str = Form(...),
    password: str = Form(...),
    is_admin: str = Form("0"),
):
    db = deps.get_db()
    user = getattr(request.state, "dashboard_user", None)
    if not user or not user.is_admin:
        raise HTTPException(status_code=403, detail="Forbidden")
    try:
        un = normalize_username(username)
    except ValueError:
        return RedirectResponse(url="/users?error=invalid_username", status_code=303)
    if await db.get_dashboard_user_by_username(un):
        return RedirectResponse(url="/users?error=exists", status_code=303)
    if len(password) < 8:
        return RedirectResponse(url="/users?error=password", status_code=303)
    make_admin = is_admin in ("1", "on", "true", "yes")
    hashed = await asyncio.to_thread(hash_password, password)
    await db.create_dashboard_user(un, hashed, make_admin)
    invalidate_dashboard_user_count_cache()
    return RedirectResponse(url="/users?created=1", status_code=303)


@router.post("/users/{user_id}/delete")
async def users_delete(request: Request, user_id: int):
    db = deps.get_db()
    user = getattr(request.state, "dashboard_user", None)
    if not user or not user.is_admin:
        raise HTTPException(status_code=403, detail="Forbidden")
    target = await db.get_dashboard_user_by_id(user_id)
    if not target:
        raise HTTPException(status_code=404, detail="User not found")
    # Pending admins can't use the dashboard, so they aren't counted and may
    # always be removed.
    usable_admin = target["is_admin"] and target.get("approved", True)
    if usable_admin and await db.count_dashboard_admins() <= 1:
        return RedirectResponse(url="/users?error=last-admin", status_code=303)
    await db.delete_dashboard_user(user_id)
    invalidate_dashboard_user_count_cache()
    return RedirectResponse(url="/users", status_code=303)


@router.post("/users/{user_id}/password")
async def users_set_password(
    request: Request,
    user_id: int,
    new_password: str = Form(...),
):
    db = deps.get_db()
    user = getattr(request.state, "dashboard_user", None)
    if not user or not user.is_admin:
        raise HTTPException(status_code=403, detail="Forbidden")
    target = await db.get_dashboard_user_by_id(user_id)
    if not target:
        raise HTTPException(status_code=404, detail="User not found")
    if target.get("auth_source", "local") != "local":
        return RedirectResponse(url="/users?error=sso-password", status_code=303)
    if len(new_password) < 8:
        return RedirectResponse(url="/users?error=password", status_code=303)
    # Ends every session of the account; an administrator changing their
    # own password keeps this one.
    hashed = await asyncio.to_thread(hash_password, new_password)
    await db.update_dashboard_user_password(user_id, hashed)
    resp = RedirectResponse(url="/users?updated=1", status_code=303)
    if user is not None and user.id == user_id:
        row = await db.get_dashboard_user_by_id(user_id)
        if row is not None:
            secret = await deps.get_dashboard_session_secret()
            claims = unsign_session(request.cookies.get(AUTH_COOKIE, ""), secret)
            _set_session_cookie(
                request,
                resp,
                row,
                secret,
                mfa_verified=bool(claims and claims.mfa_verified),
            )
    return resp


@router.post("/account/password")
async def account_change_password(request: Request) -> JSONResponse:
    """Change the signed-in user's own password.

    Body: {current_password, new_password}. Ends every other session of the
    account and keeps this one. Wrong guesses count toward the same lockout
    as password sign-in.
    """
    user = getattr(request.state, "dashboard_user", None)
    if user is None or user.id <= 0:
        raise HTTPException(status_code=403, detail="Sign in to change your password")
    body = await deps.json_body(request)
    current = body.get("current_password")
    new = body.get("new_password")
    if not isinstance(current, str) or not isinstance(new, str):
        raise HTTPException(
            status_code=400, detail="current_password and new_password are required"
        )
    db = deps.get_db()
    row = await db.get_dashboard_user_by_id(user.id)
    if row is None or row.get("auth_source", "local") != "local":
        raise HTTPException(
            status_code=400,
            detail="Your password is managed by your single sign-on provider",
        )
    if len(new) < 8:
        raise HTTPException(
            status_code=400, detail="The new password must be at least 8 characters"
        )
    keys = login_throttle.keys_for(client_ip(request) or "unknown", row["username"])
    wait = login_throttle.throttle.retry_after(keys)
    if wait > 0:
        seconds = max(1, math.ceil(wait))
        raise HTTPException(
            status_code=429,
            detail=f"Too many wrong passwords; try again in {seconds} s",
            headers={"Retry-After": str(seconds)},
        )
    # Counted before the check, as at sign-in.
    login_throttle.throttle.failure(keys)
    if not await asyncio.to_thread(verify_password, current, row["password_hash"]):
        raise HTTPException(status_code=400, detail="The current password is wrong")
    login_throttle.throttle.success(keys)
    await db.update_dashboard_user_password(user.id, hash_password(new))
    logger.info("Dashboard user %s changed their password", user.username)
    resp = JSONResponse({"status": "ok"})
    row = await db.get_dashboard_user_by_id(user.id)
    if row is not None:
        secret = await deps.get_dashboard_session_secret()
        claims = unsign_session(request.cookies.get(AUTH_COOKIE, ""), secret)
        _set_session_cookie(
            request,
            resp,
            row,
            secret,
            mfa_verified=bool(claims and claims.mfa_verified),
        )
    return resp


@router.post("/users/{user_id}/approve")
async def users_approve(request: Request, user_id: int):
    """Let a single sign-on account that is waiting for approval in."""
    _require_admin(request)
    db = deps.get_db()
    if not await db.approve_dashboard_user(user_id):
        raise HTTPException(status_code=404, detail="User not found")
    logger.info("Dashboard user %d approved", user_id)
    return RedirectResponse(url="/users?approved=1", status_code=303)


def _limits_from_form(form: Any) -> dict[str, Any]:
    """Quota fields from a form; blank is None (the default / unlimited)."""
    try:
        return {
            "daily_token_limit": _positive_int(
                form.get("daily_token_limit"), "tokens per day"
            ),
            "requests_per_minute": _positive_int(
                form.get("requests_per_minute"), "requests per minute"
            ),
            "daily_spend_cap_usd": _usd(form.get("daily_spend_cap_usd"), "spend cap"),
        }
    except HTTPException as exc:
        raise ValueError(str(exc.detail)) from exc


@router.post("/users/{user_id}/limits")
async def users_limits(request: Request, user_id: int):
    """Set a user's quotas, or (``reset``) return them to the defaults."""
    _require_admin(request)
    db = deps.get_db()
    form = await request.form()
    if form.get("reset"):
        limits: dict[str, Any] = dict.fromkeys(QUOTA_FIELDS)
    else:
        try:
            limits = _limits_from_form(form)
        except ValueError:
            return RedirectResponse(url="/users?error=limits", status_code=303)
    if not await db.set_dashboard_user_limits(user_id, limits):
        raise HTTPException(status_code=404, detail="User not found")
    logger.info("Limits of dashboard user %d set to %s", user_id, limits)
    return RedirectResponse(url="/users?updated=1", status_code=303)


@router.post("/users/quota-defaults")
async def users_quota_defaults(request: Request):
    """Set the default limits for users without their own."""
    _require_admin(request)
    db = deps.get_db()
    try:
        limits = _limits_from_form(await request.form())
    except ValueError:
        return RedirectResponse(url="/users?error=limits", status_code=303)
    await set_default_limits(db, UserLimits(**limits))
    invalidate_auth_cache(db)
    logger.info("Default user limits set to %s", limits)
    return RedirectResponse(url="/users?updated=1#quota-defaults", status_code=303)


@router.get("/api/users/usage")
async def users_usage(request: Request) -> JSONResponse:
    """Each user's usage today (tokens, USD), effective limits and refusals."""
    _require_admin(request)
    db = deps.get_db()
    defaults = await default_limits(db)
    usage = await db.usage_by_since(_utc_day_start(), "user_id")
    out = []
    for u in await db.list_dashboard_users():
        uid = int(u["id"])
        cost, tokens = usage.get(uid, (0.0, 0))
        out.append(
            {
                "id": uid,
                "username": u["username"],
                "limits": effective_limits(u, defaults).to_dict(),
                "own_limits": {f: u.get(f) for f in QUOTA_FIELDS},
                "usage": {"tokens": tokens, "cost_usd": cost},
            }
        )
    return JSONResponse({"users": out, "defaults": defaults.to_dict()})


@router.post("/users/{user_id}/unapprove")
async def users_unapprove(request: Request, user_id: int):
    """Send an account back to waiting for approval: its sessions end and
    its API keys are held ("owner not approved") until an admin lifts the
    hold; approving again does not lift it."""
    _require_admin(request)
    db = deps.get_db()
    user = getattr(request.state, "dashboard_user", None)
    if user is not None and int(user.id) == user_id:
        return RedirectResponse(url="/users?error=self", status_code=303)
    if not await db.unapprove_dashboard_user(user_id):
        raise HTTPException(status_code=404, detail="User not found")
    await hold_user_keys(db, user_id, REASON_UNAPPROVED, by_user_id=_actor(request))
    logger.info("Dashboard user %d un-approved", user_id)
    return RedirectResponse(url="/users?updated=1", status_code=303)


def _actor(request: Request) -> int | None:
    user = getattr(request.state, "dashboard_user", None)
    if user is None or user.auth_source == management_keys.AUTH_SOURCE:
        return None
    return int(user.id)


@router.post("/users/{user_id}/pause-keys")
@router.post("/api/users/{user_id}/pause-keys")
async def users_pause_keys(request: Request, user_id: int):
    """Hold every API key the user issued (an admin must lift each)."""
    _require_admin(request)
    db = deps.get_db()
    if await db.get_dashboard_user_by_id(user_id) is None:
        raise HTTPException(status_code=404, detail="User not found")
    n = await hold_user_keys(
        db, user_id, "paused by an administrator", by_user_id=_actor(request)
    )
    if request.url.path.startswith("/api/"):
        return JSONResponse({"user_id": user_id, "held": n})
    return RedirectResponse(url="/users?updated=1", status_code=303)


@router.post("/users/{user_id}/sign-out")
async def users_sign_out(request: Request, user_id: int):
    """End every session of another account."""
    _require_admin(request)
    db = deps.get_db()
    if await db.get_dashboard_user_by_id(user_id) is None:
        raise HTTPException(status_code=404, detail="User not found")
    await db.bump_dashboard_user_session_version(user_id)
    user = getattr(request.state, "dashboard_user", None)
    resp = RedirectResponse(url="/users?signed_out=1", status_code=303)
    if user is not None and user.id == user_id:
        resp = RedirectResponse(url="/login", status_code=303)
        _clear_cookie(request, resp, AUTH_COOKIE)
    return resp
