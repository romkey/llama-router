"""Require login when dashboard user accounts exist; restrict non-admin readers."""

from __future__ import annotations

import logging
import secrets
import time
from urllib.parse import urlsplit

from fastapi.responses import JSONResponse, RedirectResponse, Response
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request

from ..config import settings
from ..database import Database
from . import deps, management_keys
from .auth_core import (
    AUTH_COOKIE,
    DashboardUser,
    non_admin_forbidden_get_path,
    unsign_session,
)

logger = logging.getLogger(__name__)

_USER_COUNT_CACHE: tuple[int, float] | None = None
_USER_COUNT_TTL = 5.0


def _cached_user_count() -> int | None:
    global _USER_COUNT_CACHE
    if _USER_COUNT_CACHE is None:
        return None
    n, t = _USER_COUNT_CACHE
    if time.monotonic() - t > _USER_COUNT_TTL:
        return None
    return n


def _set_user_count_cache(n: int) -> None:
    global _USER_COUNT_CACHE
    _USER_COUNT_CACHE = (n, time.monotonic())


def invalidate_dashboard_user_count_cache() -> None:
    global _USER_COUNT_CACHE
    _USER_COUNT_CACHE = None


async def _user_count(db: Database) -> int:
    c = _cached_user_count()
    if c is not None:
        return c
    n = await db.count_dashboard_users()
    _set_user_count_cache(n)
    return n


def _metrics_token_ok(request: Request) -> bool:
    """A scraper's ``Authorization: Bearer`` matches LLAMA_ROUTER_METRICS_TOKEN."""
    token = settings.metrics_token.strip()
    if not token:
        return False
    raw = (request.headers.get("authorization") or "").strip()
    if not raw.lower().startswith("bearer "):
        return False
    return secrets.compare_digest(raw[7:].strip(), token)


async def _dispatch_management_key(
    request: Request, call_next: RequestResponseEndpoint, db: Database, token: str
) -> Response:
    """Serve a request signed with a management key, if its privileges allow."""
    key = await management_keys.authenticate(db, token)
    if key is None:
        return JSONResponse(
            status_code=401,
            content={"detail": "Invalid or expired management key"},
        )
    path = request.url.path
    needed = management_keys.required_privilege(request.method, path)
    if needed is None:
        return JSONResponse(
            status_code=403,
            content={"detail": "Not available to management keys"},
        )
    if not management_keys.has_privilege(key.privileges, needed):
        return JSONResponse(
            status_code=403,
            content={"detail": f"Management key lacks the {needed} privilege"},
        )
    request.state.management_key = key
    # Not an admin: routes that show admins more (the status payload, API
    # key details in routing explanations) must not show it to a key.
    request.state.dashboard_user = DashboardUser(
        id=0,
        username=f"key:{key.name}",
        is_admin=False,
        auth_source=management_keys.AUTH_SOURCE,
    )
    if request.method not in ("GET", "HEAD"):
        logger.info(
            "Management key %d (%s): %s %s", key.id, key.name, request.method, path
        )
    return await call_next(request)


def _own_data_path(path: str) -> bool:
    return (
        path.startswith("/api/chat/")
        or path == "/api/my-keys"
        or path.startswith("/api/my-keys/")
    )


_LOGOUT_PATHS = ("/logout", "/logout/everywhere")
_SAFE_METHODS = ("GET", "HEAD", "OPTIONS")


def _netloc(scheme: str, netloc: str) -> str:
    """*netloc* lowercased, without the default port for *scheme*."""
    n = netloc.strip().lower()
    for s, port in (("http", ":80"), ("https", ":443")):
        if scheme in (s, "") and n.endswith(port):
            return n[: -len(port)]
    return n


def _trusted_origins() -> set[str]:
    out = set()
    extra = settings.dashboard_trusted_origins.split(",")
    for raw in [settings.oidc_redirect_base_url, *extra]:
        u = urlsplit(raw.strip())
        if u.scheme and u.netloc:
            out.add(f"{u.scheme}://{_netloc(u.scheme, u.netloc)}")
    return out


def cross_site_refusal(request: Request) -> str | None:
    """Why a state-changing request looks forged by another site, or None.

    Browsers send ``Sec-Fetch-Site`` on HTTPS (and localhost); it is set by
    the browser and cannot be forged by a page. Over plain HTTP they still
    send ``Origin`` on every POST, which must name this dashboard's host.
    A request with neither header comes from a script, curl or a very old
    browser: it is let through, since no current browser leaves both out of
    a cross-site POST, and the SameSite=Lax session cookie is not sent on
    one anyway.
    """
    if request.method in _SAFE_METHODS:
        return None
    origin = (request.headers.get("origin") or "").strip()
    u = urlsplit(origin) if origin and origin != "null" else None
    canonical = f"{u.scheme}://{_netloc(u.scheme, u.netloc)}" if u else None
    if canonical is not None and canonical in _trusted_origins():
        return None
    site = (request.headers.get("sec-fetch-site") or "").strip().lower()
    if site in ("same-origin", "none"):
        return None
    if site:
        return f"Sec-Fetch-Site: {site}"
    if not origin:
        return None
    if u is None:
        return "Origin: null"
    hosts = {
        _netloc("", request.headers.get("host") or request.url.netloc),
        _netloc("", request.headers.get("x-forwarded-host", "").split(",")[0]),
    }
    if _netloc(u.scheme, u.netloc) in hosts:
        return None
    return f"Origin: {origin}"


def _forbidden(request: Request) -> Response:
    if request.url.path.startswith("/api/"):
        return JSONResponse(status_code=403, content={"detail": "Forbidden"})
    return RedirectResponse(url="/?error=forbidden", status_code=302)


class DashboardAuthMiddleware(BaseHTTPMiddleware):
    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        path = request.url.path

        if path in ("/health", "/healthz"):
            return await call_next(request)

        if path == "/metrics" and _metrics_token_ok(request):
            return await call_next(request)

        # Before sign-in handling, so a forged POST /login (signing the
        # victim in as the attacker) is refused too.
        refusal = cross_site_refusal(request)
        if refusal is not None:
            logger.warning(
                "Refused cross-site %s %s (%s)", request.method, path, refusal
            )
            return JSONResponse(
                status_code=403, content={"detail": "Cross-site request refused"}
            )

        if path == "/login" or path.startswith("/login"):
            return await call_next(request)

        db = deps.get_db()
        n_users = await _user_count(db)

        if n_users == 0:
            request.state.auth_bootstrap = True
            request.state.dashboard_users_exist = False
            request.state.dashboard_user = None
            return await call_next(request)

        request.state.auth_bootstrap = False
        request.state.dashboard_users_exist = True

        token = management_keys.bearer_token(request)
        if token is not None:
            return await _dispatch_management_key(request, call_next, db, token)

        secret = await deps.get_dashboard_session_secret()
        raw = request.cookies.get(AUTH_COOKIE)
        session = unsign_session(raw, secret) if raw else None
        user: DashboardUser | None = None
        if session is not None:
            row = await db.get_dashboard_user_by_id(session.user_id)
            # Revoked (session_version bumped), expired, or awaiting approval.
            if row and session.valid_for(row):
                user = DashboardUser(
                    id=row["id"],
                    username=row["username"],
                    is_admin=row["is_admin"],
                    auth_source=row.get("auth_source") or "local",
                )

        if user is None:
            if path.startswith("/api/"):
                return JSONResponse(
                    status_code=401, content={"detail": "Not authenticated"}
                )
            next_url = str(request.url.path)
            if request.url.query:
                next_url += "?" + str(request.url.query)
            loc = "/login"
            if next_url != "/" and not next_url.startswith("/login"):
                from urllib.parse import quote

                loc = f"/login?next={quote(next_url, safe='/:?&=')}"
            return RedirectResponse(url=loc, status_code=302)

        request.state.dashboard_user = user

        if user and request.method == "POST" and path in _LOGOUT_PATHS:
            return await call_next(request)

        if not user.is_admin:
            # Everyone may chat and issue themselves API keys; those APIs
            # only touch the user's own conversations and keys.
            if request.method != "GET" and not _own_data_path(path):
                return _forbidden(request)
            if non_admin_forbidden_get_path(path):
                return _forbidden(request)

        return await call_next(request)
