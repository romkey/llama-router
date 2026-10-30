"""Require login when dashboard user accounts exist; restrict non-admin readers."""

from __future__ import annotations

import logging
import secrets
import time

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
        uid = unsign_session(raw, secret) if raw else None
        user: DashboardUser | None = None
        if uid is not None:
            row = await db.get_dashboard_user_by_id(uid)
            if row:
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

        if user and request.method == "POST" and path == "/logout":
            return await call_next(request)

        if not user.is_admin:
            # Everyone may chat; the chat API only touches the user's own
            # conversations.
            if request.method != "GET" and not path.startswith("/api/chat/"):
                return _forbidden(request)
            if non_admin_forbidden_get_path(path):
                return _forbidden(request)

        return await call_next(request)
