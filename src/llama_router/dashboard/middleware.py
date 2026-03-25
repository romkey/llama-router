"""Require login when dashboard user accounts exist; restrict non-admin readers."""

from __future__ import annotations

import time

from fastapi.responses import JSONResponse, RedirectResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request

from . import deps
from .auth_core import (
    AUTH_COOKIE,
    DashboardUser,
    non_admin_forbidden_get_path,
    unsign_session,
)

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


async def _user_count(db) -> int:
    c = _cached_user_count()
    if c is not None:
        return c
    n = await db.count_dashboard_users()
    _set_user_count_cache(n)
    return n


def _forbidden(request: Request):
    if request.url.path.startswith("/api/"):
        return JSONResponse(status_code=403, content={"detail": "Forbidden"})
    return RedirectResponse(url="/?error=forbidden", status_code=302)


class DashboardAuthMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        path = request.url.path

        if path == "/health":
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
            if request.method != "GET":
                return _forbidden(request)
            if non_admin_forbidden_get_path(path):
                return _forbidden(request)

        return await call_next(request)
