"""Require a dashboard login; restrict non-admin readers.

With no accounts at all the dashboard is locked: it only explains how to
create the first administrator (``LLAMA_ROUTER_ADMIN_USERNAME`` and
``LLAMA_ROUTER_ADMIN_PASSWORD``, or a first single sign-on as a member of the
admin group). Nobody can claim the router from the web.
"""

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


def _own_data_path(path: str) -> bool:
    return (
        path.startswith("/api/chat/")
        or path == "/api/my-keys"
        or path.startswith("/api/my-keys/")
    )


SETUP_REQUIRED_DETAIL = (
    "No dashboard accounts: set LLAMA_ROUTER_ADMIN_USERNAME and "
    "LLAMA_ROUTER_ADMIN_PASSWORD and restart llama-router"
)


async def setup_required_response(request: Request) -> Response:
    """The only answer while there are no accounts."""
    if request.url.path.startswith("/api/"):
        return JSONResponse(status_code=503, content={"detail": SETUP_REQUIRED_DETAIL})
    # Imported here: routes imports this module.
    from . import oidc
    from .routes import templates

    cfg = oidc.current_config()
    return templates.TemplateResponse(
        request,
        "setup_required.html",
        {
            "dash_user": None,
            "show_users_tab": False,
            "oidc_enabled": cfg.usable and bool(cfg.admin_group),
            "oidc_display_name": cfg.display_name,
            "oidc_admin_group": cfg.admin_group,
        },
        status_code=503,
    )


def _forbidden(request: Request) -> Response:
    if request.url.path.startswith("/api/"):
        return JSONResponse(status_code=403, content={"detail": "Forbidden"})
    return RedirectResponse(url="/?error=forbidden", status_code=302)


class DashboardAuthMiddleware(BaseHTTPMiddleware):
    async def no_accounts(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        """Answer a request while no dashboard account exists.

        Tests that are not about sign-in replace this to act as an admin.
        """
        return await setup_required_response(request)

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
            return await self.no_accounts(request, call_next)

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
            # Everyone may chat and issue themselves API keys; those APIs
            # only touch the user's own conversations and keys.
            if request.method != "GET" and not _own_data_path(path):
                return _forbidden(request)
            if non_admin_forbidden_get_path(path):
                return _forbidden(request)

        return await call_next(request)
