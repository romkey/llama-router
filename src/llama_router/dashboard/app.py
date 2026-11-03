"""Dashboard web application served on port 80 (APIs: see ``shared_port``)."""

from __future__ import annotations

import contextlib
from http import HTTPStatus

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.middleware.base import RequestResponseEndpoint

from ..limits import BodySizeLimitMiddleware
from ..request_logger import log_request
from . import deps
from .auth_core import _cookie_secure
from .chat import router as chat_router
from .decisions import router as decisions_router
from .management_keys import router as management_keys_router
from .mfa import router as mfa_router
from .middleware import DashboardAuthMiddleware
from .model_groups import router as model_groups_router
from .my_keys import router as my_keys_router
from .pages import merge_dash_template_ctx, templates
from .routes import router

app = FastAPI(title="llama-router Dashboard")
app.add_middleware(DashboardAuthMiddleware)
app.add_middleware(BodySizeLimitMiddleware)

_SECURITY_HEADERS = {
    "X-Frame-Options": "DENY",
    # No script-src yet: the templates still use inline event handlers.
    "Content-Security-Policy": (
        "frame-ancestors 'none'; object-src 'none'; base-uri 'self'"
    ),
    "X-Content-Type-Options": "nosniff",
    "Referrer-Policy": "same-origin",
    "Cache-Control": "no-store",
}


@app.middleware("http")
async def security_headers(
    request: Request, call_next: RequestResponseEndpoint
) -> Response:
    """Add browser hardening headers to every dashboard response.

    Registered last so it wraps the auth middleware and covers its
    redirects and refusals too. The dashboard serves no static files.
    """
    response = await call_next(request)
    response.headers.update(_SECURITY_HEADERS)
    if _cookie_secure(request):
        response.headers["Strict-Transport-Security"] = "max-age=31536000"
    return response


def _wants_html(request: Request) -> bool:
    """HTML for browser pages, JSON for ``/api/*`` and everything else.

    Browsers ask for ``text/html`` when they load a page or submit a form;
    ``fetch()`` sends ``*/*``, and the WebAuthn and other script-only
    routes outside ``/api/`` need JSON back.
    """
    if request.url.path.startswith("/api/"):
        return False
    return "text/html" in request.headers.get("accept", "").lower()


def _error_page(request: Request, exc: StarletteHTTPException) -> Response:
    try:
        reason = HTTPStatus(exc.status_code).phrase
    except ValueError:
        reason = "Error"
    detail = exc.detail if isinstance(exc.detail, str) else reason
    return templates.TemplateResponse(
        request,
        "error.html",
        merge_dash_template_ctx(
            request,
            {
                "status_code": exc.status_code,
                "reason": reason,
                "detail": detail,
                "back_url": "/",
            },
        ),
        status_code=exc.status_code,
        headers=exc.headers,
    )


@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(
    request: Request, exc: StarletteHTTPException
) -> Response:
    if exc.status_code in (400, 401, 403, 404):
        with contextlib.suppress(Exception):
            db = deps.get_db()
            await log_request(
                db,
                provider=None,
                protocol="dashboard",
                endpoint=request.url.path,
                request=request,
                model=None,
                request_body=None,
                response_size=0,
                duration_ms=0.0,
                status="error",
                error_detail=f"HTTP {exc.status_code}: {exc.detail}",
            )
    if _wants_html(request):
        return _error_page(request, exc)
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail},
        headers=exc.headers,
    )


app.include_router(router)
app.include_router(decisions_router)
app.include_router(model_groups_router)
app.include_router(chat_router)
app.include_router(management_keys_router)
app.include_router(my_keys_router)

app.include_router(mfa_router)
