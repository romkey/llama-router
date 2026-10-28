"""Dashboard web application served on port 80 (APIs: see ``shared_port``)."""

from __future__ import annotations

import contextlib

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from ..request_logger import log_request
from . import deps
from .decisions import router as decisions_router
from .middleware import DashboardAuthMiddleware
from .model_groups import router as model_groups_router
from .routes import router

app = FastAPI(title="llama-router Dashboard")
app.add_middleware(DashboardAuthMiddleware)


@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(
    request: Request, exc: StarletteHTTPException
) -> JSONResponse:
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
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})


app.include_router(router)
app.include_router(decisions_router)
app.include_router(model_groups_router)
