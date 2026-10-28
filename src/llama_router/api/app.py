"""Ollama-compatible API served on port 11434."""

from __future__ import annotations

import contextlib

from fastapi import FastAPI, Request, WebSocket
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from .. import __version__
from ..config import settings
from ..llamacpp_api.audio import router as v1_audio_router
from ..llamacpp_api.chat import router as v1_chat_router
from ..llamacpp_api.completions import router as v1_completions_router
from ..llamacpp_api.decisions import router as v1_decisions_router
from ..llamacpp_api.embeddings import router as v1_embeddings_router
from ..llamacpp_api.images import router as v1_images_router
from ..llamacpp_api.models import router as v1_models_router
from ..llamacpp_api.responses import router as v1_responses_router
from ..request_logger import log_request
from ..timing import RequestTimingMiddleware
from ..worker_link.protocol import WS_PATH
from . import deps
from .chat import router as chat_router
from .embeddings import router as embeddings_router
from .generate import router as generate_router
from .tags import router as tags_router

app = FastAPI(title="llama-router Ollama API")
app.add_middleware(RequestTimingMiddleware)


@app.get("/health")
async def health() -> JSONResponse:
    return JSONResponse({"status": "ok", "version": __version__})


@app.websocket(WS_PATH)
async def worker_socket(websocket: WebSocket) -> None:
    """Hub endpoint: member routers dial in here with their worker token."""
    if not settings.worker_hub_enabled:
        await websocket.close(code=1008)
        return
    pm = deps.get_pm()
    await pm.worker_hub.serve(websocket, deps.get_db())


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
                protocol="ollama",
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


app.include_router(chat_router)
app.include_router(generate_router)
app.include_router(embeddings_router)
app.include_router(tags_router)

app.include_router(v1_audio_router)
app.include_router(v1_chat_router)
app.include_router(v1_completions_router)
app.include_router(v1_decisions_router)
app.include_router(v1_embeddings_router)
app.include_router(v1_images_router)
app.include_router(v1_models_router)
app.include_router(v1_responses_router)
