"""llama.cpp / OpenAI-compatible API served on port 8080."""

from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from .. import __version__
from ..request_logger import log_request

from . import deps
from .audio import router as audio_router
from .chat import router as chat_router
from .completions import router as completions_router
from .embeddings import router as embeddings_router
from .images import router as images_router
from .models import router as models_router
from .responses import router as responses_router

app = FastAPI(title="llama-router llama.cpp API")


@app.get("/health")
async def health():
    return JSONResponse({"status": "ok", "version": __version__})


@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    if exc.status_code in (400, 401, 403, 404):
        try:
            db = deps.get_db()
            await log_request(
                db,
                provider=None,
                protocol="v1",
                endpoint=request.url.path,
                request=request,
                model=None,
                request_body=None,
                response_size=0,
                duration_ms=0.0,
                status="error",
                error_detail=f"HTTP {exc.status_code}: {exc.detail}",
            )
        except Exception:
            pass
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})


app.include_router(audio_router)
app.include_router(chat_router)
app.include_router(completions_router)
app.include_router(embeddings_router)
app.include_router(images_router)
app.include_router(models_router)
app.include_router(responses_router)
