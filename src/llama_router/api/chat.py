from __future__ import annotations

import logging
import time

import httpx
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

from ..request_logger import StreamLogger, log_request
from . import deps

logger = logging.getLogger(__name__)
router = APIRouter()


def _forward_backend_error(exc: httpx.HTTPStatusError) -> JSONResponse:
    try:
        body = exc.response.json()
    except Exception:
        body = {"error": exc.response.text or str(exc)}
    return JSONResponse(content=body, status_code=exc.response.status_code)


@router.post("/api/chat")
async def chat(request: Request):
    body = await request.json()
    model = body.get("model")
    if not model:
        raise HTTPException(status_code=400, detail="model is required")

    rt = deps.get_router()
    pm = deps.get_pm()
    db = deps.get_db()

    result = await rt.route(model, protocol="ollama")
    if not result:
        raise HTTPException(
            status_code=404, detail=f"No available provider for model '{model}'"
        )

    provider = result.provider
    if result.resolved_model != model:
        body["model"] = result.resolved_model

    assert provider.id is not None
    body["model"] = await db.get_backend_model_name(provider.id, body["model"])
    client = pm.get_client(provider.id)
    stream = body.get("stream", True)
    start = time.monotonic()

    pm.acquire(provider.id)
    try:
        if stream:

            async def generate():
                try:
                    async for chunk in client.chat_stream(body):
                        yield chunk
                finally:
                    pm.release(provider.id)

            logged = StreamLogger(
                generate(),
                db=db,
                provider=provider,
                protocol="ollama",
                endpoint="/api/chat",
                request=request,
                model=model,
                request_body=body,
                start_time=start,
            )
            return StreamingResponse(logged, media_type="application/x-ndjson")
        else:
            result = await client.chat(body)
            pm.release(provider.id)
            import json as _json

            resp_size = len(_json.dumps(result).encode())
            duration = (time.monotonic() - start) * 1000
            await log_request(
                db,
                provider=provider,
                protocol="ollama",
                endpoint="/api/chat",
                request=request,
                model=model,
                request_body=body,
                response_size=resp_size,
                duration_ms=duration,
            )
            return JSONResponse(content=result)
    except httpx.HTTPStatusError as exc:
        pm.release(provider.id)
        duration = (time.monotonic() - start) * 1000
        logger.warning(
            "Backend %s returned HTTP %d for /api/chat %s",
            provider.name,
            exc.response.status_code,
            model,
        )
        await log_request(
            db,
            provider=provider,
            protocol="ollama",
            endpoint="/api/chat",
            request=request,
            model=model,
            request_body=body,
            response_size=0,
            duration_ms=duration,
            status="error",
            error_detail=f"HTTP {exc.response.status_code}: {exc.response.text[:400]}",
        )
        return _forward_backend_error(exc)
    except Exception as exc:
        pm.release(provider.id)
        duration = (time.monotonic() - start) * 1000
        await log_request(
            db,
            provider=provider,
            protocol="ollama",
            endpoint="/api/chat",
            request=request,
            model=model,
            request_body=body,
            response_size=0,
            duration_ms=duration,
            status="error",
            error_detail=str(exc)[:500],
        )
        raise
