from __future__ import annotations

import logging
import time

import httpx
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

from ..auth import routing_preferences_from_request
from ..httpx_errors import describe_httpx_error
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


@router.post("/api/generate")
async def generate(request: Request):
    body = await request.json()
    model = body.get("model")
    if not model:
        raise HTTPException(status_code=400, detail="model is required")

    rt = deps.get_router()
    pm = deps.get_pm()
    db = deps.get_db()
    prefs = await routing_preferences_from_request(db, request)

    route_result = await rt.route(model, protocol="ollama", preferences=prefs)
    if not route_result:
        raise HTTPException(
            status_code=404, detail=f"No available provider for model '{model}'"
        )

    provider = route_result.provider
    if route_result.resolved_model != model:
        body["model"] = route_result.resolved_model

    assert provider.id is not None
    body["model"] = await db.get_backend_model_name(provider.id, body["model"])
    client = pm.get_client(provider.id)
    stream = body.get("stream", True)
    start = time.monotonic()

    pm.acquire(provider.id)
    try:
        if stream:

            async def generate_chunks():
                try:
                    async for chunk in client.generate_stream(body):
                        yield chunk
                finally:
                    pm.release(provider.id)

            logged = StreamLogger(
                generate_chunks(),
                db=db,
                provider=provider,
                protocol="ollama",
                endpoint="/api/generate",
                request=request,
                model=model,
                request_body=body,
                start_time=start,
            )
            return StreamingResponse(logged, media_type="application/x-ndjson")

        payload = await client.generate(body)
        pm.release(provider.id)
        import json as _json

        resp_size = len(_json.dumps(payload).encode())
        duration = (time.monotonic() - start) * 1000
        await log_request(
            db,
            provider=provider,
            protocol="ollama",
            endpoint="/api/generate",
            request=request,
            model=model,
            request_body=body,
            response_size=resp_size,
            duration_ms=duration,
        )
        return JSONResponse(content=payload)
    except httpx.HTTPStatusError as exc:
        pm.release(provider.id)
        duration = (time.monotonic() - start) * 1000
        logger.warning(
            "Backend %s returned HTTP %d for /api/generate %s",
            provider.name,
            exc.response.status_code,
            model,
        )
        await log_request(
            db,
            provider=provider,
            protocol="ollama",
            endpoint="/api/generate",
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
        err_detail = str(exc)[:500]
        if isinstance(exc, httpx.HTTPError):
            err_detail = describe_httpx_error(exc)[:500]
            logger.error(
                "Upstream failure on /api/generate (model=%r, provider=%s): %s",
                model,
                provider.name,
                describe_httpx_error(exc),
                exc_info=exc,
            )
        await log_request(
            db,
            provider=provider,
            protocol="ollama",
            endpoint="/api/generate",
            request=request,
            model=model,
            request_body=body,
            response_size=0,
            duration_ms=duration,
            status="error",
            error_detail=err_detail,
        )
        raise
