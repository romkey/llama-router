from __future__ import annotations

import logging
import time

import httpx
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

from ..auth import routing_preferences_from_request
from ..httpx_errors import describe_httpx_error
from ..request_logger import StreamLogger, log_request
from ..v1_client import get_v1_client
from . import deps

logger = logging.getLogger(__name__)
router = APIRouter()


def _forward_backend_error(exc: httpx.HTTPStatusError) -> JSONResponse:
    try:
        body = exc.response.json()
    except Exception:
        body = {"error": exc.response.text or str(exc)}
    return JSONResponse(content=body, status_code=exc.response.status_code)


@router.post("/v1/chat/completions")
@router.post("/chat/completions")
async def chat_completions(request: Request):
    endpoint_path = request.url.path
    body = await request.json()
    model = body.get("model")
    if not model:
        raise HTTPException(status_code=400, detail="model is required")

    rt = deps.get_router()
    pm = deps.get_pm()
    db = deps.get_db()
    prefs = await routing_preferences_from_request(db, request)

    result = await rt.route(model, preferences=prefs)
    if not result:
        raise HTTPException(
            status_code=404, detail=f"No available provider for model '{model}'"
        )

    provider = result.provider
    if result.resolved_model != model:
        body["model"] = result.resolved_model

    assert provider.id is not None
    body["model"] = await db.get_backend_model_name(provider.id, body["model"])
    client = get_v1_client(pm, provider.id)
    stream = body.get("stream", False)
    start = time.monotonic()

    if stream:

        async def generate():
            async with pm.acquire_provider(provider.id):
                async for chunk in client.chat_completions_stream(body):
                    yield chunk

        logged = StreamLogger(
            generate(),
            db=db,
            provider=provider,
            protocol="v1",
            endpoint=endpoint_path,
            request=request,
            model=model,
            request_body=body,
            start_time=start,
        )
        return StreamingResponse(logged, media_type="text/event-stream")

    try:
        async with pm.acquire_provider(provider.id):
            resp = await client.chat_completions(body)
            import json as _json

            resp_size = len(_json.dumps(resp).encode())
            duration = (time.monotonic() - start) * 1000
            await log_request(
                db,
                provider=provider,
                protocol="v1",
                endpoint=endpoint_path,
                request=request,
                model=model,
                request_body=body,
                response_size=resp_size,
                duration_ms=duration,
            )
            return JSONResponse(content=resp)
    except httpx.HTTPStatusError as exc:
        duration = (time.monotonic() - start) * 1000
        logger.warning(
            "Backend %s returned HTTP %d for %s %s",
            provider.name,
            exc.response.status_code,
            endpoint_path,
            model,
        )
        await log_request(
            db,
            provider=provider,
            protocol="v1",
            endpoint=endpoint_path,
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
        duration = (time.monotonic() - start) * 1000
        err_detail = str(exc)[:500]
        if isinstance(exc, httpx.HTTPError):
            err_detail = describe_httpx_error(exc)[:500]
            logger.error(
                "Upstream failure on %s (model=%r, provider=%s): %s",
                endpoint_path,
                model,
                provider.name,
                describe_httpx_error(exc),
            )
        await log_request(
            db,
            provider=provider,
            protocol="v1",
            endpoint=endpoint_path,
            request=request,
            model=model,
            request_body=body,
            response_size=0,
            duration_ms=duration,
            status="error",
            error_detail=err_detail,
        )
        raise
