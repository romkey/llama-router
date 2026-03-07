from __future__ import annotations

import logging
import time

import httpx
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

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


@router.post("/v1/responses")
async def responses(request: Request):
    body = await request.json()
    model = body.get("model")
    if not model:
        raise HTTPException(status_code=400, detail="model is required")

    rt = deps.get_router()
    pm = deps.get_pm()
    db = deps.get_db()

    result = await rt.route(model)
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

    pm.acquire(provider.id)
    try:
        if stream:

            async def generate():
                try:
                    async for chunk in client.responses_stream(body):
                        yield chunk
                finally:
                    pm.release(provider.id)

            logged = StreamLogger(
                generate(),
                db=db,
                provider=provider,
                protocol="v1",
                endpoint="/v1/responses",
                request=request,
                model=model,
                request_body=body,
                start_time=start,
            )
            return StreamingResponse(logged, media_type="text/event-stream")
        else:
            resp = await client.responses(body)
            pm.release(provider.id)
            import json as _json

            resp_size = len(_json.dumps(resp).encode())
            duration = (time.monotonic() - start) * 1000
            await log_request(
                db,
                provider=provider,
                protocol="v1",
                endpoint="/v1/responses",
                request=request,
                model=model,
                request_body=body,
                response_size=resp_size,
                duration_ms=duration,
            )
            return JSONResponse(content=resp)
    except httpx.HTTPStatusError as exc:
        pm.release(provider.id)
        duration = (time.monotonic() - start) * 1000
        logger.warning(
            "Backend %s returned HTTP %d for /v1/responses %s",
            provider.name,
            exc.response.status_code,
            model,
        )
        await log_request(
            db,
            provider=provider,
            protocol="v1",
            endpoint="/v1/responses",
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
            protocol="v1",
            endpoint="/v1/responses",
            request=request,
            model=model,
            request_body=body,
            response_size=0,
            duration_ms=duration,
            status="error",
            error_detail=str(exc)[:500],
        )
        raise
