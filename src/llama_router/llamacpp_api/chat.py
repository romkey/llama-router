from __future__ import annotations

import time

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

from ..request_logger import StreamLogger, log_request
from . import deps

router = APIRouter()


@router.post("/v1/chat/completions")
async def chat_completions(request: Request):
    body = await request.json()
    model = body.get("model")
    if not model:
        raise HTTPException(status_code=400, detail="model is required")

    rt = deps.get_router()
    pm = deps.get_pm()
    db = deps.get_db()

    provider = await rt.route(model, protocol="llamacpp")
    if not provider:
        raise HTTPException(
            status_code=404, detail=f"No available provider for model '{model}'"
        )

    assert provider.id is not None
    client = pm.get_llamacpp_client(provider.id)
    stream = body.get("stream", False)
    start = time.monotonic()

    pm.acquire(provider.id)
    try:
        if stream:

            async def generate():
                try:
                    async for chunk in client.chat_completions_stream(body):
                        yield chunk
                finally:
                    pm.release(provider.id)

            logged = StreamLogger(
                generate(),
                db=db,
                provider=provider,
                protocol="llamacpp",
                endpoint="/v1/chat/completions",
                request=request,
                model=model,
                request_body=body,
                start_time=start,
            )
            return StreamingResponse(logged, media_type="text/event-stream")
        else:
            result = await client.chat_completions(body)
            pm.release(provider.id)
            import json as _json

            resp_size = len(_json.dumps(result).encode())
            duration = (time.monotonic() - start) * 1000
            await log_request(
                db,
                provider=provider,
                protocol="llamacpp",
                endpoint="/v1/chat/completions",
                request=request,
                model=model,
                request_body=body,
                response_size=resp_size,
                duration_ms=duration,
            )
            return JSONResponse(content=result)
    except Exception as exc:
        pm.release(provider.id)
        duration = (time.monotonic() - start) * 1000
        await log_request(
            db,
            provider=provider,
            protocol="llamacpp",
            endpoint="/v1/chat/completions",
            request=request,
            model=model,
            request_body=body,
            response_size=0,
            duration_ms=duration,
            status="error",
            error_detail=str(exc)[:500],
        )
        raise
