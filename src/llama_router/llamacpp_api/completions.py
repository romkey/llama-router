from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

from . import deps

router = APIRouter()


@router.post("/v1/completions")
async def completions(request: Request):
    body = await request.json()
    model = body.get("model")
    if not model:
        raise HTTPException(status_code=400, detail="model is required")

    rt = deps.get_router()
    pm = deps.get_pm()

    provider = await rt.route(model, protocol="llamacpp")
    if not provider:
        raise HTTPException(
            status_code=404, detail=f"No available provider for model '{model}'"
        )

    assert provider.id is not None
    client = pm.get_llamacpp_client(provider.id)
    stream = body.get("stream", False)

    pm.acquire(provider.id)
    try:
        if stream:

            async def generate():
                try:
                    async for chunk in client.completions_stream(body):
                        yield chunk
                finally:
                    pm.release(provider.id)

            return StreamingResponse(generate(), media_type="text/event-stream")
        else:
            result = await client.completions(body)
            pm.release(provider.id)
            return JSONResponse(content=result)
    except Exception:
        pm.release(provider.id)
        raise
