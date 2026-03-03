from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

from . import deps

router = APIRouter()


@router.post("/api/chat")
async def chat(request: Request):
    body = await request.json()
    model = body.get("model")
    if not model:
        raise HTTPException(status_code=400, detail="model is required")

    rt = deps.get_router()
    pm = deps.get_pm()

    provider = await rt.route(model)
    if not provider:
        raise HTTPException(
            status_code=404, detail=f"No available provider for model '{model}'"
        )

    assert provider.id is not None
    client = pm.get_client(provider.id)
    stream = body.get("stream", True)

    pm.acquire(provider.id)
    try:
        if stream:

            async def generate():
                try:
                    async for chunk in client.chat_stream(body):
                        yield chunk
                finally:
                    pm.release(provider.id)

            return StreamingResponse(generate(), media_type="application/x-ndjson")
        else:
            result = await client.chat(body)
            pm.release(provider.id)
            return JSONResponse(content=result)
    except Exception:
        pm.release(provider.id)
        raise
