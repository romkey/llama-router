from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse

from . import deps

router = APIRouter()


@router.post("/v1/embeddings")
async def embeddings(request: Request):
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
    pm.acquire(provider.id)
    try:
        result = await client.embeddings(body)
        return JSONResponse(content=result)
    finally:
        pm.release(provider.id)
