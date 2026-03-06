from __future__ import annotations

import json
import time

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse

from ..request_logger import log_request
from . import deps

router = APIRouter()


async def _handle_embedding(request: Request, endpoint: str, method: str):
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
    client = pm.get_client(provider.id)
    start = time.monotonic()
    pm.acquire(provider.id)
    try:
        result = await getattr(client, method)(body)
        resp_size = len(json.dumps(result).encode())
        duration = (time.monotonic() - start) * 1000
        await log_request(
            db,
            provider=provider,
            protocol="ollama",
            endpoint=endpoint,
            request=request,
            model=model,
            request_body=body,
            response_size=resp_size,
            duration_ms=duration,
        )
        return JSONResponse(content=result)
    except Exception as exc:
        duration = (time.monotonic() - start) * 1000
        await log_request(
            db,
            provider=provider,
            protocol="ollama",
            endpoint=endpoint,
            request=request,
            model=model,
            request_body=body,
            response_size=0,
            duration_ms=duration,
            status="error",
            error_detail=str(exc)[:500],
        )
        raise
    finally:
        pm.release(provider.id)


@router.post("/api/embeddings")
async def embeddings(request: Request):
    return await _handle_embedding(request, "/api/embeddings", "embeddings")


@router.post("/api/embed")
async def embed(request: Request):
    return await _handle_embedding(request, "/api/embed", "embed")
