from __future__ import annotations

import time

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse

from ..auth import from_hub, hub_provider
from ..model_listing import listed_models
from . import deps

router = APIRouter()


@router.get("/v1/models")
async def list_models(request: Request) -> JSONResponse:
    """Aggregate models from all online providers, plus model groups (OpenAI
    format). A hub sees only the providers shared with it."""
    models = await listed_models(
        deps.get_db(),
        deps.get_router(),
        hub_only=from_hub(request),
        provider_id=hub_provider(request),
    )
    return JSONResponse(
        content={
            "object": "list",
            "data": [
                {
                    "id": m["name"],
                    "object": "model",
                    "created": int(time.time()),
                    "owned_by": "llama-router",
                }
                for m in models
            ],
        }
    )


@router.get("/v1/models/{model_name:path}")
async def get_model(model_name: str, request: Request) -> JSONResponse:
    """Return a single model or model group in OpenAI format."""
    models = await listed_models(
        deps.get_db(),
        deps.get_router(),
        hub_only=from_hub(request),
        provider_id=hub_provider(request),
    )
    for m in models:
        if m["name"] == model_name:
            return JSONResponse(
                content={
                    "id": m["name"],
                    "object": "model",
                    "created": int(time.time()),
                    "owned_by": "llama-router",
                }
            )
    raise HTTPException(status_code=404, detail=f"Model '{model_name}' not found")


@router.get("/health")
async def health() -> JSONResponse:
    return JSONResponse(content={"status": "ok"})
