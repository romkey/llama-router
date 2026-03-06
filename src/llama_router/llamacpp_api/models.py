from __future__ import annotations

import time

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse

from . import deps

router = APIRouter()


@router.get("/v1/models")
async def list_models():
    """Aggregate models from all online providers (OpenAI format)."""
    db = deps.get_db()
    models = await db.list_all_models()
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
async def get_model(model_name: str):
    """Return a single model in OpenAI format."""
    db = deps.get_db()
    models = await db.list_all_models()
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
async def health():
    return JSONResponse(content={"status": "ok"})
