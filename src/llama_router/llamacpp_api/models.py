from __future__ import annotations

import time

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from . import deps
from .. import __version__

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


@router.get("/health")
async def health():
    return JSONResponse(content={"status": "ok"})
