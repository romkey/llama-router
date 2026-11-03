"""Cache tab: the Ollama registry cache."""

from __future__ import annotations

import asyncio
import logging
import time
import uuid

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import (
    JSONResponse,
)

from ...background import spawn
from ...config import settings
from .. import deps
from .pulls import _active_pulls

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/api/cache/status")
async def api_cache_status():
    """Return cache statistics."""
    cache = deps.get_cache()
    if cache is None:
        return JSONResponse({"enabled": False})
    stats = await asyncio.to_thread(cache.stats)
    stats["enabled"] = settings.cache_enabled
    return JSONResponse(stats)


@router.post("/api/cache/toggle")
async def api_cache_toggle():
    """Toggle cache enabled/disabled at runtime."""
    settings.cache_enabled = not settings.cache_enabled
    logger.info(
        "Cache %s via dashboard", "enabled" if settings.cache_enabled else "disabled"
    )
    return JSONResponse({"enabled": settings.cache_enabled})


@router.post("/api/cache/clear")
async def api_cache_clear():
    """Clear the registry cache."""
    cache = deps.get_cache()
    if cache is None:
        raise HTTPException(status_code=400, detail="Cache not available")
    cache.clear()
    return JSONResponse({"status": "cleared"})


@router.post("/api/cache/model")
async def api_cache_model(request: Request):
    """Pre-cache a model by downloading its manifest and blobs."""
    body = await deps.json_body(request)
    model = body.get("model", "").strip()
    if not model:
        raise HTTPException(status_code=400, detail="model is required")

    cache = deps.get_cache()
    if cache is None:
        raise HTTPException(status_code=400, detail="Cache not available")

    from ...registry_cache.app import precache_model

    pull_id = str(uuid.uuid4())
    _active_pulls[pull_id] = {
        "model": model,
        "provider_ids": [],
        "status": "pulling",
        "completed": [],
        "failed": [],
        "progress": "caching…",
    }

    async def _run():
        pull_entry = _active_pulls[pull_id]
        try:

            def _on_progress(msg: str) -> None:
                pull_entry["progress"] = msg

            await precache_model(cache, model, progress_callback=_on_progress)
            pull_entry["completed"].append(0)
            pull_entry["progress"] = "cached"
        except Exception as exc:
            logger.error("Cache model %s failed: %s", model, exc)
            pull_entry["failed"].append(0)
            pull_entry["progress"] = f"FAILED: {exc}"
        pull_entry["status"] = "done"
        pull_entry["completed_at"] = time.monotonic()

    spawn(_run())
    return JSONResponse({"pull_id": pull_id, "status": "pulling"})
