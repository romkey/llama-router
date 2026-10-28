from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from ..auth import from_hub
from . import deps

router = APIRouter()


@router.get("/api/tags")
async def list_tags(request: Request) -> JSONResponse:
    """Aggregate models from all online providers (a hub sees only the
    providers shared with it)."""
    db = deps.get_db()
    models = await db.list_all_models(hub_only=from_hub(request))
    return JSONResponse(content={"models": models})


@router.get("/api/version")
async def version() -> JSONResponse:
    from .. import __version__

    return JSONResponse(content={"version": __version__})


@router.get("/api/ps")
async def ps(request: Request) -> JSONResponse:
    """Show which models are being served across providers (a hub sees only
    the providers shared with it)."""
    pm = deps.get_pm()
    infos = await pm.list_provider_infos()
    hub = from_hub(request)
    running = []
    for info in infos:
        if hub and not info.provider.hub_shared:
            continue
        if info.active_requests > 0:
            for m in info.models:
                running.append(
                    {
                        "name": m.name,
                        "size": m.size,
                        "digest": m.digest,
                        "details": m.details or {},
                    }
                )
    return JSONResponse(content={"models": running})
