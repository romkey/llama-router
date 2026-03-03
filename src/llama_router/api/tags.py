from __future__ import annotations

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from . import deps

router = APIRouter()


@router.get("/api/tags")
async def list_tags():
    """Aggregate models from all online providers."""
    db = deps.get_db()
    models = await db.list_all_models()
    return JSONResponse(content={"models": models})


@router.get("/api/version")
async def version():
    from .. import __version__

    return JSONResponse(content={"version": __version__})


@router.get("/api/ps")
async def ps():
    """Show which models are being served across providers."""
    pm = deps.get_pm()
    infos = await pm.list_provider_infos()
    running = []
    for info in infos:
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
