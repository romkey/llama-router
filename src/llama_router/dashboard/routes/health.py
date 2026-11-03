"""Health checks and Prometheus metrics."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from fastapi.responses import (
    JSONResponse,
    PlainTextResponse,
)

from ... import __version__
from ... import metrics as metrics_mod
from ...config import settings
from .. import deps

router = APIRouter()


@router.get("/healthz")
async def healthz() -> JSONResponse:
    """Liveness for container health checks: no auth, no database, no backends."""
    return JSONResponse({"status": "ok"})


@router.get("/metrics")
async def metrics() -> PlainTextResponse:
    """Prometheus metrics (see docs/metrics.md).

    Reached with ``Authorization: Bearer <LLAMA_ROUTER_METRICS_TOKEN>`` or a
    dashboard sign-in; ``LLAMA_ROUTER_METRICS_ENABLED=false`` turns it off.
    """
    if not settings.metrics_enabled:
        raise HTTPException(status_code=404, detail="Metrics are disabled")
    pm = deps.get_pm()
    families = metrics_mod.collect(
        await pm.list_provider_infos(),
        pm.stats,
        router=pm.router,
        hub=pm.worker_hub,
        worker_link=await deps.get_worker_link().status(),
        auto_benchmark=deps.auto_benchmark_if_running(),
    )
    return PlainTextResponse(
        metrics_mod.render(families), media_type=metrics_mod.CONTENT_TYPE
    )


@router.get("/health")
async def health():
    """Health check endpoint for container orchestrators."""
    pm = deps.get_pm()
    infos = await pm.list_provider_infos()
    online = sum(1 for i in infos if i.provider.status.value != "offline")
    return JSONResponse(
        {
            "status": "ok",
            "version": __version__,
            "providers": len(infos),
            "providers_online": online,
        }
    )
