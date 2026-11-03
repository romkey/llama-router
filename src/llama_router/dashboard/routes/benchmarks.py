"""Benchmarks tab and auto-benchmarking."""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from fastapi.encoders import jsonable_encoder
from fastapi.responses import (
    JSONResponse,
    RedirectResponse,
)

from ...background import spawn
from .. import deps
from ..pages import _provider_url, _same_origin_referer_path

logger = logging.getLogger(__name__)
router = APIRouter()
_active_benchmarks: dict[str, dict] = {}
_active_fill_benchmarks: dict[str, dict] = {}


@router.post("/api/benchmark")
async def api_start_benchmark(request: Request):
    """Start a benchmark as a background task and return immediately."""
    body = await deps.json_body(request)
    try:
        provider_id = int(body["provider_id"])
    except (KeyError, TypeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail="provider_id is required") from exc
    model_name = body.get("model")
    if not model_name or not isinstance(model_name, str):
        raise HTTPException(status_code=400, detail="model is required")
    benchmark_api = (body.get("benchmark_api") or "auto").strip().lower()
    if benchmark_api not in {"auto", "ollama", "openai", "llamacpp"}:
        raise HTTPException(status_code=400, detail="Invalid benchmark_api value")

    pm = deps.get_pm()
    db = deps.get_db()
    provider = await db.get_provider(provider_id)
    if not provider:
        raise HTTPException(status_code=404, detail="Provider not found")

    bench_id = str(uuid.uuid4())
    _active_benchmarks[bench_id] = {
        "provider_id": provider_id,
        "provider_name": provider.name,
        "model": model_name,
        "benchmark_api": benchmark_api,
        "status": "running",
        "result": None,
        "error": None,
    }

    async def _run_benchmark():
        entry = _active_benchmarks[bench_id]
        try:
            result = await pm.benchmark_provider(
                provider_id, model_name, benchmark_api=benchmark_api
            )
            entry["status"] = "done"
            entry["completed_at"] = time.monotonic()
            entry["result"] = {
                "startup_time_ms": result.startup_time_ms,
                "tokens_per_second": result.tokens_per_second,
                "protocol": result.protocol,
            }
        except Exception as exc:
            entry["status"] = "failed"
            entry["completed_at"] = time.monotonic()
            entry["error"] = str(exc)

    spawn(_run_benchmark())
    return JSONResponse({"bench_id": bench_id, "status": "running"})


@router.get("/api/benchmarks/{bench_id}")
async def api_benchmark_status(bench_id: str):
    entry = _active_benchmarks.get(bench_id)
    if not entry:
        raise HTTPException(status_code=404, detail="Benchmark not found")
    return JSONResponse(
        {
            "bench_id": bench_id,
            "provider_id": entry["provider_id"],
            "provider_name": entry["provider_name"],
            "model": entry["model"],
            "benchmark_api": entry.get("benchmark_api", "auto"),
            "status": entry["status"],
            "result": entry["result"],
            "error": entry["error"],
        }
    )


@router.post("/providers/{provider_id}/benchmark/{model_name:path}")
async def benchmark_model(provider_id: int, model_name: str):
    """Legacy form-based benchmark start — redirects back to provider page."""
    pm = deps.get_pm()
    db = deps.get_db()
    provider = await db.get_provider(provider_id)
    if not provider:
        raise HTTPException(status_code=404, detail="Provider not found")

    bench_id = str(uuid.uuid4())
    _active_benchmarks[bench_id] = {
        "provider_id": provider_id,
        "provider_name": provider.name,
        "model": model_name,
        "status": "running",
        "result": None,
        "error": None,
    }

    async def _run_benchmark():
        entry = _active_benchmarks[bench_id]
        try:
            result = await pm.benchmark_provider(provider_id, model_name)
            entry["status"] = "done"
            entry["completed_at"] = time.monotonic()
            entry["result"] = {
                "startup_time_ms": result.startup_time_ms,
                "tokens_per_second": result.tokens_per_second,
                "protocol": result.protocol,
            }
        except Exception as exc:
            entry["status"] = "failed"
            entry["completed_at"] = time.monotonic()
            entry["error"] = str(exc)

    spawn(_run_benchmark())
    return RedirectResponse(url=_provider_url(provider_id), status_code=303)


@router.post("/benchmarks/{benchmark_id}/delete")
async def delete_benchmark(request: Request, benchmark_id: int):
    db = deps.get_db()
    await db.delete_benchmark(benchmark_id)
    dest = _same_origin_referer_path(request, "/#benchmarks-pane")
    return RedirectResponse(url=dest, status_code=303)


@router.post("/benchmarks/delete-model/{model_name:path}")
async def delete_benchmarks_for_model(model_name: str):
    db = deps.get_db()
    await db.delete_benchmarks_for_model(model_name)
    return RedirectResponse(url="/#benchmarks-pane", status_code=303)


@router.post("/benchmarks/delete-all")
async def delete_all_benchmarks():
    db = deps.get_db()
    await db.delete_all_benchmarks()
    return RedirectResponse(url="/#benchmarks-pane", status_code=303)


@router.get("/api/auto-benchmark")
async def api_auto_benchmark_status():
    """Idle auto-benchmark settings, what is running, what is due, recent results."""
    return JSONResponse(jsonable_encoder(await deps.get_auto_benchmark().status()))


@router.post("/api/auto-benchmark")
async def api_auto_benchmark_configure(request: Request):
    """Body: any of ``enabled`` (bool), ``idle_minutes``, ``max_age_days``."""
    body = await deps.json_body(request)
    kwargs: dict[str, Any] = {}
    try:
        if "enabled" in body:
            kwargs["enabled"] = bool(body["enabled"])
        if body.get("idle_minutes") not in (None, ""):
            kwargs["idle_seconds"] = round(float(body["idle_minutes"]) * 60)
        if body.get("max_age_days") not in (None, ""):
            kwargs["max_age_days"] = float(body["max_age_days"])
        cfg = await deps.get_auto_benchmark().set_config(**kwargs)
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return JSONResponse(cfg.to_dict())


@router.post("/api/auto-benchmark/check")
async def api_auto_benchmark_check():
    """Look for idle providers with due benchmarks now instead of at the next tick."""
    ab = deps.get_auto_benchmark()
    await ab.run_once()
    return JSONResponse(jsonable_encoder(await ab.status()))


@router.post("/api/benchmarks/fill-missing")
async def api_fill_missing_benchmarks():
    """Run missing benchmarks for every provider and model.

    Providers that opt out of automatic benchmarks (cloud ones by default,
    where each run costs money) are left out; benchmark those one by one.
    """
    pm = deps.get_pm()
    db = deps.get_db()
    infos = await pm.list_provider_infos()
    all_benchmarks = await db.get_all_benchmarks()
    existing = {
        (int(b["provider_id"]), b["model_name"], (b.get("protocol") or "").lower())
        for b in all_benchmarks
    }

    provider_runs: dict[int, list[tuple[str, str]]] = {}
    provider_names: dict[int, str] = {}
    for info in infos:
        provider = info.provider
        if provider.id is None or not provider.auto_benchmarks:
            continue
        api = provider.protocol.value
        runs: list[tuple[str, str]] = [
            (m.name, api)
            for m in info.models
            if (provider.id, m.name, api) not in existing
        ]
        if runs:
            provider_runs[provider.id] = runs
            provider_names[provider.id] = provider.name

    job_id = str(uuid.uuid4())
    total = sum(len(v) for v in provider_runs.values())
    _active_fill_benchmarks[job_id] = {
        "status": "running",
        "total": total,
        "completed": 0,
        "failed": 0,
        "progress": "starting…",
        "errors": [],
    }

    async def _run() -> None:
        job = _active_fill_benchmarks[job_id]
        lock = asyncio.Lock()

        async def _run_provider(pid: int, runs: list[tuple[str, str]]) -> None:
            pname = provider_names.get(pid, str(pid))
            for idx, (model, protocol) in enumerate(runs, start=1):
                async with lock:
                    job["progress"] = (
                        f"{pname}: benchmark {idx}/{len(runs)} {model} ({protocol})"
                    )
                try:
                    await pm.benchmark_provider(pid, model, benchmark_api=protocol)
                    async with lock:
                        job["completed"] += 1
                except Exception as exc:
                    async with lock:
                        job["failed"] += 1
                        if len(job["errors"]) < 20:
                            job["errors"].append(f"{pname} {model} ({protocol}): {exc}")

        try:
            await asyncio.gather(
                *[_run_provider(pid, runs) for pid, runs in provider_runs.items()]
            )
            job["status"] = "done"
            job["completed_at"] = time.monotonic()
            if total == 0:
                job["progress"] = "No missing benchmarks."
            else:
                job["progress"] = (
                    f"Done: {job['completed']} benchmarks, {job['failed']} failed."
                )
        except Exception as exc:
            job["status"] = "failed"
            job["completed_at"] = time.monotonic()
            job["progress"] = "Bulk benchmark job failed."
            job["errors"].append(str(exc))
            logger.exception("Bulk fill-missing benchmark job failed")

    spawn(_run())
    return JSONResponse({"job_id": job_id, "status": "running"})


@router.get("/api/benchmarks/fill-missing/{job_id}")
async def api_fill_missing_benchmarks_status(job_id: str):
    job = _active_fill_benchmarks.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Benchmark fill job not found")
    return JSONResponse({"job_id": job_id, **job})
