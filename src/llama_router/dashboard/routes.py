from __future__ import annotations

import asyncio
import logging
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.templating import Jinja2Templates

from ..config import settings
from ..models import ProviderType, RequestLog
from . import deps

from .. import __version__

logger = logging.getLogger(__name__)

_active_pulls: dict[str, dict] = {}


def _cache_registry_url() -> str | None:
    """Return the cache registry URL if cache is enabled, else None.

    Uses cache_external_host (the address backends can reach) rather than
    cache_host (the bind address).
    """
    if not settings.cache_enabled:
        return None
    host = settings.cache_external_host
    if not host:
        logger.warning(
            "LLAMA_ROUTER_CACHE_EXTERNAL_HOST is not set — cache pulls will "
            "use 127.0.0.1 which only works if Ollama runs on the same host"
        )
        host = "127.0.0.1"
    return f"http://{host}:{settings.cache_port}"


def _localtime(value: str | datetime | None, fmt: str = "%Y-%m-%d %H:%M:%S") -> str:
    """Jinja2 filter: convert a UTC timestamp to the local timezone (honours TZ)."""
    if value is None:
        return "—"
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return "—"
        for pattern in (
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%dT%H:%M:%S",
            "%Y-%m-%dT%H:%M:%S.%f",
        ):
            try:
                value = datetime.strptime(value, pattern)
                break
            except ValueError:
                continue
        else:
            return str(value)
    if isinstance(value, datetime):
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return value.astimezone().strftime(fmt)
    return str(value)


_TEMPLATE_DIR = Path(__file__).resolve().parent.parent / "templates"
templates = Jinja2Templates(directory=str(_TEMPLATE_DIR))
templates.env.globals["version"] = __version__
templates.env.filters["localtime"] = _localtime

router = APIRouter()


@router.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    pm = deps.get_pm()
    db = deps.get_db()
    infos = await pm.list_provider_infos()
    all_models = await db.list_all_models()
    all_benchmarks = await db.get_all_benchmarks()

    benchmarks_by_model: dict[str, list[dict]] = {}
    for b in all_benchmarks:
        benchmarks_by_model.setdefault(b["model_name"], []).append(b)

    ollama_providers = [i for i in infos if i.provider.supports_ollama]
    ollama_count = len(ollama_providers)
    model_provider_counts: dict[str, int] = {}
    for info in ollama_providers:
        for m in info.models:
            model_provider_counts[m.name] = model_provider_counts.get(m.name, 0) + 1

    all_provider_counts: dict[str, int] = {}
    for info in infos:
        for m in info.models:
            all_provider_counts[m.name] = all_provider_counts.get(m.name, 0) + 1

    model_request_counts = await db.get_model_request_counts()
    model_fallbacks = await db.get_all_model_fallbacks()

    log_page = int(request.query_params.get("log_page", "1"))
    log_per_page = 100
    log_total = await db.count_request_logs()
    log_entries = await db.get_request_logs(
        limit=log_per_page, offset=(log_page - 1) * log_per_page
    )
    log_pages = max(1, (log_total + log_per_page - 1) // log_per_page)

    cache = deps.get_cache()
    cache_stats = cache.stats() if cache else None

    return templates.TemplateResponse(
        "dashboard.html",
        {
            "request": request,
            "providers": infos,
            "models": all_models,
            "benchmarks_by_model": benchmarks_by_model,
            "ollama_count": ollama_count,
            "model_provider_counts": model_provider_counts,
            "all_provider_counts": all_provider_counts,
            "model_request_counts": model_request_counts,
            "model_fallbacks": model_fallbacks,
            "log_entries": log_entries,
            "log_page": log_page,
            "log_pages": log_pages,
            "log_total": log_total,
            "cache_stats": cache_stats,
        },
    )


@router.get("/api/status")
async def api_status():
    pm = deps.get_pm()
    db = deps.get_db()
    infos = await pm.list_provider_infos()
    all_models = await db.list_all_models()
    log_total = await db.count_request_logs()

    providers_data = []
    for info in infos:
        providers_data.append(
            {
                "id": info.provider.id,
                "name": info.provider.name,
                "status": info.provider.status.value,
                "provider_type": info.provider.provider_type.value,
                "model_count": len(info.models),
                "active_requests": info.active_requests,
                "addresses": [
                    {
                        "id": a.id,
                        "url": a.url,
                        "is_live": a.is_live,
                        "is_preferred": a.is_preferred,
                    }
                    for a in info.addresses
                ],
            }
        )

    active_pulls = {
        pid: {
            "model": p["model"],
            "status": p["status"],
            "total": len(p["provider_ids"]),
            "completed": len(p["completed"]),
            "failed": len(p["failed"]),
        }
        for pid, p in _active_pulls.items()
        if p["status"] == "pulling"
    }

    cache = deps.get_cache()
    cache_stats = cache.stats() if cache else None

    return JSONResponse(
        {
            "provider_count": len(infos),
            "online_count": sum(
                1 for i in infos if i.provider.status.value != "offline"
            ),
            "busy_count": sum(1 for i in infos if i.active_requests > 0),
            "model_count": len(all_models),
            "log_total": log_total,
            "providers": providers_data,
            "active_pulls": active_pulls,
            "cache": cache_stats,
        }
    )


@router.get("/providers/{provider_id}", response_class=HTMLResponse)
async def provider_detail(request: Request, provider_id: int):
    pm = deps.get_pm()
    db = deps.get_db()
    info = await pm.get_provider_info(provider_id)
    if not info:
        raise HTTPException(status_code=404, detail="Provider not found")

    local_names = {m.name for m in info.models}
    all_models = await db.list_all_models()
    missing_models = [m for m in all_models if m["name"] not in local_names]

    return templates.TemplateResponse(
        "provider_detail.html",
        {"request": request, "info": info, "missing_models": missing_models},
    )


@router.post("/providers/add")
async def add_provider(
    name: str = Form(...),
    url: str = Form(...),
    provider_type: str = Form("ollama"),
    llamacpp_url: Optional[str] = Form(None),
    machine_type: Optional[str] = Form(None),
    gpu_type: Optional[str] = Form(None),
    gpu_ram: Optional[str] = Form(None),
):
    pm = deps.get_pm()
    ptype = ProviderType(provider_type)
    lcpp_url = llamacpp_url if llamacpp_url else None
    try:
        await pm.add_provider(
            name,
            url,
            ptype,
            lcpp_url,
            machine_type=machine_type or None,
            gpu_type=gpu_type or None,
            gpu_ram=gpu_ram or None,
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    return RedirectResponse(url="/", status_code=303)


@router.post("/providers/{provider_id}/edit")
async def edit_provider(
    provider_id: int,
    name: str = Form(...),
    url: str = Form(...),
    provider_type: str = Form("ollama"),
    llamacpp_url: Optional[str] = Form(None),
    machine_type: Optional[str] = Form(None),
    gpu_type: Optional[str] = Form(None),
    gpu_ram: Optional[str] = Form(None),
):
    pm = deps.get_pm()
    ptype = ProviderType(provider_type)
    lcpp_url = llamacpp_url if llamacpp_url else None
    try:
        await pm.update_provider(
            provider_id,
            name,
            url,
            ptype,
            lcpp_url,
            machine_type=machine_type or None,
            gpu_type=gpu_type or None,
            gpu_ram=gpu_ram or None,
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    return RedirectResponse(url=f"/providers/{provider_id}", status_code=303)


@router.post("/providers/{provider_id}/remove")
async def remove_provider(provider_id: int):
    pm = deps.get_pm()
    await pm.remove_provider(provider_id)
    return RedirectResponse(url="/", status_code=303)


@router.post("/providers/{provider_id}/refresh")
async def refresh_provider(provider_id: int):
    pm = deps.get_pm()
    await pm.refresh_provider(provider_id)
    return RedirectResponse(url=f"/providers/{provider_id}", status_code=303)


@router.post("/providers/{provider_id}/benchmark/{model_name:path}")
async def benchmark_model(provider_id: int, model_name: str):
    pm = deps.get_pm()
    result = await pm.benchmark_provider(provider_id, model_name)
    if not result:
        raise HTTPException(status_code=500, detail="Benchmark failed")
    return RedirectResponse(url=f"/providers/{provider_id}", status_code=303)


@router.post("/providers/{provider_id}/delete-model/{model_name:path}")
async def delete_model(provider_id: int, model_name: str):
    pm = deps.get_pm()
    try:
        await pm.delete_remote_model(provider_id, model_name)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return RedirectResponse(url=f"/providers/{provider_id}", status_code=303)


@router.post("/providers/{provider_id}/addresses/add")
async def add_address(
    provider_id: int,
    url: str = Form(...),
    llamacpp_url: Optional[str] = Form(None),
    is_preferred: Optional[str] = Form(None),
):
    pm = deps.get_pm()
    lcpp = llamacpp_url if llamacpp_url else None
    try:
        await pm.add_address(provider_id, url, lcpp, is_preferred=bool(is_preferred))
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    return RedirectResponse(url=f"/providers/{provider_id}", status_code=303)


@router.post("/providers/{provider_id}/addresses/{address_id}/edit")
async def edit_address(
    provider_id: int,
    address_id: int,
    url: str = Form(...),
    llamacpp_url: Optional[str] = Form(None),
    is_preferred: Optional[str] = Form(None),
):
    pm = deps.get_pm()
    lcpp = llamacpp_url if llamacpp_url else None
    try:
        await pm.update_address(address_id, url, lcpp, is_preferred=bool(is_preferred))
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    return RedirectResponse(url=f"/providers/{provider_id}", status_code=303)


@router.post("/providers/{provider_id}/addresses/{address_id}/remove")
async def remove_address(provider_id: int, address_id: int):
    pm = deps.get_pm()
    await pm.remove_address(address_id)
    return RedirectResponse(url=f"/providers/{provider_id}", status_code=303)


@router.post("/providers/{provider_id}/addresses/{address_id}/toggle-preferred")
async def toggle_preferred(provider_id: int, address_id: int):
    pm = deps.get_pm()
    await pm.toggle_address_preferred(address_id)
    return RedirectResponse(url=f"/providers/{provider_id}", status_code=303)


@router.post("/api/pull")
async def api_pull_model(request: Request):
    """Start a pull as a background task and return immediately."""
    body = await request.json()
    model = body.get("model")
    provider_id = body.get("provider_id")
    if not model:
        raise HTTPException(status_code=400, detail="model is required")

    pm = deps.get_pm()
    pull_id = str(uuid.uuid4())

    if provider_id is not None:
        provider_ids = [int(provider_id)]
    else:
        infos = await pm.list_provider_infos()
        provider_ids = [
            i.provider.id
            for i in infos
            if i.provider.supports_ollama and i.provider.id is not None
        ]

    _active_pulls[pull_id] = {
        "model": model,
        "provider_ids": provider_ids,
        "status": "pulling",
        "completed": [],
        "failed": [],
        "progress": "",
    }

    cache_url = _cache_registry_url()
    db = deps.get_db()

    async def _run_pull():
        pull_entry = _active_pulls[pull_id]
        for idx, pid in enumerate(provider_ids):
            provider = await db.get_provider(pid)
            pname = provider.name if provider else str(pid)
            prefix = f"[{idx + 1}/{len(provider_ids)}] {pname}"
            pull_entry["progress"] = f"{prefix}: starting…"
            logger.info("Pull %s starting on provider %s (id=%d)", model, pname, pid)
            start = time.monotonic()

            def _on_progress(info: dict, _pfx: str = prefix) -> None:
                text = info.get("status", "")
                pct = info.get("percent")
                if pct is not None:
                    pull_entry["progress"] = f"{_pfx}: {text} {pct}%"
                else:
                    pull_entry["progress"] = f"{_pfx}: {text}"

            try:
                client = pm.get_ollama_client(pid)
                await client.pull_model(
                    model,
                    cache_registry_url=cache_url,
                    progress_callback=_on_progress,
                )
                await pm.refresh_provider(pid)
                pull_entry["completed"].append(pid)
                duration = (time.monotonic() - start) * 1000
                pull_entry["progress"] = f"{prefix}: done ({duration / 1000:.0f}s)"
                logger.info(
                    "Pull %s succeeded on provider %s in %.1fs",
                    model,
                    pname,
                    duration / 1000,
                )
                await db.save_request_log(
                    RequestLog(
                        provider_id=pid,
                        provider_name=pname,
                        protocol="ollama",
                        endpoint="/api/pull",
                        model=model,
                        duration_ms=duration,
                        status="ok",
                    )
                )
            except Exception as exc:
                duration = (time.monotonic() - start) * 1000
                logger.error(
                    "Pull %s FAILED on provider %s after %.1fs: %s",
                    model,
                    pname,
                    duration / 1000,
                    exc,
                )
                pull_entry["failed"].append(pid)
                pull_entry["progress"] = f"{prefix}: FAILED"
                await db.save_request_log(
                    RequestLog(
                        provider_id=pid,
                        provider_name=pname,
                        protocol="ollama",
                        endpoint="/api/pull",
                        model=model,
                        duration_ms=duration,
                        status="error",
                        error_detail=str(exc)[:500],
                    )
                )
        pull_entry["status"] = "done"

    asyncio.create_task(_run_pull())

    return JSONResponse({"pull_id": pull_id, "status": "pulling"})


@router.get("/api/pulls")
async def api_active_pulls():
    """Return all active/recent pull operations."""
    return JSONResponse(
        {
            pid: {
                "model": p["model"],
                "status": p["status"],
                "total": len(p["provider_ids"]),
                "completed": len(p["completed"]),
                "failed": len(p["failed"]),
                "progress": p.get("progress", ""),
            }
            for pid, p in _active_pulls.items()
        }
    )


@router.get("/api/pulls/{pull_id}")
async def api_pull_status(pull_id: str):
    """Check status of a specific pull."""
    entry = _active_pulls.get(pull_id)
    if not entry:
        raise HTTPException(status_code=404, detail="Pull not found")
    return JSONResponse(
        {
            "pull_id": pull_id,
            "model": entry["model"],
            "status": entry["status"],
            "total": len(entry["provider_ids"]),
            "completed": len(entry["completed"]),
            "failed": len(entry["failed"]),
            "progress": entry.get("progress", ""),
        }
    )


@router.post("/providers/{provider_id}/pull")
async def pull_model_legacy(provider_id: int, model: str = Form(...)):
    """Legacy form-based pull — redirects immediately, pull runs in background."""
    pm = deps.get_pm()
    db = deps.get_db()
    pull_id = str(uuid.uuid4())
    _active_pulls[pull_id] = {
        "model": model,
        "provider_ids": [provider_id],
        "status": "pulling",
        "completed": [],
        "failed": [],
        "progress": "",
    }
    cache_url = _cache_registry_url()

    async def _run():
        pull_entry = _active_pulls[pull_id]
        provider = await db.get_provider(provider_id)
        pname = provider.name if provider else str(provider_id)
        pull_entry["progress"] = f"{pname}: starting…"
        logger.info(
            "Pull %s starting on provider %s (id=%d)", model, pname, provider_id
        )
        start = time.monotonic()

        def _on_progress(info: dict) -> None:
            text = info.get("status", "")
            pct = info.get("percent")
            if pct is not None:
                pull_entry["progress"] = f"{pname}: {text} {pct}%"
            else:
                pull_entry["progress"] = f"{pname}: {text}"

        try:
            client = pm.get_ollama_client(provider_id)
            await client.pull_model(
                model,
                cache_registry_url=cache_url,
                progress_callback=_on_progress,
            )
            await pm.refresh_provider(provider_id)
            pull_entry["completed"].append(provider_id)
            duration = (time.monotonic() - start) * 1000
            pull_entry["progress"] = f"{pname}: done ({duration / 1000:.0f}s)"
            logger.info(
                "Pull %s succeeded on provider %s in %.1fs",
                model,
                pname,
                duration / 1000,
            )
            await db.save_request_log(
                RequestLog(
                    provider_id=provider_id,
                    provider_name=pname,
                    protocol="ollama",
                    endpoint="/api/pull",
                    model=model,
                    duration_ms=duration,
                    status="ok",
                )
            )
        except Exception as exc:
            duration = (time.monotonic() - start) * 1000
            logger.error(
                "Pull %s FAILED on provider %s after %.1fs: %s",
                model,
                pname,
                duration / 1000,
                exc,
            )
            pull_entry["failed"].append(provider_id)
            pull_entry["progress"] = f"{pname}: FAILED"
            await db.save_request_log(
                RequestLog(
                    provider_id=provider_id,
                    provider_name=pname,
                    protocol="ollama",
                    endpoint="/api/pull",
                    model=model,
                    duration_ms=duration,
                    status="error",
                    error_detail=str(exc)[:500],
                )
            )
        pull_entry["status"] = "done"

    asyncio.create_task(_run())
    return RedirectResponse(url=f"/providers/{provider_id}", status_code=303)


@router.post("/models/pull-all")
async def pull_model_all_legacy(model: str = Form(...)):
    """Legacy form-based pull-all — redirects immediately, pull runs in background."""
    pm = deps.get_pm()
    db = deps.get_db()
    infos = await pm.list_provider_infos()
    provider_ids = [
        i.provider.id
        for i in infos
        if i.provider.supports_ollama and i.provider.id is not None
    ]
    pull_id = str(uuid.uuid4())
    _active_pulls[pull_id] = {
        "model": model,
        "provider_ids": provider_ids,
        "status": "pulling",
        "completed": [],
        "failed": [],
        "progress": "",
    }
    cache_url = _cache_registry_url()

    async def _run():
        pull_entry = _active_pulls[pull_id]
        for idx, pid in enumerate(provider_ids):
            provider = await db.get_provider(pid)
            pname = provider.name if provider else str(pid)
            prefix = f"[{idx + 1}/{len(provider_ids)}] {pname}"
            pull_entry["progress"] = f"{prefix}: starting…"
            logger.info("Pull %s starting on provider %s (id=%d)", model, pname, pid)
            start = time.monotonic()

            def _on_progress(info: dict, _pfx: str = prefix) -> None:
                text = info.get("status", "")
                pct = info.get("percent")
                if pct is not None:
                    pull_entry["progress"] = f"{_pfx}: {text} {pct}%"
                else:
                    pull_entry["progress"] = f"{_pfx}: {text}"

            try:
                client = pm.get_ollama_client(pid)
                await client.pull_model(
                    model,
                    cache_registry_url=cache_url,
                    progress_callback=_on_progress,
                )
                await pm.refresh_provider(pid)
                pull_entry["completed"].append(pid)
                duration = (time.monotonic() - start) * 1000
                pull_entry["progress"] = f"{prefix}: done ({duration / 1000:.0f}s)"
                logger.info(
                    "Pull %s succeeded on provider %s in %.1fs",
                    model,
                    pname,
                    duration / 1000,
                )
                await db.save_request_log(
                    RequestLog(
                        provider_id=pid,
                        provider_name=pname,
                        protocol="ollama",
                        endpoint="/api/pull",
                        model=model,
                        duration_ms=duration,
                        status="ok",
                    )
                )
            except Exception as exc:
                duration = (time.monotonic() - start) * 1000
                logger.error(
                    "Pull %s FAILED on provider %s after %.1fs: %s",
                    model,
                    pname,
                    duration / 1000,
                    exc,
                )
                pull_entry["failed"].append(pid)
                pull_entry["progress"] = f"{prefix}: FAILED"
                await db.save_request_log(
                    RequestLog(
                        provider_id=pid,
                        provider_name=pname,
                        protocol="ollama",
                        endpoint="/api/pull",
                        model=model,
                        duration_ms=duration,
                        status="error",
                        error_detail=str(exc)[:500],
                    )
                )
        pull_entry["status"] = "done"

    asyncio.create_task(_run())
    return RedirectResponse(url="/#models-pane", status_code=303)


@router.get("/api/cache/status")
async def api_cache_status():
    """Return cache statistics."""
    cache = deps.get_cache()
    if cache is None:
        return JSONResponse({"enabled": False})
    stats = cache.stats()
    stats["enabled"] = True
    return JSONResponse(stats)


@router.post("/api/cache/clear")
async def api_cache_clear():
    """Clear the registry cache."""
    cache = deps.get_cache()
    if cache is None:
        raise HTTPException(status_code=400, detail="Cache not enabled")
    cache.clear()
    return JSONResponse({"status": "cleared"})


@router.post("/api/fallbacks")
async def set_fallback(request: Request):
    """Set or update a model fallback. Body: {model: str, fallback: str}."""
    body = await request.json()
    model = body.get("model", "").strip()
    fallback = body.get("fallback", "").strip()
    if not model or not fallback:
        raise HTTPException(status_code=400, detail="model and fallback are required")
    if model == fallback:
        raise HTTPException(
            status_code=400, detail="A model cannot be its own fallback"
        )
    db = deps.get_db()
    await db.set_model_fallback(model, fallback)
    return JSONResponse({"status": "ok", "model": model, "fallback": fallback})


@router.delete("/api/fallbacks/{model_name:path}")
async def remove_fallback(model_name: str):
    """Remove a model fallback."""
    db = deps.get_db()
    await db.remove_model_fallback(model_name)
    return JSONResponse({"status": "ok"})


@router.get("/api/fallbacks")
async def list_fallbacks():
    """Return all configured model fallbacks."""
    db = deps.get_db()
    fallbacks = await db.get_all_model_fallbacks()
    return JSONResponse(fallbacks)


@router.post("/models/delete-all")
async def delete_model_all_providers(model: str = Form(...)):
    pm = deps.get_pm()
    infos = await pm.list_provider_infos()
    targets = [
        i
        for i in infos
        if i.provider.supports_ollama and any(m.name == model for m in i.models)
    ]

    async def _delete_one(info):
        assert info.provider.id is not None
        try:
            await pm.delete_remote_model(info.provider.id, model)
        except Exception:
            pass

    await asyncio.gather(*[_delete_one(i) for i in targets])
    return RedirectResponse(url="/#models-pane", status_code=303)
