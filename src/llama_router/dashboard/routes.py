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

from ..auth import generate_api_key, key_hash, key_prefix
from ..config import settings
from ..models import ProviderType, RequestLog
from ..request_logger import log_request
from . import deps

from .. import __version__

logger = logging.getLogger(__name__)

_active_pulls: dict[str, dict] = {}
_active_benchmarks: dict[str, dict] = {}
_active_fill_pulls: dict[str, dict] = {}
_active_fill_benchmarks: dict[str, dict] = {}


def _model_in_api(model: object, api: str, provider_type: ProviderType) -> bool:
    details = getattr(model, "details", None) or {}
    if api == "ollama":
        if "_in_ollama" in details:
            return bool(details.get("_in_ollama"))
        return provider_type in (ProviderType.OLLAMA, ProviderType.BOTH)
    if api == "llamacpp":
        if "_in_llamacpp" in details:
            return bool(details.get("_in_llamacpp"))
        return provider_type in (ProviderType.LLAMACPP, ProviderType.BOTH)
    return False


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
    api_keys = await db.list_api_keys()
    allow_unauthenticated = await db.get_allow_unauthenticated()

    # Preview likely provider preference for each key routing mode.
    bench_by_provider: dict[int, list[dict]] = {}
    for b in all_benchmarks:
        pid = int(b["provider_id"])
        bench_by_provider.setdefault(pid, []).append(b)

    preview_rows: list[dict] = []
    for info in infos:
        pid = info.provider.id
        if pid is None or info.provider.status.value == "offline":
            continue
        pb = bench_by_provider.get(pid, [])
        startup_vals = [
            float(b["startup_time_ms"])
            for b in pb
            if b.get("startup_time_ms") is not None
        ]
        tps_vals = [
            float(b["tokens_per_second"])
            for b in pb
            if b.get("tokens_per_second") is not None
        ]
        avg_startup = sum(startup_vals) / len(startup_vals) if startup_vals else None
        avg_tps = sum(tps_vals) / len(tps_vals) if tps_vals else None
        preview_rows.append(
            {
                "name": info.provider.name,
                "active_requests": info.active_requests,
                "avg_startup_ms": avg_startup,
                "avg_tps": avg_tps,
                "model_count": len(info.models),
            }
        )

    latency_preview = sorted(
        preview_rows,
        key=lambda r: (
            r["active_requests"],
            r["avg_startup_ms"] if r["avg_startup_ms"] is not None else 1_000_000.0,
            -r["model_count"],
        ),
    )[:5]
    throughput_preview = sorted(
        preview_rows,
        key=lambda r: (
            -(r["avg_tps"] if r["avg_tps"] is not None else 0.0),
            r["active_requests"],
            r["avg_startup_ms"] if r["avg_startup_ms"] is not None else 1_000_000.0,
        ),
    )[:5]
    chaos_preview = sorted(preview_rows, key=lambda r: r["name"])[:5]

    key_routing_preview = {
        "latency": latency_preview,
        "throughput": throughput_preview,
        "chaos": chaos_preview,
    }

    log_page = int(request.query_params.get("log_page", "1"))
    log_per_page = 100
    log_total = await db.count_request_logs()
    log_entries = await db.get_request_logs(
        limit=log_per_page, offset=(log_page - 1) * log_per_page
    )
    log_pages = max(1, (log_total + log_per_page - 1) // log_per_page)

    cache = deps.get_cache()
    cache_stats = cache.stats() if cache else None
    if cache_stats is not None:
        cache_stats["enabled"] = settings.cache_enabled
    cached_models = cache.cached_models() if cache else set()

    provider_model_names = {m["name"] for m in all_models}
    if cache:
        for detail in cache.cached_model_details():
            if detail["name"] not in provider_model_names:
                total_size = sum(b["size"] for b in detail["blobs"])
                all_models.append(
                    {
                        "name": detail["name"],
                        "size": total_size,
                        "digest": None,
                        "modified_at": None,
                        "details": {},
                    }
                )
        all_models.sort(key=lambda m: m["name"])

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
            "api_keys": api_keys,
            "allow_unauthenticated": allow_unauthenticated,
            "key_routing_preview": key_routing_preview,
            "log_entries": log_entries,
            "log_page": log_page,
            "log_pages": log_pages,
            "log_total": log_total,
            "cache_stats": cache_stats,
            "cached_models": cached_models,
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
                "hot_models": info.hot_models,
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

    active_benchmarks = {
        bid: {
            "provider_id": b["provider_id"],
            "provider_name": b["provider_name"],
            "model": b["model"],
            "status": b["status"],
            "result": b["result"],
            "error": b["error"],
        }
        for bid, b in _active_benchmarks.items()
        if b["status"] == "running"
    }

    cache = deps.get_cache()
    cache_stats = cache.stats() if cache else None
    if cache_stats is not None:
        cache_stats["enabled"] = settings.cache_enabled

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
            "active_benchmarks": active_benchmarks,
            "cache": cache_stats,
        }
    )


@router.post("/api/keys/generate")
async def api_generate_key(request: Request):
    db = deps.get_db()
    body = await request.json()
    routing_mode = (body.get("routing_mode") or "latency").strip().lower()
    allow_fallback = bool(body.get("allow_fallback", True))
    if routing_mode not in {"latency", "throughput", "chaos"}:
        raise HTTPException(status_code=400, detail="Invalid routing_mode")

    plaintext = generate_api_key()
    await db.create_api_key(
        key_prefix=key_prefix(plaintext),
        key_hash=key_hash(plaintext),
        routing_mode=routing_mode,
        allow_fallback=allow_fallback,
    )
    return JSONResponse(
        {
            "api_key": plaintext,
            "routing_mode": routing_mode,
            "allow_fallback": allow_fallback,
        }
    )


@router.delete("/api/keys/{key_id}")
async def api_delete_key(key_id: int):
    db = deps.get_db()
    await db.delete_api_key(key_id)
    return JSONResponse({"ok": True})


@router.post("/api/auth/allow-unauthenticated")
async def api_set_allow_unauthenticated(request: Request):
    db = deps.get_db()
    body = await request.json()
    allow = bool(body.get("allow", True))
    await db.set_allow_unauthenticated(allow)
    return JSONResponse({"allow_unauthenticated": allow})


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

    cache = deps.get_cache()
    cached_models: set[str] = set()
    cached_only_models: list[dict] = []
    if cache:
        cached_models = cache.cached_models()
        provider_and_missing_names = local_names | {m["name"] for m in missing_models}
        for detail in cache.cached_model_details():
            if detail["name"] not in provider_and_missing_names:
                total_size = sum(b["size"] for b in detail["blobs"])
                cached_only_models.append({"name": detail["name"], "size": total_size})
        cached_only_models.sort(key=lambda m: m["name"])

    return templates.TemplateResponse(
        "provider_detail.html",
        {
            "request": request,
            "info": info,
            "missing_models": missing_models,
            "cached_models": cached_models,
            "cached_only_models": cached_only_models,
        },
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


@router.post("/api/benchmark")
async def api_start_benchmark(request: Request):
    """Start a benchmark as a background task and return immediately."""
    body = await request.json()
    provider_id = int(body["provider_id"])
    model_name = body["model"]
    benchmark_api = (body.get("benchmark_api") or "auto").strip().lower()
    if benchmark_api not in {"auto", "ollama", "llamacpp"}:
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
            entry["result"] = {
                "startup_time_ms": result.startup_time_ms,
                "tokens_per_second": result.tokens_per_second,
                "protocol": result.protocol,
            }
        except Exception as exc:
            entry["status"] = "failed"
            entry["error"] = str(exc)

    asyncio.create_task(_run_benchmark())
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
            entry["result"] = {
                "startup_time_ms": result.startup_time_ms,
                "tokens_per_second": result.tokens_per_second,
                "protocol": result.protocol,
            }
        except Exception as exc:
            entry["status"] = "failed"
            entry["error"] = str(exc)

    asyncio.create_task(_run_benchmark())
    return RedirectResponse(url=f"/providers/{provider_id}", status_code=303)


@router.post("/benchmarks/{benchmark_id}/delete")
async def delete_benchmark(request: Request, benchmark_id: int):
    db = deps.get_db()
    await db.delete_benchmark(benchmark_id)
    referer = request.headers.get("referer", "/")
    return RedirectResponse(url=referer, status_code=303)


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


@router.post("/providers/{provider_id}/delete-model/{model_name:path}")
async def delete_model(provider_id: int, model_name: str):
    import httpx

    pm = deps.get_pm()
    try:
        await pm.delete_remote_model(provider_id, model_name)
    except httpx.HTTPStatusError as exc:
        if exc.response.status_code == 404:
            return RedirectResponse(
                url=f"/providers/{provider_id}?error=Model+{model_name}+not+found+on+backend+(removed+from+local+list)",
                status_code=303,
            )
        raise HTTPException(status_code=500, detail=str(exc))
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
    pull_api = (body.get("pull_api") or "auto").strip().lower()
    if not model:
        raise HTTPException(status_code=400, detail="model is required")
    if pull_api not in {"auto", "ollama", "llamacpp"}:
        raise HTTPException(status_code=400, detail="Invalid pull_api value")
    if pull_api == "llamacpp":
        raise HTTPException(
            status_code=400,
            detail="llama.cpp pull is not supported yet; use ollama",
        )

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
        "pull_api": pull_api,
        "provider_ids": provider_ids,
        "status": "pulling",
        "completed": [],
        "failed": [],
        "progress": "",
    }

    cache_url = _cache_registry_url()
    db = deps.get_db()
    source_ip = request.headers.get("x-forwarded-for", "").split(",")[0].strip() or (
        request.client.host if request.client else "unknown"
    )
    await log_request(
        db,
        provider=None,
        protocol="ollama",
        endpoint="/api/pull",
        request=request,
        source_ip=source_ip,
        model=model,
        request_body={
            "model": model,
            "provider_id": provider_id,
            "pull_api": pull_api,
            "targets": provider_ids,
        },
        duration_ms=0.0,
        status="ok",
    )

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
                        endpoint="/api/pull/provider",
                        source_ip=source_ip,
                        model=model,
                        request_size=0,
                        response_size=0,
                        request_meta=f"pull_api={pull_api}",
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
                        endpoint="/api/pull/provider",
                        source_ip=source_ip,
                        model=model,
                        request_size=0,
                        response_size=0,
                        request_meta=f"pull_api={pull_api}",
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
                "pull_api": p.get("pull_api", "auto"),
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
            "pull_api": entry.get("pull_api", "auto"),
            "status": entry["status"],
            "total": len(entry["provider_ids"]),
            "completed": len(entry["completed"]),
            "failed": len(entry["failed"]),
            "progress": entry.get("progress", ""),
        }
    )


@router.post("/api/pulls/fill-missing")
async def api_fill_missing_pulls():
    """Pull missing models to all Ollama-capable providers."""
    pm = deps.get_pm()
    cache_url = _cache_registry_url()

    infos = await pm.list_provider_infos()
    all_model_names = sorted({m.name for info in infos for m in info.models})

    provider_missing: dict[int, list[str]] = {}
    provider_names: dict[int, str] = {}
    for info in infos:
        provider = info.provider
        if not provider.supports_ollama or provider.id is None:
            continue
        existing_ollama = {
            m.name
            for m in info.models
            if _model_in_api(m, "ollama", provider.provider_type)
        }
        missing = [name for name in all_model_names if name not in existing_ollama]
        if missing:
            provider_missing[provider.id] = missing
            provider_names[provider.id] = provider.name

    job_id = str(uuid.uuid4())
    total = sum(len(v) for v in provider_missing.values())
    _active_fill_pulls[job_id] = {
        "status": "running",
        "total": total,
        "completed": 0,
        "failed": 0,
        "progress": "starting…",
        "errors": [],
    }

    async def _run() -> None:
        job = _active_fill_pulls[job_id]
        lock = asyncio.Lock()

        async def _run_provider(pid: int, models: list[str]) -> None:
            pname = provider_names.get(pid, str(pid))
            try:
                client = pm.get_ollama_client(pid)
            except Exception as exc:
                async with lock:
                    job["failed"] += len(models)
                    job["errors"].append(f"{pname}: no ollama client ({exc})")
                return

            for idx, model in enumerate(models, start=1):
                async with lock:
                    job["progress"] = f"{pname}: pulling {idx}/{len(models)} {model}"
                try:
                    await client.pull_model(model, cache_registry_url=cache_url)
                    async with lock:
                        job["completed"] += 1
                except Exception as exc:
                    async with lock:
                        job["failed"] += 1
                        if len(job["errors"]) < 20:
                            job["errors"].append(f"{pname} {model}: {exc}")
            try:
                await pm.refresh_provider(pid)
            except Exception:
                logger.exception(
                    "Refresh failed after bulk pull for provider %s", pname
                )

        try:
            await asyncio.gather(
                *[
                    _run_provider(pid, models)
                    for pid, models in provider_missing.items()
                ]
            )
            job["status"] = "done"
            if total == 0:
                job["progress"] = "No missing ollama models to pull."
            else:
                job["progress"] = (
                    f"Done: {job['completed']} pulled, {job['failed']} failed."
                )
        except Exception as exc:
            job["status"] = "failed"
            job["progress"] = "Bulk pull failed."
            job["errors"].append(str(exc))
            logger.exception("Bulk fill-missing pull job failed")

    asyncio.create_task(_run())
    return JSONResponse({"job_id": job_id, "status": "running"})


@router.get("/api/pulls/fill-missing/{job_id}")
async def api_fill_missing_pulls_status(job_id: str):
    job = _active_fill_pulls.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Pull fill job not found")
    return JSONResponse({"job_id": job_id, **job})


@router.post("/api/benchmarks/fill-missing")
async def api_fill_missing_benchmarks():
    """Run missing protocol-specific benchmarks for all providers/models."""
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
        if provider.id is None:
            continue
        runs: list[tuple[str, str]] = []
        for m in info.models:
            if provider.supports_ollama and _model_in_api(
                m, "ollama", provider.provider_type
            ):
                key = (provider.id, m.name, "ollama")
                if key not in existing:
                    runs.append((m.name, "ollama"))
            if provider.supports_llamacpp and _model_in_api(
                m, "llamacpp", provider.provider_type
            ):
                key = (provider.id, m.name, "llamacpp")
                if key not in existing:
                    runs.append((m.name, "llamacpp"))
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
            if total == 0:
                job["progress"] = "No missing benchmarks."
            else:
                job["progress"] = (
                    f"Done: {job['completed']} benchmarks, {job['failed']} failed."
                )
        except Exception as exc:
            job["status"] = "failed"
            job["progress"] = "Bulk benchmark job failed."
            job["errors"].append(str(exc))
            logger.exception("Bulk fill-missing benchmark job failed")

    asyncio.create_task(_run())
    return JSONResponse({"job_id": job_id, "status": "running"})


@router.get("/api/benchmarks/fill-missing/{job_id}")
async def api_fill_missing_benchmarks_status(job_id: str):
    job = _active_fill_benchmarks.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Benchmark fill job not found")
    return JSONResponse({"job_id": job_id, **job})


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
    body = await request.json()
    model = body.get("model", "").strip()
    if not model:
        raise HTTPException(status_code=400, detail="model is required")

    cache = deps.get_cache()
    if cache is None:
        raise HTTPException(status_code=400, detail="Cache not available")

    from ..registry_cache.app import precache_model

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

    asyncio.create_task(_run())
    return JSONResponse({"pull_id": pull_id, "status": "pulling"})


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
            logger.debug(
                "delete_remote_model failed for provider %d model %s",
                info.provider.id,
                model,
            )

    await asyncio.gather(*[_delete_one(i) for i in targets])
    return RedirectResponse(url="/#models-pane", status_code=303)
