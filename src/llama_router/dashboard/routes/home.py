"""The dashboard page and its status feed."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import (
    HTMLResponse,
    JSONResponse,
)

from ...cloud_presets import PRESETS
from ...config import settings
from ...key_lifecycle import (
    holds_by_default,
    renewal_mode,
    renewal_period,
    with_status,
)
from ...models import RequestLog
from ...router import (
    ENDPOINT_PROTOCOLS,
)
from ...routing_table import MODES
from .. import deps
from ..form_values import _utc_day_start
from ..model_groups import group_rows
from ..pages import (
    _log_count,
    _prune_completed,
    merge_dash_template_ctx,
    template_auth_ctx,
    templates,
)
from .benchmarks import _active_benchmarks, _active_fill_benchmarks
from .providers import _ADD_FORM_COOKIE, _add_provider_prefill, _clear_add_form
from .pulls import _active_fill_pulls, _active_pulls
from .routing import _DEFAULT_EXPLAIN_ENDPOINT
from .workers import _worker_rows

router = APIRouter()


@router.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    pm = deps.get_pm()
    db = deps.get_db()
    infos = await pm.list_provider_infos()
    all_models = await db.list_all_models()
    all_benchmarks = await db.get_all_benchmarks()
    dash = template_auth_ctx(request)
    read_only = dash["read_only"]

    benchmarks_by_model: dict[str, list[dict]] = {}
    for b in all_benchmarks:
        benchmarks_by_model.setdefault(b["model_name"], []).append(b)

    ollama_providers = [i for i in infos if i.provider.is_ollama]
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
    model_aliases = await db.get_model_aliases()
    group_router = deps.get_pm().router
    model_groups = group_rows(
        await db.list_model_groups(),
        group_router.table if group_router is not None else None,
        {i.provider.id: i.provider.name for i in infos if i.provider.id is not None},
    )

    if read_only:
        api_keys: list[Any] = []
        key_lifecycle: dict[str, Any] | None = None
        allow_unauthenticated = False
        key_routing_preview = {
            "balanced": [],
            "latency": [],
            "throughput": [],
            "chaos": [],
        }
        routing_summary = None
        cache_stats = None
        cached_models: set[str] = set()
        workers: list[dict[str, Any]] = []
        worker_link_status: dict[str, Any] | None = None
    else:
        api_keys = with_status(await db.list_api_keys())
        key_lifecycle = {
            "renewal_mode": await renewal_mode(db),
            "holds_by_default": await holds_by_default(db),
            "renewal_days": renewal_period().days,
            "max_key_lifetime_days": settings.max_key_lifetime_days,
        }
        spent = await db.spend_by_key_since(_utc_day_start())
        for k in api_keys:
            k["spent_today_usd"] = spent.get(int(k["id"]), 0.0)
        allow_unauthenticated = await db.get_allow_unauthenticated()
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
            avg_startup = (
                sum(startup_vals) / len(startup_vals) if startup_vals else None
            )
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
        balanced_preview = sorted(
            preview_rows,
            key=lambda r: (
                r["active_requests"],
                (r["avg_startup_ms"] if r["avg_startup_ms"] is not None else 10_000.0)
                + 256 / (r["avg_tps"] or 10.0) * 1000,
            ),
        )[:5]
        chaos_preview = sorted(preview_rows, key=lambda r: r["name"])[:5]

        key_routing_preview = {
            "balanced": balanced_preview,
            "latency": latency_preview,
            "throughput": throughput_preview,
            "chaos": chaos_preview,
        }
        routing_summary = _routing_summary(pm)

        cache = deps.get_cache()
        # Each walks the cache directory: keep that off the event loop.
        cache_stats = await asyncio.to_thread(cache.stats) if cache else None
        if cache_stats is not None:
            cache_stats["enabled"] = settings.cache_enabled
        cached_models = await asyncio.to_thread(cache.cached_models) if cache else set()

        workers = await _worker_rows(db, pm)
        worker_link_status = await deps.get_worker_link().status()

    log_page = 1
    log_pages = 1
    log_total = 0
    log_entries: list[RequestLog] = []
    if dash["show_log"]:
        owner_user_id = dash["dash_user"].id if read_only else None
        log_per_page = 100
        log_total = await _log_count(db, owner_user_id)
        log_pages = max(1, (log_total + log_per_page - 1) // log_per_page)
        try:
            log_page = int(request.query_params.get("log_page", "1"))
        except ValueError:
            log_page = 1
        log_page = max(1, min(log_page, log_pages))
        log_entries = await db.get_request_logs(
            limit=log_per_page,
            offset=(log_page - 1) * log_per_page,
            owner_user_id=owner_user_id,
        )

    provider_model_names = {m["name"] for m in all_models}
    if not read_only:
        cache = deps.get_cache()
        if cache:
            for detail in await asyncio.to_thread(cache.cached_model_details):
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

    resp = templates.TemplateResponse(
        request,
        "dashboard.html",
        merge_dash_template_ctx(
            request,
            {
                "providers": infos,
                "providers_seed": [
                    _provider_status_payload(i, admin=not read_only) for i in infos
                ],
                "cloud_presets": [p.to_dict() for p in PRESETS],
                "hf_models_dir": settings.hf_models_dir,
                "models": all_models,
                "benchmarks_by_model": benchmarks_by_model,
                "ollama_count": ollama_count,
                "model_provider_counts": model_provider_counts,
                "all_provider_counts": all_provider_counts,
                "model_request_counts": model_request_counts,
                "model_fallbacks": model_fallbacks,
                "model_aliases": model_aliases,
                "model_groups": model_groups,
                "group_picker": {
                    "providers": [
                        {
                            "id": i.provider.id,
                            "name": i.provider.name,
                            "models": sorted({m.name for m in i.models}),
                        }
                        for i in infos
                        if i.provider.id is not None
                    ],
                    "groups": {
                        str(g["id"]): {"name": g["name"], "members": g["members"]}
                        for g in model_groups
                    },
                },
                "api_keys": api_keys,
                "key_lifecycle": key_lifecycle,
                "session_affinity_default": (
                    deps.get_pm().router.session_affinity_default
                    if deps.get_pm().router is not None
                    else settings.session_affinity_enabled
                ),
                "routing_summary": routing_summary,
                "allow_unauthenticated": allow_unauthenticated,
                "key_routing_preview": key_routing_preview,
                "log_entries": log_entries,
                "log_page": log_page,
                "log_pages": log_pages,
                "log_total": log_total,
                "cache_stats": cache_stats,
                "cached_models": cached_models,
                "cache_external_host_set": bool(
                    (settings.cache_external_host or "").strip()
                ),
                "workers": workers,
                "worker_link": worker_link_status,
                "worker_hub_enabled": settings.worker_hub_enabled,
                "api_port": settings.api_port,
                "api_on_dashboard_port": settings.api_on_dashboard_port,
                "add_provider_prefill": (
                    _add_provider_prefill(request)
                    if request.query_params.get("add_provider") and not read_only
                    else None
                ),
            },
        ),
    )
    if _ADD_FORM_COOKIE in request.cookies:
        _clear_add_form(request, resp)
    return resp


# Shown instead of a provider address to users who are not administrators:
# addresses reveal internal hosts and ports.
REDACTED_ADDRESS = "(hidden)"


def _redacted_worker(worker: dict[str, Any] | None) -> dict[str, Any] | None:
    if worker is None:
        return None
    return {k: v for k, v in worker.items() if k != "remote_addr"}


def _provider_status_payload(info: Any, *, admin: bool = True) -> dict[str, Any]:
    """One provider as the dashboard sees it; shared by the page and /api/status.

    For non-admins (*admin* false) address URLs and a worker's remote address
    are hidden.
    """
    return {
        "id": info.provider.id,
        "name": info.provider.name,
        "status": info.provider.status.value,
        "status_detail": info.status_detail,
        "protocol": info.provider.protocol.value,
        "flavor": info.provider.flavor,
        "locality": info.provider.locality,
        "is_worker": info.provider.is_worker,
        "model_count": len(info.models),
        "model_names": sorted({m.name for m in info.models}),
        "active_requests": info.active_requests,
        "max_concurrent": info.provider.max_concurrent,
        "queued_requests": info.queued_requests,
        "hot_models": info.hot_models,
        "worker": info.worker if admin else _redacted_worker(info.worker),
        "addresses": [
            {
                "id": a.id,
                "url": a.url if admin else REDACTED_ADDRESS,
                "is_live": a.is_live,
                "is_preferred": a.is_preferred,
            }
            for a in info.addresses
        ],
    }


@router.get("/api/status")
async def api_status(request: Request):
    _prune_completed(_active_pulls)
    _prune_completed(_active_benchmarks)
    _prune_completed(_active_fill_pulls)
    _prune_completed(_active_fill_benchmarks)

    pm = deps.get_pm()
    db = deps.get_db()
    dash_user = getattr(request.state, "dashboard_user", None)
    admin_payload = dash_user is not None and dash_user.is_admin
    infos = await pm.list_provider_infos()
    all_models = await db.list_all_models()
    log_total = 0
    if admin_payload:
        log_total = await _log_count(db)
    elif dash_user is not None:
        log_total = await _log_count(db, dash_user.id)

    providers_data = [
        _provider_status_payload(info, admin=admin_payload) for info in infos
    ]

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

    if not admin_payload:
        active_pulls = {}
        active_benchmarks = {}

    cache = deps.get_cache()
    cache_stats = None
    if admin_payload and cache:
        cache_stats = await asyncio.to_thread(cache.stats)
        if cache_stats is not None:
            cache_stats["enabled"] = settings.cache_enabled

    payload: dict[str, Any] = {
        "provider_count": len(infos),
        "online_count": sum(1 for i in infos if i.provider.status.value != "offline"),
        "busy_count": sum(1 for i in infos if i.active_requests > 0),
        "model_count": len(all_models),
        "log_total": log_total,
        "providers": providers_data,
        "active_pulls": active_pulls,
        "active_benchmarks": active_benchmarks,
    }
    if admin_payload:
        payload["cache"] = cache_stats
        payload["workers"] = await _worker_rows(db, pm)
        payload["worker_link"] = await deps.get_worker_link().status()
    return JSONResponse(payload)


def _routing_summary(pm) -> dict[str, Any] | None:
    """First choice per model and mode from the current routing table."""
    rt = pm.router
    table = rt.table if rt is not None else None
    if table is None:
        return None
    rows = table.first_choices(_DEFAULT_EXPLAIN_ENDPOINT)
    return {
        "version": table.version,
        "built_at": datetime.fromtimestamp(table.built_at, tz=UTC),
        "build_ms": table.build_ms,
        "reasons": list(table.reasons),
        "modes": list(MODES),
        "rows": sorted(rows.items()),
        "endpoints": list(ENDPOINT_PROTOCOLS),
        "default_endpoint": _DEFAULT_EXPLAIN_ENDPOINT,
        "health": provider_health_rows(pm),
    }


def provider_health_rows(pm) -> list[dict[str, Any]]:
    """Health score and health-check latency tally for every provider."""
    rt = pm.router
    inputs = rt.inputs if rt is not None else None
    if inputs is None:
        return []
    out = []
    for p in inputs.providers:
        if p.id is None:
            continue
        health = pm.stats.provider_health(p.id)
        addrs = []
        for a in inputs.addresses.get(p.id, []):
            tally = pm.stats.tally(p.id, a.id) if a.id is not None else None
            addrs.append(
                {
                    "url": a.url,
                    "is_live": a.is_live,
                    "latency": tally.to_dict() if tally else None,
                }
            )
        out.append(
            {
                "provider_id": p.id,
                "provider": p.name,
                "status": p.status.value,
                "health": health.to_dict(),
                "addresses": addrs,
            }
        )
    return out
