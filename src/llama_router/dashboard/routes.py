from __future__ import annotations

import asyncio
import ipaddress
import logging
import secrets
import socket
import time
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional
from urllib.parse import quote

import httpx
from fastapi import APIRouter, Form, HTTPException, Request
from pydantic import BaseModel
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.templating import Jinja2Templates

from ..auth import generate_api_key, key_hash, key_prefix
from ..config import settings
from ..httpx_errors import describe_httpx_error
from ..models import ProviderType, RequestLog
from . import deps

from .. import __version__
from ..wireguard_config import generate_wireguard_private_key, is_valid_wg_key_b64
from ..wireguard_manager import debug_peer_connectivity
from ..wireguard_sync import sync_wireguard_config_to_disk

from .auth_core import (
    AUTH_COOKIE,
    SESSION_MAX_AGE,
    hash_password,
    normalize_username,
    sign_session,
    verify_password,
)
from .middleware import invalidate_dashboard_user_count_cache

logger = logging.getLogger(__name__)

_active_pulls: dict[str, dict] = {}
_active_benchmarks: dict[str, dict] = {}
_active_fill_pulls: dict[str, dict] = {}
_active_fill_benchmarks: dict[str, dict] = {}


def _prune_completed(store: dict, max_age_seconds: float = 600) -> None:
    now = time.monotonic()
    stale = [
        k
        for k, v in store.items()
        if v.get("status") in ("done", "failed")
        and v.get("completed_at") is not None
        and now - v["completed_at"] > max_age_seconds
    ]
    for k in stale:
        del store[k]


async def _pull_one_ollama(
    *,
    pid: int,
    idx: int,
    total: int,
    model: str,
    pull_api: str,
    cache_url: str | None,
    pm: Any,
    db: Any,
    pull_entry: dict,
    progress_lock: asyncio.Lock,
    source_ip: str | None = None,
    log_endpoint: str = "/api/pull/provider",
    request_meta: str | None = None,
) -> None:
    """Pull *model* on a single Ollama provider; updates *pull_entry* under *progress_lock*."""
    provider = await db.get_provider(pid)
    pname = provider.name if provider else str(pid)
    prefix = f"[{idx + 1}/{total}] {pname}" if total > 1 else pname

    async with progress_lock:
        pull_entry["progress"] = f"{prefix}: starting…"
    logger.info("Pull %s starting on provider %s (id=%d)", model, pname, pid)
    start = time.monotonic()
    last_total_bytes: list[int] = [0]

    def _on_progress(info: dict, _pfx: str = prefix) -> None:
        text = info.get("status", "")
        pct = info.get("percent")
        completed = info.get("completed") or 0
        total = info.get("total") or 0
        if total:
            last_total_bytes[0] = total
        elapsed = time.monotonic() - start
        speed_str = ""
        if elapsed > 0 and completed > 0:
            speed_mbps = (completed / (1024 * 1024)) / elapsed
            speed_str = f" {speed_mbps:.1f} MB/s"
        if pct is not None:
            msg = f"{_pfx}: {text} {pct}%{speed_str}"
        else:
            msg = f"{_pfx}: {text}{speed_str}"

        async def _write() -> None:
            async with progress_lock:
                pull_entry["progress"] = msg

        try:
            asyncio.get_running_loop().create_task(_write())
        except RuntimeError:
            pull_entry["progress"] = msg

    try:
        client = pm.get_ollama_client(pid)
        await client.pull_model(
            model,
            cache_registry_url=cache_url,
            progress_callback=_on_progress,
        )
        await pm.refresh_provider(pid)
        duration = (time.monotonic() - start) * 1000
        total_bytes = last_total_bytes[0]
        speed_mbps = (
            (total_bytes / (1024 * 1024)) / (duration / 1000)
            if total_bytes and duration > 0
            else 0
        )
        done_msg = f"{prefix}: done ({duration / 1000:.0f}s"
        if speed_mbps > 0:
            done_msg += f", {speed_mbps:.1f} MB/s"
        done_msg += ")"
        async with progress_lock:
            pull_entry["completed"].append(pid)
            pull_entry["progress"] = done_msg
        logger.info(
            "Pull %s succeeded on provider %s in %.1fs",
            model,
            pname,
            duration / 1000,
        )
        log_kw: dict[str, Any] = {
            "provider_id": pid,
            "provider_name": pname,
            "protocol": "ollama",
            "endpoint": log_endpoint,
            "model": model,
            "request_size": 0,
            "response_size": 0,
            "duration_ms": duration,
            "status": "ok",
        }
        if source_ip is not None:
            log_kw["source_ip"] = source_ip
        meta = request_meta or ""
        if speed_mbps > 0:
            meta = f"{meta} {speed_mbps:.1f} MB/s".strip()
        if meta:
            log_kw["request_meta"] = meta
        await db.save_request_log(RequestLog(**log_kw))
    except Exception as exc:
        duration = (time.monotonic() - start) * 1000
        logger.error(
            "Pull %s FAILED on provider %s after %.1fs: %s",
            model,
            pname,
            duration / 1000,
            exc,
        )
        async with progress_lock:
            pull_entry["failed"].append(pid)
            pull_entry["progress"] = f"{prefix}: FAILED"
        log_kw = {
            "provider_id": pid,
            "provider_name": pname,
            "protocol": "ollama",
            "endpoint": log_endpoint,
            "model": model,
            "request_size": 0,
            "response_size": 0,
            "duration_ms": duration,
            "status": "error",
            "error_detail": str(exc)[:500],
        }
        if source_ip is not None:
            log_kw["source_ip"] = source_ip
        if request_meta is not None:
            log_kw["request_meta"] = request_meta
        await db.save_request_log(RequestLog(**log_kw))


async def _api_key_exists(db, key_id: int) -> bool:
    keys = await db.list_api_keys()
    return any(int(k["id"]) == int(key_id) for k in keys)


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


def _ollama_library_slug(model_name: str) -> str:
    """Extract an Ollama library slug from raw model names/URLs.

    Handles cache/registry-prefixed names like
    ``host:9200/library/llama3.2:latest`` by removing host and ``library/``
    prefix, then stripping tag/digest for a stable library page URL.
    """
    name = model_name.strip()
    if not name:
        return ""

    if "://" in name:
        name = name.split("://", 1)[1]

    parts = name.split("/")
    if len(parts) > 1:
        first = parts[0]
        if "." in first or ":" in first or first == "localhost":
            name = "/".join(parts[1:])

    if name.startswith("library/"):
        name = name[len("library/") :]

    if "@" in name:
        name = name.split("@", 1)[0]

    slash_idx = name.rfind("/")
    colon_idx = name.rfind(":")
    if colon_idx > slash_idx:
        name = name[:colon_idx]

    return name


def _tunnel_ip_from_cidr(address_cidr: str) -> str | None:
    s = (address_cidr or "").strip().split("/")[0].strip()
    return s or None


def _mask_peering_key(api_key: str) -> str:
    k = (api_key or "").strip()
    if not k:
        return ""
    return k[:8] + "..." if len(k) > 8 else "..."


async def _peering_key_matches(request: Request, db: Any) -> bool:
    hdr = (request.headers.get("x-peering-key") or "").strip()
    cfg = await db.get_wireguard_peering_config()
    key = (cfg.get("peering_api_key") or "").strip()
    if not key or not hdr:
        return False
    if not secrets.compare_digest(hdr, key):
        return False
    expires = cfg.get("peering_key_expires_at")
    if expires is not None:
        if datetime.utcnow() > expires:
            return False
    max_uses = cfg.get("peering_key_max_uses")
    if max_uses is not None:
        use_count = await db.increment_peering_key_use_count()
        if use_count > max_uses:
            return False
    return True


async def _wireguard_peer_info_payload(db: Any) -> dict:
    iface = await db.get_wireguard_interface()
    ip = _tunnel_ip_from_cidr(iface.get("address_cidr") or "")
    out: dict[str, Any] = {
        "public_key": iface.get("public_key") or "",
        "name": socket.gethostname(),
        "ollama_url": f"http://{ip}:{settings.api_port}" if ip else "",
        "llamacpp_url": f"http://{ip}:{settings.llamacpp_port}" if ip else "",
        "version": __version__,
    }
    ep = (iface.get("endpoint_public") or "").strip()
    if ep:
        out["endpoint"] = ep
    return out


def _ollama_library_url(model_name: str) -> str:
    """Build an Ollama model library URL from a model name."""
    slug = _ollama_library_slug(model_name)
    if not slug:
        return "https://ollama.com/library"
    return f"https://ollama.com/library/{quote(slug, safe='/')}"


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
templates.env.globals["ollama_library_url"] = _ollama_library_url
templates.env.filters["localtime"] = _localtime

router = APIRouter()


def template_auth_ctx(request: Request) -> dict[str, Any]:
    if getattr(request.state, "auth_bootstrap", False):
        return {
            "auth_bootstrap": True,
            "read_only": False,
            "show_api_keys": True,
            "show_wireguard": True,
            "show_log": True,
            "show_cache": True,
            "show_users_tab": True,
            "dash_user": None,
        }
    user = getattr(request.state, "dashboard_user", None)
    if user is None:
        return {
            "auth_bootstrap": False,
            "read_only": True,
            "show_api_keys": False,
            "show_wireguard": False,
            "show_log": False,
            "show_cache": False,
            "show_users_tab": False,
            "dash_user": None,
        }
    adm = user.is_admin
    return {
        "auth_bootstrap": False,
        "read_only": not adm,
        "show_api_keys": adm,
        "show_wireguard": adm,
        "show_log": adm,
        "show_cache": adm,
        "show_users_tab": adm,
        "dash_user": user,
    }


def merge_dash_template_ctx(request: Request, ctx: dict[str, Any]) -> dict[str, Any]:
    merged = template_auth_ctx(request)
    merged.update(ctx)
    return merged


def _minimal_wg_iface() -> dict[str, Any]:
    return {
        "enabled": False,
        "public_key": "",
        "address_cidr": "",
        "listen_port": 51820,
        "endpoint_public": None,
        "mtu": None,
        "private_key": "",
        "peering_enabled": False,
        "peering_api_key": "",
    }


def _safe_next_url(next_raw: str | None) -> str:
    n = (next_raw or "/").strip()
    if not n.startswith("/") or n.startswith("//"):
        return "/"
    return n


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


@router.get("/login", response_class=HTMLResponse)
async def login_page(request: Request):
    db = deps.get_db()
    if await db.count_dashboard_users() == 0:
        return RedirectResponse(url="/", status_code=302)
    err = request.query_params.get("error")
    next_url = _safe_next_url(request.query_params.get("next"))
    return templates.TemplateResponse(
        request,
        "login.html",
        {
            "error": err,
            "next_url": next_url,
            "dash_user": None,
            "show_users_tab": False,
        },
    )


@router.post("/login")
async def login_submit(
    request: Request,
    username: str = Form(...),
    password: str = Form(...),
    next: str = Form("/"),
):
    db = deps.get_db()
    if await db.count_dashboard_users() == 0:
        return RedirectResponse(url="/", status_code=302)
    try:
        un = normalize_username(username)
    except ValueError:
        return RedirectResponse(url="/login?error=invalid", status_code=303)
    row = await db.get_dashboard_user_by_username(un)
    if not row or not verify_password(password, row["password_hash"]):
        return RedirectResponse(url="/login?error=invalid", status_code=303)
    secret = await deps.get_dashboard_session_secret()
    token = sign_session(row["id"], secret)
    dest = _safe_next_url(next)
    resp = RedirectResponse(url=dest, status_code=303)
    resp.set_cookie(
        AUTH_COOKIE,
        token,
        httponly=True,
        samesite="lax",
        max_age=SESSION_MAX_AGE,
        secure=settings.dashboard_cookie_secure,
    )
    return resp


@router.post("/logout")
async def logout_post():
    resp = RedirectResponse(url="/login", status_code=303)
    resp.delete_cookie(AUTH_COOKIE)
    return resp


@router.get("/users", response_class=HTMLResponse)
async def users_page(request: Request):
    db = deps.get_db()
    n = await db.count_dashboard_users()
    bootstrap = getattr(request.state, "auth_bootstrap", False)
    user = getattr(request.state, "dashboard_user", None)
    if n > 0 and not bootstrap and (not user or not user.is_admin):
        raise HTTPException(status_code=403, detail="Forbidden")
    users = await db.list_dashboard_users() if n else []
    return templates.TemplateResponse(
        request,
        "users.html",
        merge_dash_template_ctx(
            request,
            {
                "dashboard_users": users,
                "users_exist": n > 0,
                "query_error": request.query_params.get("error"),
                "query_created": request.query_params.get("created"),
                "query_updated": request.query_params.get("updated"),
            },
        ),
    )


@router.post("/users/add")
async def users_add(
    request: Request,
    username: str = Form(...),
    password: str = Form(...),
    is_admin: str = Form("0"),
):
    db = deps.get_db()
    n = await db.count_dashboard_users()
    bootstrap = getattr(request.state, "auth_bootstrap", False)
    user = getattr(request.state, "dashboard_user", None)
    if n > 0 and not bootstrap and (not user or not user.is_admin):
        raise HTTPException(status_code=403, detail="Forbidden")
    try:
        un = normalize_username(username)
    except ValueError as e:
        q = quote(str(e))
        return RedirectResponse(url=f"/users?error={q}", status_code=303)
    if await db.get_dashboard_user_by_username(un):
        return RedirectResponse(url="/users?error=exists", status_code=303)
    if len(password) < 8:
        return RedirectResponse(url="/users?error=password", status_code=303)
    make_admin = is_admin in ("1", "on", "true", "yes")
    if n == 0:
        make_admin = True
    await db.create_dashboard_user(un, hash_password(password), make_admin)
    invalidate_dashboard_user_count_cache()
    return RedirectResponse(url="/users?created=1", status_code=303)


@router.post("/users/{user_id}/delete")
async def users_delete(request: Request, user_id: int):
    db = deps.get_db()
    n = await db.count_dashboard_users()
    bootstrap = getattr(request.state, "auth_bootstrap", False)
    user = getattr(request.state, "dashboard_user", None)
    if n > 0 and not bootstrap and (not user or not user.is_admin):
        raise HTTPException(status_code=403, detail="Forbidden")
    target = await db.get_dashboard_user_by_id(user_id)
    if not target:
        raise HTTPException(status_code=404, detail="User not found")
    if target["is_admin"] and await db.count_dashboard_admins() <= 1:
        return RedirectResponse(url="/users?error=last-admin", status_code=303)
    await db.delete_dashboard_user(user_id)
    invalidate_dashboard_user_count_cache()
    return RedirectResponse(url="/users", status_code=303)


@router.post("/users/{user_id}/password")
async def users_set_password(
    request: Request,
    user_id: int,
    new_password: str = Form(...),
):
    db = deps.get_db()
    n = await db.count_dashboard_users()
    bootstrap = getattr(request.state, "auth_bootstrap", False)
    user = getattr(request.state, "dashboard_user", None)
    if n > 0 and not bootstrap and (not user or not user.is_admin):
        raise HTTPException(status_code=403, detail="Forbidden")
    target = await db.get_dashboard_user_by_id(user_id)
    if not target:
        raise HTTPException(status_code=404, detail="User not found")
    if len(new_password) < 8:
        return RedirectResponse(url="/users?error=password", status_code=303)
    await db.update_dashboard_user_password(user_id, hash_password(new_password))
    return RedirectResponse(url="/users?updated=1", status_code=303)


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

    if read_only:
        api_keys: list[Any] = []
        allow_unauthenticated = False
        key_routing_preview = {"latency": [], "throughput": [], "chaos": []}
        log_page = 1
        log_pages = 1
        log_total = 0
        log_entries: list[Any] = []
        cache_stats = None
        cached_models: set[str] = set()
        wg_iface = _minimal_wg_iface()
        wg_peers: list[Any] = []
        wg_path_set = bool((settings.wireguard_config_path or "").strip())
        wg_peering_key_masked = ""
        peer_to_provider: dict[int, dict[str, Any]] = {}
    else:
        api_keys = await db.list_api_keys()
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

        wg_iface = await db.get_wireguard_interface()
        wg_peers = await db.list_wireguard_peers()
        wg_path_set = bool((settings.wireguard_config_path or "").strip())
        wg_peering_key_masked = _mask_peering_key(wg_iface.get("peering_api_key") or "")
        peer_to_provider = {}
        for info in infos:
            pid = info.provider.wireguard_peer_id
            if pid is not None and info.provider.id is not None:
                peer_to_provider[pid] = {
                    "id": info.provider.id,
                    "name": info.provider.name,
                }

    provider_model_names = {m["name"] for m in all_models}
    if not read_only:
        cache = deps.get_cache()
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
        request,
        "dashboard.html",
        merge_dash_template_ctx(
            request,
            {
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
                "cache_external_host_set": bool(
                    (settings.cache_external_host or "").strip()
                ),
                "wg_iface": wg_iface,
                "wg_peers": wg_peers,
                "wg_peer_to_provider": peer_to_provider,
                "wg_peering_key_masked": wg_peering_key_masked,
                "wireguard_config_path": settings.wireguard_config_path or "",
                "wireguard_path_set": wg_path_set,
                "wireguard_legacy_volume": settings.wireguard_legacy_volume,
            },
        ),
    )


@router.get("/api/status")
async def api_status(request: Request):
    _prune_completed(_active_pulls)
    _prune_completed(_active_benchmarks)
    _prune_completed(_active_fill_pulls)
    _prune_completed(_active_fill_benchmarks)

    pm = deps.get_pm()
    db = deps.get_db()
    dash_user = getattr(request.state, "dashboard_user", None)
    bootstrap = getattr(request.state, "auth_bootstrap", False)
    admin_payload = bootstrap or (dash_user is not None and dash_user.is_admin)
    infos = await pm.list_provider_infos()
    all_models = await db.list_all_models()
    log_total = await db.count_request_logs() if admin_payload else 0

    wg_status_payload: dict[str, Any] | None = None
    if admin_payload:
        from ..wireguard_manager import get_tunnel_status, is_wireguard_available

        wg_avail = await is_wireguard_available()
        wg_tun = (
            await get_tunnel_status() if wg_avail else {"running": False, "peers": []}
        )
        wg_linked: dict[str, dict[str, Any]] = {}
        for info in infos:
            if info.provider.wireguard_peer_id and info.provider.id is not None:
                wg_linked[str(info.provider.wireguard_peer_id)] = {
                    "id": info.provider.id,
                    "name": info.provider.name,
                }

        wg_status_payload = {
            "available": wg_avail,
            "running": bool(wg_tun.get("running")),
            "peer_count": len(wg_tun.get("peers") or []),
            "peers": wg_tun.get("peers") or [],
            "linked_providers": wg_linked,
        }

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
                "wireguard_ok": info.wireguard_ok,
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

    if not admin_payload:
        active_pulls = {}
        active_benchmarks = {}

    cache = deps.get_cache()
    cache_stats = None
    if admin_payload and cache:
        cache_stats = cache.stats()
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
        payload["wireguard"] = wg_status_payload
    return JSONResponse(payload)


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


@router.get("/api/keys/{key_id}/pins")
async def api_list_key_pins(key_id: int):
    db = deps.get_db()
    if not await _api_key_exists(db, key_id):
        raise HTTPException(status_code=404, detail="API key not found")
    pins = await db.list_api_key_model_pins(key_id)
    return JSONResponse({"pins": pins})


@router.post("/api/keys/{key_id}/pins")
async def api_set_key_pin(key_id: int, request: Request):
    db = deps.get_db()
    if not await _api_key_exists(db, key_id):
        raise HTTPException(status_code=404, detail="API key not found")
    body = await request.json()
    model_name = str(body.get("model_name") or "").strip()
    provider_id_raw = body.get("provider_id")
    if not model_name or provider_id_raw is None:
        raise HTTPException(
            status_code=400, detail="model_name and provider_id are required"
        )
    provider_id = int(provider_id_raw)
    provider = await db.get_provider(provider_id)
    if provider is None:
        raise HTTPException(status_code=404, detail="Provider not found")
    await db.set_api_key_model_pin(key_id, model_name, provider_id)
    return JSONResponse({"ok": True})


@router.delete("/api/keys/{key_id}/pins/{model_name:path}")
async def api_delete_key_pin(key_id: int, model_name: str):
    db = deps.get_db()
    if not await _api_key_exists(db, key_id):
        raise HTTPException(status_code=404, detail="API key not found")
    await db.remove_api_key_model_pin(key_id, model_name)
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
        request,
        "provider_detail.html",
        merge_dash_template_ctx(
            request,
            {
                "info": info,
                "missing_models": missing_models,
                "cached_models": cached_models,
                "cached_only_models": cached_only_models,
                "cache_external_host_set": bool(
                    (settings.cache_external_host or "").strip()
                ),
                "cache_enabled": settings.cache_enabled,
            },
        ),
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
    except httpx.HTTPError as exc:
        logger.error(
            "Delete model failed (provider_id=%s, model=%r): %s",
            provider_id,
            model_name,
            describe_httpx_error(exc),
        )
        raise HTTPException(status_code=500, detail=describe_httpx_error(exc))
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

    progress_lock = asyncio.Lock()

    async def _run_pull():
        pull_entry = _active_pulls[pull_id]
        n = len(provider_ids)
        try:
            await asyncio.gather(
                *[
                    _pull_one_ollama(
                        pid=pid,
                        idx=i,
                        total=n,
                        model=model,
                        pull_api=pull_api,
                        cache_url=cache_url,
                        pm=pm,
                        db=db,
                        pull_entry=pull_entry,
                        progress_lock=progress_lock,
                        source_ip=source_ip,
                        log_endpoint="/api/pull",
                        request_meta=f"pull_api={pull_api}",
                    )
                    for i, pid in enumerate(provider_ids)
                ]
            )
        finally:
            async with progress_lock:
                pull_entry["status"] = "done"
                pull_entry["completed_at"] = time.monotonic()

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
            job["completed_at"] = time.monotonic()
            if total == 0:
                job["progress"] = "No missing ollama models to pull."
            else:
                job["progress"] = (
                    f"Done: {job['completed']} pulled, {job['failed']} failed."
                )
        except Exception as exc:
            job["status"] = "failed"
            job["completed_at"] = time.monotonic()
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
    progress_lock = asyncio.Lock()

    async def _run():
        pull_entry = _active_pulls[pull_id]
        try:
            await _pull_one_ollama(
                pid=provider_id,
                idx=0,
                total=1,
                model=model,
                pull_api="auto",
                cache_url=cache_url,
                pm=pm,
                db=db,
                pull_entry=pull_entry,
                progress_lock=progress_lock,
                source_ip=None,
                log_endpoint="/api/pull",
                request_meta=None,
            )
        finally:
            async with progress_lock:
                pull_entry["status"] = "done"
                pull_entry["completed_at"] = time.monotonic()

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
    progress_lock = asyncio.Lock()

    async def _run():
        pull_entry = _active_pulls[pull_id]
        n = len(provider_ids)
        try:
            await asyncio.gather(
                *[
                    _pull_one_ollama(
                        pid=pid,
                        idx=i,
                        total=n,
                        model=model,
                        pull_api="auto",
                        cache_url=cache_url,
                        pm=pm,
                        db=db,
                        pull_entry=pull_entry,
                        progress_lock=progress_lock,
                        source_ip=None,
                        log_endpoint="/api/pull",
                        request_meta=None,
                    )
                    for i, pid in enumerate(provider_ids)
                ]
            )
        finally:
            async with progress_lock:
                pull_entry["status"] = "done"
                pull_entry["completed_at"] = time.monotonic()

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
        pull_entry["completed_at"] = time.monotonic()

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


def _wg_tab_redirect() -> RedirectResponse:
    return RedirectResponse(url="/?tab=wireguard", status_code=303)


class PeeringConfigBody(BaseModel):
    enabled: bool = False
    api_key: str = ""
    regenerate_api_key: bool = False
    expires_hours: int | None = None
    max_uses: int | None = None


class PeerImportBody(BaseModel):
    peer_config: dict[str, Any]
    our_tunnel_ip: str
    add_as_provider: bool = True


class PeerRequestBody(BaseModel):
    public_key: str
    tunnel_ip: str = ""
    allowed_ips: str
    endpoint: str = ""
    name: str = ""
    ollama_url: str = ""
    llamacpp_url: str = ""
    add_as_provider: bool = True


class WireGuardConnectBody(BaseModel):
    remote_url: str
    remote_api_key: str
    our_tunnel_ip: str
    their_tunnel_ip: str
    add_as_provider: bool = True


class WireGuardPeerDebugBody(BaseModel):
    peer_id: int


@router.get("/api/wireguard/status")
async def api_wireguard_status():
    """Summarize WireGuard settings (no private keys)."""
    db = deps.get_db()
    iface = await db.get_wireguard_interface()
    peers = await db.list_wireguard_peers()
    return JSONResponse(
        {
            "config_path_set": bool((settings.wireguard_config_path or "").strip()),
            "config_path": settings.wireguard_config_path or "",
            "enabled": iface["enabled"],
            "public_key": iface["public_key"],
            "address_cidr": iface["address_cidr"],
            "listen_port": iface["listen_port"],
            "endpoint_public": iface.get("endpoint_public"),
            "mtu": iface.get("mtu"),
            "peer_count": len(peers),
            "active_peers": sum(1 for p in peers if p["enabled"]),
        }
    )


@router.post("/api/wireguard/apply")
async def api_wireguard_apply():
    """Render WireGuard config from the database, write the file, and sync the tunnel."""
    db = deps.get_db()
    ok, msg = await sync_wireguard_config_to_disk(db)
    return JSONResponse({"ok": ok, "message": msg})


@router.post("/api/wireguard/debug-peer")
async def api_wireguard_debug_peer(body: WireGuardPeerDebugBody):
    """Run ping / curl / host diagnostics toward a peer (admin session)."""
    db = deps.get_db()
    result = await debug_peer_connectivity(db, body.peer_id)
    if result.get("error") == "peer_not_found":
        raise HTTPException(status_code=404, detail="Peer not found")
    return JSONResponse(result)


@router.get("/api/wireguard/peering-config")
async def api_wireguard_peering_config_get():
    db = deps.get_db()
    cfg = await db.get_wireguard_peering_config()
    exp = cfg.get("peering_key_expires_at")
    return JSONResponse(
        {
            "enabled": cfg["peering_enabled"],
            "api_key_masked": _mask_peering_key(cfg["peering_api_key"]),
            "peering_key_expires_at": exp.isoformat() + "Z" if exp else None,
            "peering_key_use_count": cfg.get("peering_key_use_count", 0),
            "peering_key_max_uses": cfg.get("peering_key_max_uses"),
        }
    )


@router.post("/api/wireguard/peering-config")
async def api_wireguard_peering_config_post(body: PeeringConfigBody):
    db = deps.get_db()
    prev = await db.get_wireguard_peering_config()
    prev_key = (prev.get("peering_api_key") or "").strip()

    if body.regenerate_api_key:
        key = secrets.token_urlsafe(32)
    else:
        new_key = (body.api_key or "").strip()
        if new_key:
            key = new_key
        else:
            key = prev_key
            if not key:
                key = secrets.token_urlsafe(32)

    data = body.model_dump(exclude_unset=True)
    if "expires_hours" in data:
        eh = data["expires_hours"]
        if eh is not None and eh > 0:
            peering_key_expires_at = datetime.utcnow() + timedelta(hours=int(eh))
        else:
            peering_key_expires_at = None
    else:
        peering_key_expires_at = prev.get("peering_key_expires_at")

    if "max_uses" in data:
        mu = data["max_uses"]
        peering_key_max_uses = int(mu) if mu is not None and int(mu) > 0 else None
    else:
        peering_key_max_uses = prev.get("peering_key_max_uses")

    reset_uc = body.regenerate_api_key
    submitted_key = (body.api_key or "").strip()
    if not reset_uc and submitted_key and submitted_key != prev_key:
        reset_uc = True
    if "expires_hours" in data or "max_uses" in data:
        reset_uc = True

    await db.set_wireguard_peering_config(
        body.enabled,
        key,
        peering_key_expires_at=peering_key_expires_at,
        peering_key_max_uses=peering_key_max_uses,
        reset_peering_key_use_count=reset_uc,
    )
    return JSONResponse(
        {
            "enabled": body.enabled,
            "api_key": key,
            "api_key_masked": _mask_peering_key(key),
        }
    )


def _parse_tunnel_ip(label: str, raw: str) -> str:
    s = raw.strip()
    if not s:
        raise HTTPException(status_code=400, detail=f"{label} is required")
    try:
        ipaddress.ip_address(s)
    except ValueError as e:
        raise HTTPException(
            status_code=400,
            detail=f"{label} must be a valid IPv4 or IPv6 address",
        ) from e
    return s


@router.get("/api/wireguard/export-peer-config")
async def api_wireguard_export_peer_config():
    """Return a JSON blob the operator can share out-of-band (session auth)."""
    db = deps.get_db()
    iface = await db.get_wireguard_interface()
    payload = await _wireguard_peer_info_payload(db)
    tip = _tunnel_ip_from_cidr(iface.get("address_cidr") or "")
    payload["tunnel_ip"] = tip or ""
    payload["allowed_ips"] = f"{tip}/32" if tip else ""
    payload["exported_at"] = datetime.utcnow().isoformat() + "Z"
    return JSONResponse(payload)


@router.post("/api/wireguard/import-peer-config")
async def api_wireguard_import_peer_config(body: PeerImportBody):
    """Apply a peer blob from another router; no outbound HTTP (session auth)."""
    _parse_tunnel_ip("our_tunnel_ip", body.our_tunnel_ip)

    db = deps.get_db()
    pm = deps.get_pm()

    remote_pk = (body.peer_config.get("public_key") or "").strip()
    if not remote_pk or not is_valid_wg_key_b64(remote_pk):
        raise HTTPException(
            status_code=400, detail="Invalid or missing public_key in peer config"
        )

    name = (body.peer_config.get("name") or "remote").strip() or "remote"
    endpoint = (body.peer_config.get("endpoint") or "").strip() or None
    allowed_ips = (body.peer_config.get("allowed_ips") or "").strip()
    if not allowed_ips:
        tunnel_ip = (body.peer_config.get("tunnel_ip") or "").strip()
        if tunnel_ip:
            allowed_ips = f"{tunnel_ip}/32"
    if not allowed_ips:
        raise HTTPException(
            status_code=400, detail="Cannot determine AllowedIPs from peer config"
        )

    existing = await db.find_wireguard_peer_by_public_key(remote_pk)
    if existing:
        peer_id = int(existing["id"])
        await db.update_wireguard_peer(
            peer_id,
            name=name,
            public_key=remote_pk,
            allowed_ips=allowed_ips,
            preshared_key=existing.get("preshared_key"),
            endpoint=endpoint,
            persistent_keepalive=existing.get("persistent_keepalive") or 25,
            enabled=True,
        )
    else:
        peer_id = await db.add_wireguard_peer(
            name=name,
            public_key=remote_pk,
            allowed_ips=allowed_ips,
            endpoint=endpoint,
            persistent_keepalive=25,
            enabled=True,
        )

    ok, msg = await sync_wireguard_config_to_disk(db)
    if not ok:
        logger.warning("import-peer-config apply: %s", msg)

    added_provider = False
    if body.add_as_provider:
        ro = (body.peer_config.get("ollama_url") or "").strip().rstrip("/")
        rl = (body.peer_config.get("llamacpp_url") or "").strip().rstrip("/")
        linked = await db.get_providers_by_peer_id(peer_id)
        if not linked and ro:
            ptype = ProviderType.BOTH if rl else ProviderType.OLLAMA
            await pm.add_provider(
                name,
                ro,
                provider_type=ptype,
                llamacpp_url=rl or None,
                wireguard_peer_id=peer_id,
            )
            added_provider = True

    return JSONResponse(
        {
            "ok": True,
            "peer_id": peer_id,
            "added_provider": added_provider,
            "message": msg if not ok else "peer imported",
        }
    )


@router.get("/api/wireguard/peer-info")
async def api_wireguard_peer_info(request: Request):
    db = deps.get_db()
    if not await _peering_key_matches(request, db):
        raise HTTPException(status_code=401, detail="Invalid or missing X-Peering-Key")
    return JSONResponse(await _wireguard_peer_info_payload(db))


@router.post("/api/wireguard/peer-request")
async def api_wireguard_peer_request(request: Request, body: PeerRequestBody):
    db = deps.get_db()
    if not await _peering_key_matches(request, db):
        raise HTTPException(status_code=401, detail="Invalid or missing X-Peering-Key")
    cfg = await db.get_wireguard_peering_config()
    if not cfg["peering_enabled"]:
        raise HTTPException(
            status_code=403, detail="Peering is not enabled on this router"
        )
    pk = body.public_key.strip()
    if not is_valid_wg_key_b64(pk):
        raise HTTPException(status_code=400, detail="Invalid public_key")

    existing = await db.find_wireguard_peer_by_public_key(pk)
    if existing:
        peer_id = int(existing["id"])
        await db.update_wireguard_peer(
            peer_id,
            name=body.name or existing.get("name") or "",
            public_key=pk,
            allowed_ips=body.allowed_ips.strip(),
            preshared_key=existing.get("preshared_key"),
            endpoint=body.endpoint.strip() or None,
            persistent_keepalive=existing.get("persistent_keepalive"),
            enabled=True,
        )
    else:
        peer_id = await db.add_wireguard_peer(
            name=(body.name or "peer").strip() or "peer",
            public_key=pk,
            allowed_ips=body.allowed_ips.strip(),
            endpoint=body.endpoint.strip() or None,
            persistent_keepalive=25,
            enabled=True,
        )

    ok, msg = await sync_wireguard_config_to_disk(db)
    if not ok:
        logger.warning("peer-request apply config: %s", msg)

    if body.add_as_provider and body.ollama_url.strip():
        linked = await db.get_providers_by_peer_id(peer_id)
        if not linked:
            pm = deps.get_pm()
            ou = body.ollama_url.strip().rstrip("/")
            lu = body.llamacpp_url.strip().rstrip("/") if body.llamacpp_url else None
            if lu and ou:
                ptype = ProviderType.BOTH
            elif lu:
                ptype = ProviderType.LLAMACPP
            else:
                ptype = ProviderType.OLLAMA
            pname = (body.name or f"peer-{peer_id}").strip() or f"peer-{peer_id}"
            await pm.add_provider(
                pname,
                ou if ptype != ProviderType.LLAMACPP else (lu or ou),
                provider_type=ptype,
                llamacpp_url=lu if ptype != ProviderType.OLLAMA else None,
                wireguard_peer_id=peer_id,
            )

    out = await _wireguard_peer_info_payload(db)
    out["add_as_provider"] = bool(body.add_as_provider)
    return JSONResponse(out)


@router.post("/api/wireguard/connect")
async def api_wireguard_connect(body: WireGuardConnectBody):
    db = deps.get_db()
    pm = deps.get_pm()
    base = body.remote_url.strip().rstrip("/")
    if not base.startswith("https://"):
        raise HTTPException(
            status_code=400,
            detail=(
                "remote_url must use HTTPS to protect the peering API key in transit. "
                "Use the manual peer exchange workflow for HTTP-only remotes."
            ),
        )

    timeout = httpx.Timeout(30.0, connect=10.0)
    headers = {"X-Peering-Key": body.remote_api_key.strip()}
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            r = await client.get(f"{base}/api/wireguard/peer-info", headers=headers)
            if r.status_code != 200:
                raise HTTPException(
                    status_code=502,
                    detail=f"peer-info failed: HTTP {r.status_code} {r.text[:200]}",
                )
            remote = r.json()
    except HTTPException:
        raise
    except httpx.RequestError as exc:
        raise HTTPException(
            status_code=502, detail=f"peer-info unreachable: {exc}"
        ) from exc

    remote_pk = (remote.get("public_key") or "").strip()
    if not remote_pk or not is_valid_wg_key_b64(remote_pk):
        raise HTTPException(
            status_code=502, detail="Remote returned invalid public_key"
        )

    our_iface = await db.get_wireguard_interface()
    our_pub = (our_iface.get("public_key") or "").strip()
    if not our_pub:
        raise HTTPException(
            status_code=400,
            detail="Local WireGuard public key missing; set private key first",
        )
    our_ep = (our_iface.get("endpoint_public") or "").strip()
    if not our_ep:
        raise HTTPException(
            status_code=400,
            detail="Set Public endpoint on the WireGuard interface so the remote peer can reach you",
        )

    our_ip = _parse_tunnel_ip("our_tunnel_ip", body.our_tunnel_ip)
    their_ip = _parse_tunnel_ip("their_tunnel_ip", body.their_tunnel_ip)
    allowed_us = f"{our_ip}/32"
    ollama_us = f"http://{our_ip}:{settings.api_port}"
    lcpp_us = f"http://{our_ip}:{settings.llamacpp_port}"

    existing_remote = await db.find_wireguard_peer_by_public_key(remote_pk)
    registered = False
    remote_added_us = False
    if not existing_remote:
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                pr = await client.post(
                    f"{base}/api/wireguard/peer-request",
                    headers=headers,
                    json={
                        "public_key": our_pub,
                        "tunnel_ip": our_ip,
                        "allowed_ips": allowed_us,
                        "endpoint": our_ep,
                        "name": socket.gethostname(),
                        "ollama_url": ollama_us,
                        "llamacpp_url": lcpp_us,
                        "add_as_provider": body.add_as_provider,
                    },
                )
                if pr.status_code != 200:
                    raise HTTPException(
                        status_code=502,
                        detail=f"peer-request failed: HTTP {pr.status_code} {pr.text[:200]}",
                    )
                registered = True
                try:
                    remote_added_us = bool(pr.json().get("add_as_provider"))
                except Exception:
                    remote_added_us = body.add_as_provider
        except HTTPException:
            raise
        except httpx.RequestError as exc:
            raise HTTPException(
                status_code=502, detail=f"peer-request unreachable: {exc}"
            ) from exc

    remote_endpoint = (remote.get("endpoint") or "").strip()
    peer_row = await db.find_wireguard_peer_by_public_key(remote_pk)
    if peer_row:
        peer_id = int(peer_row["id"])
        await db.update_wireguard_peer(
            peer_id,
            name=(remote.get("name") or peer_row.get("name") or "remote").strip(),
            public_key=remote_pk,
            allowed_ips=f"{their_ip}/32",
            preshared_key=peer_row.get("preshared_key"),
            endpoint=remote_endpoint or None,
            persistent_keepalive=peer_row.get("persistent_keepalive") or 25,
            enabled=True,
        )
    else:
        peer_id = await db.add_wireguard_peer(
            name=(remote.get("name") or "remote").strip() or "remote",
            public_key=remote_pk,
            allowed_ips=f"{their_ip}/32",
            endpoint=remote_endpoint or None,
            persistent_keepalive=25,
            enabled=True,
        )

    ok, msg = await sync_wireguard_config_to_disk(db)
    if not ok:
        logger.warning("connect apply config: %s", msg)

    added_local_provider = False

    if body.add_as_provider:
        ro = (remote.get("ollama_url") or "").strip().rstrip("/")
        rl = (remote.get("llamacpp_url") or "").strip().rstrip("/")
        linked = await db.get_providers_by_peer_id(peer_id)
        if not linked and ro:
            if rl and ro:
                ptype = ProviderType.BOTH
            elif rl:
                ptype = ProviderType.LLAMACPP
            else:
                ptype = ProviderType.OLLAMA
            pname = (remote.get("name") or "remote-router").strip() or "remote-router"
            await pm.add_provider(
                pname,
                ro if ptype != ProviderType.LLAMACPP else (rl or ro),
                provider_type=ptype,
                llamacpp_url=rl if ptype != ProviderType.OLLAMA else None,
                wireguard_peer_id=peer_id,
            )
            added_local_provider = True

    return JSONResponse(
        {
            "ok": True,
            "peer_id": peer_id,
            "registered_on_remote": registered,
            "remote_added_us_as_provider": remote_added_us,
            "added_local_provider": added_local_provider,
            "message": msg if not ok else "connected",
        }
    )


@router.post("/api/wireguard/peers/{peer_id}/remove")
async def api_wireguard_peer_remove(peer_id: int, request: Request):
    db = deps.get_db()
    pm = deps.get_pm()
    try:
        body = await request.json()
    except Exception:
        body = {}
    remove_providers = bool(body.get("remove_providers"))
    peer = await db.get_wireguard_peer(peer_id)
    if not peer:
        raise HTTPException(status_code=404, detail="Peer not found")

    linked = await db.get_providers_by_peer_id(peer_id)
    if linked and not remove_providers:
        return JSONResponse(
            status_code=409,
            content={
                "error": "peer_has_linked_providers",
                "providers": [{"id": p.id, "name": p.name} for p in linked],
                "message": "Remove linked providers too?",
            },
        )

    if linked and remove_providers:
        for p in linked:
            assert p.id is not None
            await pm.remove_provider(p.id)
    elif linked:
        await db.unlink_providers_from_wireguard_peer(peer_id)

    await db.remove_wireguard_peer(peer_id)
    await sync_wireguard_config_to_disk(db)
    return JSONResponse({"ok": True})


@router.post("/wireguard/interface")
async def wireguard_save_interface(
    listen_port: int = Form(51820),
    address_cidr: str = Form(...),
    mtu: str = Form(""),
    endpoint_public: str = Form(""),
    private_key: str = Form(""),
    clear_private_key: Optional[str] = Form(None),
    enabled: Optional[str] = Form(None),
):
    db = deps.get_db()
    enabled_b = enabled in ("1", "on", "true", "yes")
    if listen_port < 1 or listen_port > 65535:
        raise HTTPException(status_code=400, detail="Listen port must be 1–65535")
    mtu_val: int | None = None
    if mtu.strip():
        try:
            mtu_val = int(mtu.strip())
            if mtu_val <= 0:
                raise ValueError("MTU must be positive")
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

    new_priv: str | None = None
    if clear_private_key in ("1", "on", "true", "yes"):
        new_priv = ""
    elif private_key.strip():
        pk = private_key.strip()
        if not is_valid_wg_key_b64(pk):
            raise HTTPException(status_code=400, detail="Invalid private key format")
        new_priv = pk

    try:
        await db.update_wireguard_interface(
            enabled=enabled_b,
            listen_port=listen_port,
            address_cidr=address_cidr.strip(),
            mtu=mtu_val,
            endpoint_public=endpoint_public.strip() or None,
            new_private_key=new_priv,
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    ok, msg = await sync_wireguard_config_to_disk(db)
    if not ok:
        raise HTTPException(
            status_code=400, detail=f"Saved settings but config file: {msg}"
        )
    return _wg_tab_redirect()


@router.post("/wireguard/generate-keys")
async def wireguard_generate_keys():
    db = deps.get_db()
    key = generate_wireguard_private_key()
    await db.set_wireguard_private_key(key)
    ok, msg = await sync_wireguard_config_to_disk(db)
    if not ok:
        raise HTTPException(status_code=400, detail=f"Key saved but config file: {msg}")
    return _wg_tab_redirect()


@router.post("/wireguard/peers/save")
async def wireguard_peer_save(
    peer_id: str = Form(""),
    name: str = Form(""),
    public_key: str = Form(...),
    allowed_ips: str = Form(...),
    preshared_key: str = Form(""),
    endpoint: str = Form(""),
    persistent_keepalive: str = Form(""),
    peer_enabled: str = Form("1"),
    link_provider: Optional[str] = Form(None),
    provider_name: str = Form(""),
    provider_type: str = Form("ollama"),
    ollama_url: str = Form(""),
    llamacpp_url: str = Form(""),
):
    db = deps.get_db()
    pm = deps.get_pm()
    pk = public_key.strip()
    if not is_valid_wg_key_b64(pk):
        raise HTTPException(status_code=400, detail="Invalid peer public key format")
    psk = preshared_key.strip()
    if psk and not is_valid_wg_key_b64(psk):
        raise HTTPException(status_code=400, detail="Invalid preshared key format")

    ka: int | None = None
    if persistent_keepalive.strip():
        try:
            ka = int(persistent_keepalive.strip())
            if ka < 0:
                raise ValueError()
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail="Persistent keepalive must be a non-negative integer",
            )
        if ka == 0:
            ka = None

    en = peer_enabled == "1"
    do_link = link_provider in ("1", "on", "true", "yes")
    pid: int

    try:
        if peer_id.strip():
            pid = int(peer_id.strip())
            existing = await db.get_wireguard_peer(pid)
            if not existing:
                raise HTTPException(status_code=404, detail="Peer not found")
            await db.update_wireguard_peer(
                pid,
                name=name,
                public_key=pk,
                allowed_ips=allowed_ips.strip(),
                preshared_key=psk or None,
                endpoint=endpoint.strip() or None,
                persistent_keepalive=ka,
                enabled=en,
            )
        else:
            pid = await db.add_wireguard_peer(
                name=name,
                public_key=pk,
                allowed_ips=allowed_ips.strip(),
                preshared_key=psk or None,
                endpoint=endpoint.strip() or None,
                persistent_keepalive=ka,
                enabled=en,
            )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    if do_link:
        ou = ollama_url.strip().rstrip("/")
        lu = llamacpp_url.strip().rstrip("/") if llamacpp_url.strip() else None
        linked = await db.get_providers_by_peer_id(pid)
        if not linked and ou:
            pt_raw = (provider_type or "ollama").strip().lower()
            if pt_raw == "both" or (lu and ou):
                ptype = ProviderType.BOTH
            elif pt_raw == "llamacpp" or (lu and not ou):
                ptype = ProviderType.LLAMACPP
            else:
                ptype = ProviderType.OLLAMA
            pname = provider_name.strip() or name.strip() or f"peer-{pid}"
            await pm.add_provider(
                pname,
                ou if ptype != ProviderType.LLAMACPP else (lu or ou),
                provider_type=ptype,
                llamacpp_url=lu if ptype != ProviderType.OLLAMA else None,
                wireguard_peer_id=pid,
            )
        elif not linked and lu and not ou:
            pname = provider_name.strip() or name.strip() or f"peer-{pid}"
            await pm.add_provider(
                pname,
                lu,
                provider_type=ProviderType.LLAMACPP,
                llamacpp_url=None,
                wireguard_peer_id=pid,
            )

    ok, msg = await sync_wireguard_config_to_disk(db)
    if not ok:
        raise HTTPException(
            status_code=400, detail=f"Peer saved but config file: {msg}"
        )
    return _wg_tab_redirect()
