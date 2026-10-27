from __future__ import annotations

import asyncio
import logging
import re
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional
from urllib.parse import quote, urlparse

import httpx
from fastapi import APIRouter, Form, HTTPException, Request
from pydantic import BaseModel
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.encoders import jsonable_encoder
from fastapi.templating import Jinja2Templates

from ..affinity import session_key
from ..auth import generate_api_key, key_hash, key_prefix
from ..router import (
    ENDPOINT_PROTOCOLS,
    ROUTING_MODES,
    RoutingPreferences,
)
from ..routing_table import DEFAULT_MODE, MODES
from .. import hf_download
from ..probe import FAMILY_LABELS, probe_url
from ..config import settings
from ..httpx_errors import describe_httpx_error
from ..cloud_presets import PRESETS, get_preset
from ..models import ProviderProtocol, RequestLog
from ..provider_manager import preset_settings
from ..provider_secrets import format_extra_headers, parse_extra_headers
from ..upstream_url import LOCALITIES
from . import deps

from .. import __version__
from ..worker_link.hub import (
    generate_worker_token,
    worker_token_hash,
    worker_token_prefix,
)
from ..worker_link.protocol import CLOSE_REVOKED
from ..worker_link.worker import normalize_hub_url
from .auth_core import (
    AUTH_COOKIE,
    SESSION_MAX_AGE,
    hash_password,
    normalize_username,
    sign_session,
    verify_password,
)
from . import oidc
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
            "error_detail": "pull failed",
        }
        if source_ip is not None:
            log_kw["source_ip"] = source_ip
        if request_meta is not None:
            log_kw["request_meta"] = request_meta
        await db.save_request_log(RequestLog(**log_kw))


async def _api_key_exists(db, key_id: int) -> bool:
    keys = await db.list_api_keys()
    return any(int(k["id"]) == int(key_id) for k in keys)


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
            "show_workers": True,
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
            "show_workers": False,
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
        "show_workers": adm,
        "show_log": adm,
        "show_cache": adm,
        "show_users_tab": adm,
        "dash_user": user,
    }


def merge_dash_template_ctx(request: Request, ctx: dict[str, Any]) -> dict[str, Any]:
    merged = template_auth_ctx(request)
    merged.update(ctx)
    return merged


def _safe_next_url(next_raw: str | None) -> str:
    """Reject absolute URLs and scheme-relative paths; allow same-site relative paths only."""
    n = (next_raw or "/").strip()
    if not n or "\n" in n or "\r" in n:
        return "/"
    parsed = urlparse(n)
    if parsed.scheme or parsed.netloc:
        return "/"
    path = parsed.path or "/"
    if not path.startswith("/") or path.startswith("//") or "\\" in path:
        return "/"
    # Rebuild from a constant leading "/" so the caller controls only the suffix.
    out = "/" + path[1:]
    if parsed.query:
        out += "?" + parsed.query
    if parsed.fragment:
        out += "#" + parsed.fragment
    return out


def _provider_url(provider_id: int, query: str = "") -> str:
    """Same-site URL for a provider detail page; the id only ever forms the suffix."""
    out = "/providers/" + str(int(provider_id))
    if query:
        out += "?" + query
    return out


def _same_origin_referer_path(request: Request, fallback: str) -> str:
    raw = (request.headers.get("referer") or "").strip()
    if not raw:
        return fallback
    try:
        ref = urlparse(raw)
        base = urlparse(str(request.base_url))
    except ValueError:
        return fallback
    if ref.scheme.lower() not in ("http", "https"):
        return fallback

    def _host_port(u) -> tuple[str, int]:
        host = (u.hostname or "").lower()
        port = u.port
        if port is None:
            port = 443 if u.scheme.lower() == "https" else 80
        return host, port

    rh, rp = _host_port(ref)
    bh, bp = _host_port(base)
    if not rh or rh != bh or rp != bp:
        return fallback
    path = ref.path or "/"
    if not path.startswith("/") or path.startswith("//"):
        return fallback
    if ref.query:
        return f"{path}?{ref.query}"
    return path


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


def _set_session_cookie(resp: RedirectResponse, user_id: int, secret: str) -> None:
    resp.set_cookie(
        AUTH_COOKIE,
        sign_session(user_id, secret),
        httponly=True,
        samesite="lax",
        max_age=SESSION_MAX_AGE,
        secure=settings.dashboard_cookie_secure,
    )


def _external_origin(request: Request) -> tuple[str, str]:
    """Scheme and host the browser used, honoring a reverse proxy's headers."""
    proto = request.headers.get("x-forwarded-proto", "").split(",")[0].strip()
    host = request.headers.get("x-forwarded-host", "").split(",")[0].strip()
    scheme = proto if proto in ("http", "https") else request.url.scheme
    return scheme, host or request.headers.get("host") or request.url.netloc


@router.get("/login", response_class=HTMLResponse)
async def login_page(request: Request):
    db = deps.get_db()
    cfg = oidc.current_config()
    users_exist = await db.count_dashboard_users() > 0
    if not users_exist and not cfg.usable:
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
            "oidc_enabled": cfg.usable,
            "oidc_display_name": cfg.display_name,
            # With no accounts yet only SSO can sign anyone in.
            "password_login": cfg.password_login_enabled and users_exist,
        },
    )


@router.post("/login")
async def login_submit(
    request: Request,
    username: str = Form(...),
    password: str = Form(...),
    next: str = Form("/"),
):
    if not oidc.current_config().password_login_enabled:
        raise HTTPException(status_code=404, detail="Not Found")
    db = deps.get_db()
    if await db.count_dashboard_users() == 0:
        return RedirectResponse(url="/", status_code=302)
    try:
        un = normalize_username(username)
    except ValueError:
        return RedirectResponse(url="/login?error=invalid", status_code=303)
    row = await db.get_dashboard_user_by_username(un)
    if (
        not row
        or row.get("auth_source", "local") != "local"
        or not verify_password(password, row["password_hash"])
    ):
        return RedirectResponse(url="/login?error=invalid", status_code=303)
    await db.touch_dashboard_user_login(row["id"])
    secret = await deps.get_dashboard_session_secret()
    resp = RedirectResponse(url=_safe_next_url(next), status_code=303)
    _set_session_cookie(resp, row["id"], secret)
    return resp


@router.get("/login/oidc")
async def login_oidc_start(request: Request):
    cfg = oidc.current_config()
    if not cfg.usable:
        raise HTTPException(status_code=404, detail="Not Found")
    try:
        meta = await oidc.provider_metadata(cfg)
    except oidc.OIDCError as exc:
        logger.warning("OIDC sign-in unavailable: %s", exc)
        return RedirectResponse(url="/login?error=oidc_provider", status_code=303)
    scheme, host = _external_origin(request)
    url, state = oidc.begin_login(
        cfg,
        meta,
        callback_url=oidc.redirect_uri(cfg, scheme, host),
        next_url=_safe_next_url(request.query_params.get("next")),
        secret=await deps.get_dashboard_session_secret(),
    )
    resp = RedirectResponse(url=url, status_code=302)
    resp.set_cookie(
        oidc.STATE_COOKIE,
        state,
        httponly=True,
        samesite="lax",
        max_age=oidc.STATE_MAX_AGE,
        path="/login/oidc",
        secure=settings.dashboard_cookie_secure,
    )
    return resp


@router.get(oidc.CALLBACK_PATH)
async def login_oidc_callback(request: Request):
    cfg = oidc.current_config()
    if not cfg.usable:
        raise HTTPException(status_code=404, detail="Not Found")
    db = deps.get_db()
    secret = await deps.get_dashboard_session_secret()
    try:
        claims, next_url = await oidc.complete_login(
            cfg,
            request.query_params,
            request.cookies.get(oidc.STATE_COOKIE),
            secret,
        )
        user = await oidc.sign_in(db, cfg, claims)
    except oidc.OIDCError as exc:
        level = logging.WARNING if exc.code in ("provider", "token") else logging.INFO
        logger.log(level, "OIDC sign-in failed: %s", exc)
        resp = RedirectResponse(url=f"/login?error=oidc_{exc.code}", status_code=303)
        resp.delete_cookie(oidc.STATE_COOKIE, path="/login/oidc")
        return resp
    invalidate_dashboard_user_count_cache()
    resp = RedirectResponse(url=_safe_next_url(next_url), status_code=303)
    resp.delete_cookie(oidc.STATE_COOKIE, path="/login/oidc")
    _set_session_cookie(resp, user["id"], secret)
    return resp


@router.post("/logout")
async def logout_post(request: Request):
    dest = "/login"
    user = getattr(request.state, "dashboard_user", None)
    cfg = oidc.current_config()
    if user is not None and user.is_sso and cfg.usable and cfg.logout_at_idp:
        try:
            meta = await oidc.provider_metadata(cfg)
            scheme, host = _external_origin(request)
            base = cfg.redirect_base_url or f"{scheme}://{host}"
            dest = oidc.logout_url(cfg, meta, base + "/login") or dest
        except oidc.OIDCError as exc:
            logger.warning("OIDC logout at the provider skipped: %s", exc)
    resp = RedirectResponse(url=dest, status_code=303)
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
    cfg = oidc.current_config()
    scheme, host = _external_origin(request)
    return templates.TemplateResponse(
        request,
        "users.html",
        merge_dash_template_ctx(
            request,
            {
                "dashboard_users": users,
                "users_exist": n > 0,
                "oidc": cfg if cfg.enabled else None,
                "oidc_problems": cfg.problems(),
                "oidc_callback": oidc.redirect_uri(cfg, scheme, host),
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
    except ValueError:
        return RedirectResponse(url="/users?error=invalid_username", status_code=303)
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
    if target.get("auth_source", "local") != "local":
        return RedirectResponse(url="/users?error=sso-password", status_code=303)
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

    if read_only:
        api_keys: list[Any] = []
        allow_unauthenticated = False
        key_routing_preview = {
            "balanced": [],
            "latency": [],
            "throughput": [],
            "chaos": [],
        }
        routing_summary = None
        log_page = 1
        log_pages = 1
        log_total = 0
        log_entries: list[Any] = []
        cache_stats = None
        cached_models: set[str] = set()
        workers: list[dict[str, Any]] = []
        worker_link_status: dict[str, Any] | None = None
    else:
        api_keys = await db.list_api_keys()
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

        workers = await _worker_rows(db, pm)
        worker_link_status = await deps.get_worker_link().status()

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
                "providers_seed": [_provider_status_payload(i) for i in infos],
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
                "api_keys": api_keys,
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
            },
        ),
    )


def _provider_status_payload(info: Any) -> dict[str, Any]:
    """One provider as the dashboard sees it; shared by the page and /api/status."""
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
        "hot_models": info.hot_models,
        "worker": info.worker,
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

    providers_data = [_provider_status_payload(info) for info in infos]

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
        "built_at": datetime.fromtimestamp(table.built_at, tz=timezone.utc),
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


@router.get("/api/routing-table")
async def api_routing_table():
    """The current routing table: candidates, scores and order per mode."""
    pm = deps.get_pm()
    rt = pm.router
    if rt is None:
        raise HTTPException(status_code=503, detail="router not running")
    table = rt.table or await rt.rebuild()
    out = table.to_dict()
    out["session_affinity"] = {
        "enabled_by_default": rt.session_affinity_default,
        **rt.affinity.stats(),
    }
    return JSONResponse(out)


_DEFAULT_EXPLAIN_ENDPOINT = "/v1/chat/completions"


def _tristate(value) -> bool | None:
    """Parse an optional on/off flag: None, "", "default" mean unset."""
    if value is None or isinstance(value, bool):
        return value
    v = str(value).strip().lower()
    if v in ("", "default", "none", "null"):
        return None
    if v in ("1", "true", "yes", "on"):
        return True
    if v in ("0", "false", "no", "off"):
        return False
    raise HTTPException(
        status_code=400, detail="session_affinity must be on, off or default"
    )


def _is_admin(request: Request) -> bool:
    if getattr(request.state, "auth_bootstrap", False):
        return True
    user = getattr(request.state, "dashboard_user", None)
    return bool(user and user.is_admin)


@router.get("/api/routing/explain")
async def api_explain_routing(request: Request):
    """How a request would be routed right now, and why.

    Query: ``model`` (required), ``endpoint`` (default /v1/chat/completions),
    and either ``mode`` (balanced, latency, throughput, chaos) with optional
    ``allow_fallback`` and ``session_affinity``, or ``key_id`` to use an API
    key's mode, fallback, session affinity setting and pins (admins only).
    ``session_id`` shows how a client sending that ``X-Session-ID`` would be
    routed, including its current binding.
    """
    q = request.query_params
    model = (q.get("model") or "").strip()
    if not model:
        raise HTTPException(status_code=400, detail="model is required")
    endpoint = (q.get("endpoint") or _DEFAULT_EXPLAIN_ENDPOINT).strip()
    if not endpoint.startswith("/"):
        endpoint = "/" + endpoint
    normalized = (
        endpoint if endpoint.startswith(("/v1/", "/api/")) else "/v1" + endpoint
    )
    if normalized not in ENDPOINT_PROTOCOLS:
        raise HTTPException(
            status_code=400,
            detail="endpoint must be one of: " + ", ".join(ENDPOINT_PROTOCOLS),
        )

    db = deps.get_db()
    key_info = None
    key_id = None
    raw_key_id = (q.get("key_id") or "").strip()
    if raw_key_id:
        if not _is_admin(request):
            raise HTTPException(status_code=403, detail="Forbidden")
        try:
            key_id = int(raw_key_id)
        except ValueError:
            raise HTTPException(status_code=400, detail="key_id must be a number")
        key = next((k for k in await db.list_api_keys() if k["id"] == key_id), None)
        if key is None:
            raise HTTPException(status_code=404, detail="API key not found")
        pins = await db.list_api_key_model_pins(key_id)
        mode = str(key["routing_mode"]).lower()
        prefs = RoutingPreferences(
            mode=mode if mode in ROUTING_MODES else DEFAULT_MODE,
            allow_fallback=bool(key["allow_fallback"]),
            pinned_providers={
                str(p["model_name"]): int(p["provider_id"]) for p in pins
            },
            session_affinity=key.get("session_affinity"),
            key_id=key_id,
            locality=key.get("locality"),
        )
        key_info = {"id": key_id, "key_prefix": key["key_prefix"]}
    else:
        mode = (q.get("mode") or DEFAULT_MODE).strip().lower()
        if mode not in ROUTING_MODES:
            raise HTTPException(
                status_code=400,
                detail="mode must be one of: " + ", ".join(sorted(ROUTING_MODES)),
            )
        allow = (q.get("allow_fallback") or "true").strip().lower()
        prefs = RoutingPreferences(
            mode=mode,
            allow_fallback=allow not in ("0", "false", "no", "off"),
            session_affinity=_tristate(q.get("session_affinity")),
            locality=_key_locality(q.get("locality")),
        )
    session_id = (q.get("session_id") or "").strip()
    session = (
        session_key({"x-session-id": session_id}, {"model": model}, key_id)
        if session_id
        else None
    )

    rt = deps.get_pm().router
    if rt is None:
        raise HTTPException(status_code=503, detail="router not running")
    result = await rt.explain(model, normalized, prefs, session=session)
    result["api_key"] = key_info
    return JSONResponse(result)


@router.post("/api/routing-table/rebuild")
async def api_rebuild_routing_table():
    pm = deps.get_pm()
    rt = pm.router
    if rt is None:
        raise HTTPException(status_code=503, detail="router not running")
    rt.mark_dirty("manual")
    table = await rt.rebuild()
    return JSONResponse({"version": table.version, "build_ms": table.build_ms})


@router.post("/api/keys/generate")
async def api_generate_key(request: Request):
    db = deps.get_db()
    body = await request.json()
    routing_mode = (body.get("routing_mode") or DEFAULT_MODE).strip().lower()
    allow_fallback = bool(body.get("allow_fallback", True))
    session_affinity = _tristate(body.get("session_affinity"))
    locality = _key_locality(body.get("locality"))
    cap = _usd(body.get("daily_spend_cap_usd"), "daily spend cap")
    if routing_mode not in ROUTING_MODES:
        raise HTTPException(status_code=400, detail="Invalid routing_mode")
    # PBKDF2 is slow; keep it off the event loop.
    plaintext = generate_api_key()
    hashed = await asyncio.to_thread(key_hash, plaintext)

    await db.create_api_key(
        key_prefix=key_prefix(plaintext),
        key_hash=hashed,
        routing_mode=routing_mode,
        allow_fallback=allow_fallback,
        session_affinity=session_affinity,
        locality=locality,
        daily_spend_cap_usd=cap,
    )
    return JSONResponse(
        {
            "api_key": plaintext,
            "daily_spend_cap_usd": cap,
            "routing_mode": routing_mode,
            "allow_fallback": allow_fallback,
            "session_affinity": session_affinity,
            "locality": locality,
        }
    )


@router.post("/api/keys/{key_id}/spend-cap")
async def api_set_key_spend_cap(key_id: int, request: Request):
    """Body: {daily_spend_cap_usd: number or null}. Past the cap, the key is
    limited to local providers until the next UTC day."""
    try:
        body = await request.json()
    except ValueError:
        raise HTTPException(status_code=400, detail="JSON body required")
    cap = _usd(body.get("daily_spend_cap_usd"), "daily spend cap")
    if not await deps.get_db().set_api_key_spend_cap(key_id, cap):
        raise HTTPException(status_code=404, detail="API key not found")
    return JSONResponse({"id": key_id, "daily_spend_cap_usd": cap})


@router.get("/api/providers/{provider_id}/prices")
async def api_list_prices(provider_id: int):
    """Prices set by hand, and how many models the provider's list prices."""
    db = deps.get_db()
    if await db.get_provider(provider_id) is None:
        raise HTTPException(status_code=404, detail="Provider not found")
    from ..pricing import listed_price

    listed = [
        {
            "model": m.name,
            "input_per_mtok": p.input_per_mtok,
            "output_per_mtok": p.output_per_mtok,
        }
        for m in await db.get_provider_models(provider_id)
        if (p := listed_price(m)) is not None
    ]
    return JSONResponse(
        {"prices": await db.list_model_prices(provider_id), "listed": listed}
    )


@router.post("/api/providers/{provider_id}/prices")
async def api_set_price(provider_id: int, request: Request):
    """Body: {model_pattern, input_per_mtok, output_per_mtok} in USD per
    million tokens. "*" prices every model of the provider."""
    db = deps.get_db()
    if await db.get_provider(provider_id) is None:
        raise HTTPException(status_code=404, detail="Provider not found")
    body = await request.json()
    pattern = str(body.get("model_pattern") or "").strip()
    if not pattern or len(pattern) > 255:
        raise HTTPException(status_code=400, detail="model_pattern is required")
    inp = _usd(body.get("input_per_mtok"), "input price")
    out = _usd(body.get("output_per_mtok"), "output price")
    if inp is None or out is None:
        raise HTTPException(status_code=400, detail="both prices are required")
    await db.set_model_price(provider_id, pattern, inp, out)
    return JSONResponse(
        {"model_pattern": pattern, "input_per_mtok": inp, "output_per_mtok": out}
    )


@router.delete("/api/providers/{provider_id}/prices/{price_id}")
async def api_delete_price(provider_id: int, price_id: int):
    if not await deps.get_db().remove_model_price(provider_id, price_id):
        raise HTTPException(status_code=404, detail="price not found")
    return JSONResponse({"status": "ok"})


@router.post("/api/keys/{key_id}/locality")
async def api_set_key_locality(key_id: int, request: Request):
    """Limit a key to local or cloud providers, or allow either (``any``)."""
    try:
        body = await request.json()
    except ValueError:
        raise HTTPException(status_code=400, detail="JSON body required")
    locality = _key_locality(body.get("locality"))
    if not await deps.get_db().set_api_key_locality(key_id, locality):
        raise HTTPException(status_code=404, detail="API key not found")
    return JSONResponse({"id": key_id, "locality": locality})


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
                "extra_headers_text": format_extra_headers(info.provider.extra_headers),
                "manual_models_text": "\n".join(info.provider.manual_models or []),
                "missing_models": missing_models,
                "cached_models": cached_models,
                "cached_only_models": cached_only_models,
                "cache_external_host_set": bool(
                    (settings.cache_external_host or "").strip()
                ),
                "cache_enabled": settings.cache_enabled,
                "family_labels": FAMILY_LABELS,
            },
        ),
    )


def _utc_day_start() -> datetime:
    now = datetime.now(timezone.utc)
    return now.replace(hour=0, minute=0, second=0, microsecond=0, tzinfo=None)


def _usd(value: Any, what: str) -> float | None:
    """A non-negative USD amount from a form or JSON value; blank is None."""
    raw = str(value if value is not None else "").strip().lstrip("$")
    if not raw:
        return None
    try:
        amount = float(raw)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"{what} must be a number")
    if amount < 0 or amount > 1_000_000:
        raise HTTPException(status_code=400, detail=f"{what} is out of range")
    return amount


def _parse_locality(value: str | None) -> str:
    v = (value or "").strip().lower()
    if v not in LOCALITIES:
        raise ValueError("locality must be local or cloud")
    return v


def _key_locality(value: Any) -> str | None:
    """An API key's locality from a request body: local, cloud, or any."""
    v = str(value or "").strip().lower()
    if v in ("", "any", "none", "either"):
        return None
    if v not in LOCALITIES:
        raise HTTPException(
            status_code=400, detail="locality must be local, cloud or any"
        )
    return v


def _model_filter(text: str | None) -> str | None:
    """Model filter patterns, normalised to one per line; None for none."""
    patterns = _model_lines(text)
    return "\n".join(patterns) or None


def _timeout(value: str | None) -> float | None:
    raw = (value or "").strip()
    if not raw:
        return None
    try:
        seconds = float(raw)
    except ValueError:
        raise ValueError("read timeout must be a number of seconds")
    if not 1 <= seconds <= 3600:
        raise ValueError("read timeout must be between 1 and 3600 seconds")
    return seconds


def _model_lines(text: str | None) -> list[str]:
    """Model names, one per line or comma-separated."""
    out: list[str] = []
    for part in (text or "").replace(",", "\n").splitlines():
        name = part.strip()
        if name and name not in out:
            if len(name) > 200:
                raise ValueError("model name too long")
            out.append(name)
    return out


@router.post("/providers/add")
async def add_provider(
    name: str = Form(...),
    url: str = Form(""),
    protocol: str = Form("ollama"),
    machine_type: Optional[str] = Form(None),
    gpu_type: Optional[str] = Form(None),
    gpu_ram: Optional[str] = Form(None),
    api_key: Optional[str] = Form(None),
    api_key_header: Optional[str] = Form(None),
    preset: Optional[str] = Form(None),
    preset_resource: Optional[str] = Form(None),
    extra_headers: Optional[str] = Form(None),
    manual_models: Optional[str] = Form(None),
    locality: Optional[str] = Form(None),
    model_filter: Optional[str] = Form(None),
):
    """Create a provider; *url* becomes its first address. It is probed in the
    background, so the stored endpoint list does not wait on the dialog.

    With *preset*, the cloud preset's URL, protocol, key header and extra
    headers fill whatever the form left blank.
    """
    pm = deps.get_pm()
    settings_: dict[str, Any] = {}
    try:
        proto = ProviderProtocol.parse(protocol)
        chosen = get_preset(preset) if preset else None
        if preset and chosen is None:
            raise ValueError("unknown preset")
        if chosen is not None:
            settings_ = preset_settings(chosen)
            proto = ProviderProtocol(chosen.protocol)
            if chosen.url_params or not url.strip():
                url = chosen.url(resource=preset_resource or "")
            if not (api_key_header or "").strip():
                api_key_header = chosen.api_key_header
        if extra_headers is not None and extra_headers.strip():
            settings_["extra_headers"] = parse_extra_headers(extra_headers)
        if manual_models is not None and manual_models.strip():
            settings_["manual_models"] = _model_lines(manual_models)
        if not url.strip():
            raise ValueError("URL is required")
        if (locality or "auto") != "auto":
            settings_["locality"] = _parse_locality(locality)
        if model_filter is not None and model_filter.strip():
            settings_["model_filter"] = _model_filter(model_filter)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    try:
        await pm.add_provider(
            name,
            url,
            proto,
            machine_type=machine_type or None,
            gpu_type=gpu_type or None,
            gpu_ram=gpu_ram or None,
            api_key=(api_key or "").strip() or None,
            api_key_header=api_key_header or None,
            provider_settings=settings_ or None,
        )
    except Exception:
        logger.exception("add_provider failed")
        raise HTTPException(status_code=400, detail="Could not add provider")
    return RedirectResponse(url="/", status_code=303)


@router.post("/api/providers/probe")
async def api_probe_url(request: Request):
    """Probe a URL before a provider is saved (Add Provider dialog)."""
    try:
        body = await request.json()
    except ValueError:
        raise HTTPException(status_code=400, detail="JSON body required")
    url = str(body.get("url") or "")
    api_key = (str(body.get("api_key") or "")).strip() or None
    header = (str(body.get("api_key_header") or "")).strip() or None
    try:
        headers = parse_extra_headers(str(body.get("extra_headers") or ""))
        chosen = get_preset(str(body.get("preset") or ""))
        if chosen is not None:
            headers = {**chosen.extra_headers, **headers}
            header = header or chosen.api_key_header
        if api_key:
            from ..provider_secrets import auth_headers

            headers.update(auth_headers(api_key, header))
        result = await probe_url(url, headers=headers or None, key_sent=bool(api_key))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    data = result.to_dict()
    data["family_labels"] = FAMILY_LABELS
    return JSONResponse(data)


@router.post("/api/providers/{provider_id}/probe")
async def api_probe_provider(provider_id: int):
    """Probe a saved provider and store the endpoints its upstream supports."""
    pm = deps.get_pm()
    try:
        caps = await pm.probe_provider(provider_id)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    if caps is None:
        raise HTTPException(status_code=404, detail="Provider not found")
    return JSONResponse(caps)


@router.post("/providers/{provider_id}/edit")
async def edit_provider(
    provider_id: int,
    name: str = Form(...),
    protocol: str = Form("ollama"),
    machine_type: Optional[str] = Form(None),
    gpu_type: Optional[str] = Form(None),
    gpu_ram: Optional[str] = Form(None),
    api_key: Optional[str] = Form(None),
    api_key_header: Optional[str] = Form(None),
    clear_api_key: Optional[str] = Form(None),
    extra_headers: Optional[str] = Form(None),
    manual_models: Optional[str] = Form(None),
    settings_form: Optional[str] = Form(None),
    locality: Optional[str] = Form(None),
    model_filter: Optional[str] = Form(None),
    read_timeout_s: Optional[str] = Form(None),
    auto_benchmark: Optional[str] = Form(None),
):
    """Edit provider details. A blank API key keeps the stored one.

    The extra headers and added models are written only when the form says
    it carries them (*settings_form*): an empty field then clears them.
    """
    pm = deps.get_pm()
    settings_: dict[str, Any] = {}
    try:
        proto = ProviderProtocol.parse(protocol)
        if settings_form:
            settings_["extra_headers"] = parse_extra_headers(extra_headers) or None
            settings_["manual_models"] = _model_lines(manual_models) or None
            settings_["model_filter"] = _model_filter(model_filter)
            settings_["read_timeout_s"] = _timeout(read_timeout_s)
            settings_["auto_benchmark"] = _tristate(auto_benchmark)
        if locality:
            settings_["locality"] = _parse_locality(locality)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc) or "Invalid protocol")
    try:
        await pm.update_provider(
            provider_id,
            name,
            proto,
            machine_type=machine_type or None,
            gpu_type=gpu_type or None,
            gpu_ram=gpu_ram or None,
            api_key=(api_key or "").strip() or None,
            api_key_header=api_key_header,
            clear_api_key=bool(clear_api_key),
            provider_settings=settings_ or None,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception:
        logger.exception("update_provider failed (provider_id=%s)", provider_id)
        raise HTTPException(status_code=400, detail="Could not update provider")
    return RedirectResponse(url=_provider_url(provider_id), status_code=303)


@router.post("/providers/{provider_id}/remove")
async def remove_provider(provider_id: int):
    pm = deps.get_pm()
    await pm.remove_provider(provider_id)
    return RedirectResponse(url="/", status_code=303)


@router.post("/providers/{provider_id}/refresh")
async def refresh_provider(provider_id: int):
    pm = deps.get_pm()
    await pm.refresh_provider(provider_id)
    return RedirectResponse(url=_provider_url(provider_id), status_code=303)


@router.post("/api/benchmark")
async def api_start_benchmark(request: Request):
    """Start a benchmark as a background task and return immediately."""
    body = await request.json()
    provider_id = int(body["provider_id"])
    model_name = body["model"]
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


@router.post("/providers/{provider_id}/delete-model/{model_name:path}")
async def delete_model(provider_id: int, model_name: str):
    pm = deps.get_pm()
    try:
        await pm.delete_remote_model(provider_id, model_name)
    except httpx.HTTPStatusError as exc:
        if exc.response.status_code == 404:
            msg = f"Model {model_name} not found on backend (removed from local list)"
            return RedirectResponse(
                url=_provider_url(provider_id, "error=" + quote(msg)),
                status_code=303,
            )
        logger.error(
            "Delete model: backend HTTP %s for provider_id=%s model=%r",
            exc.response.status_code,
            provider_id,
            model_name,
        )
        raise HTTPException(status_code=500, detail="Could not delete model on backend")
    except httpx.HTTPError as exc:
        logger.error(
            "Delete model failed (provider_id=%s, model=%r): %s",
            provider_id,
            model_name,
            describe_httpx_error(exc),
        )
        raise HTTPException(status_code=500, detail="Could not delete model on backend")
    except Exception:
        logger.exception(
            "Delete model failed (provider_id=%s, model=%r)",
            provider_id,
            model_name,
        )
        raise HTTPException(status_code=500, detail="Could not delete model on backend")
    return RedirectResponse(url=_provider_url(provider_id), status_code=303)


@router.post("/providers/{provider_id}/addresses/add")
async def add_address(
    provider_id: int,
    url: str = Form(...),
    is_preferred: Optional[str] = Form(None),
):
    pm = deps.get_pm()
    try:
        await pm.add_address(provider_id, url, is_preferred=bool(is_preferred))
    except Exception:
        logger.exception("add_address failed (provider_id=%s)", provider_id)
        raise HTTPException(status_code=400, detail="Could not add address")
    return RedirectResponse(url=_provider_url(provider_id), status_code=303)


@router.post("/providers/{provider_id}/addresses/save")
async def save_provider_address(
    provider_id: int,
    url: str = Form(...),
    is_preferred: Optional[str] = Form(None),
    address_id: Optional[str] = Form(None),
):
    """Add or update an address; used by dashboard forms with a stable action URL."""
    db = deps.get_db()
    pm = deps.get_pm()
    pref = bool(is_preferred)
    raw_id = (address_id or "").strip()
    try:
        if not raw_id:
            await pm.add_address(provider_id, url, is_preferred=pref)
        else:
            aid = int(raw_id)
            addr = await db.get_address(aid)
            if addr is None or addr.provider_id != provider_id:
                raise HTTPException(status_code=400, detail="Invalid address")
            await pm.update_address(aid, url, is_preferred=pref)
    except HTTPException:
        raise
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid address")
    except Exception:
        logger.exception(
            "save_provider_address failed (provider_id=%s address_id=%r)",
            provider_id,
            raw_id or None,
        )
        raise HTTPException(status_code=400, detail="Could not save address")
    return RedirectResponse(url=_provider_url(provider_id), status_code=303)


@router.post("/providers/{provider_id}/addresses/{address_id}/edit")
async def edit_address(
    provider_id: int,
    address_id: int,
    url: str = Form(...),
    is_preferred: Optional[str] = Form(None),
):
    pm = deps.get_pm()
    try:
        await pm.update_address(address_id, url, is_preferred=bool(is_preferred))
    except Exception:
        logger.exception(
            "update_address failed (provider_id=%s address_id=%s)",
            provider_id,
            address_id,
        )
        raise HTTPException(status_code=400, detail="Could not update address")
    return RedirectResponse(url=_provider_url(provider_id), status_code=303)


@router.post("/providers/{provider_id}/addresses/{address_id}/remove")
async def remove_address(provider_id: int, address_id: int):
    pm = deps.get_pm()
    await pm.remove_address(address_id)
    return RedirectResponse(url=_provider_url(provider_id), status_code=303)


@router.post("/providers/{provider_id}/addresses/{address_id}/toggle-preferred")
async def toggle_preferred(provider_id: int, address_id: int):
    pm = deps.get_pm()
    await pm.toggle_address_preferred(address_id)
    return RedirectResponse(url=_provider_url(provider_id), status_code=303)


@router.post("/api/pull")
async def api_pull_model(request: Request):
    """Start a pull as a background task and return immediately."""
    body = await request.json()
    model = body.get("model")
    provider_id = body.get("provider_id")
    pull_api = (body.get("pull_api") or "auto").strip().lower()
    if not model:
        raise HTTPException(status_code=400, detail="model is required")
    if pull_api not in {"auto", "ollama", "openai", "llamacpp"}:
        raise HTTPException(status_code=400, detail="Invalid pull_api value")
    if pull_api in {"openai", "llamacpp"}:
        raise HTTPException(
            status_code=400,
            detail="pulling over the OpenAI API is not supported; use ollama",
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
            if i.provider.is_ollama
            and not i.provider.lacks("/api/pull")
            and i.provider.id is not None
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


@router.get("/api/hf/env")
async def api_hf_env():
    """Hugging Face settings the hf CLI will see (secrets masked)."""
    try:
        hf_download.hf_executable()
        available = True
    except hf_download.HfError:
        available = False
    return JSONResponse(
        {
            "hf_available": available,
            "models_dir": str(hf_download.models_root()),
            "env": hf_download.env_summary(),
        }
    )


@router.post("/api/hf/files")
async def api_hf_files(request: Request):
    """List the files a repository and pattern select, without downloading."""
    body = await request.json()
    try:
        repo = hf_download.validate_repo(str(body.get("repo") or ""))
        pattern = hf_download.validate_pattern(body.get("pattern"))
        revision = hf_download.validate_revision(body.get("revision"))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    try:
        files = await hf_download.plan(repo, pattern, revision)
    except hf_download.HfError as exc:
        raise HTTPException(status_code=502, detail=str(exc))
    models = hf_download.model_files([f["file"] for f in files])
    suggested = (
        hf_download.default_ollama_name(repo, models[0]) if len(models) == 1 else None
    )
    return JSONResponse(
        {
            "repo": repo,
            "pattern": pattern,
            "files": files,
            "model_files": models,
            "split": hf_download.split_parts(models),
            "suggested_name": suggested,
        }
    )


async def _hf_deliver_ollama(
    *,
    pm: Any,
    db: Any,
    pid: int,
    path: Path,
    digest: str,
    model_name: str,
    entry: dict,
    lock: asyncio.Lock,
) -> None:
    provider = await db.get_provider(pid)
    pname = provider.name if provider else str(pid)
    start = time.monotonic()
    status, detail = "ok", None
    try:
        client = pm.get_ollama_client(pid)
        if await client.has_blob(digest):
            async with lock:
                entry["progress"] = f"{pname}: blob already present"
        else:

            def on_sent(sent: int, total: int) -> None:
                pct = int(sent * 100 / total) if total else 0
                entry["progress"] = f"{pname}: uploading {pct}%"

            await client.push_blob(digest, str(path), on_sent)
        async with lock:
            entry["progress"] = f"{pname}: creating {model_name}"
        await client.create_from_files(model_name, {path.name: digest})
        await pm.refresh_provider(pid)
        async with lock:
            entry["completed"].append(pid)
    except Exception as exc:
        status = "error"
        detail = (
            describe_httpx_error(exc) if isinstance(exc, httpx.HTTPError) else str(exc)
        )
        logger.error("HF push of %s to %s failed: %s", model_name, pname, detail)
        async with lock:
            entry["failed"].append(pid)
            entry["progress"] = f"{pname}: failed: {detail}"
    await db.save_request_log(
        RequestLog(
            provider_id=pid,
            provider_name=pname,
            protocol="system",
            endpoint="hf-pull",
            source_ip="internal",
            model=model_name,
            duration_ms=(time.monotonic() - start) * 1000,
            status=status,
            error_detail=detail,
        )
    )


@router.post("/api/hf/pull")
async def api_hf_pull(request: Request):
    """Download from Hugging Face, then push to Ollama providers and/or refresh
    llama.cpp providers that share the models directory."""
    body = await request.json()
    try:
        repo = hf_download.validate_repo(str(body.get("repo") or ""))
        pattern = hf_download.validate_pattern(body.get("pattern"))
        revision = hf_download.validate_revision(body.get("revision"))
        name_raw = str(body.get("model_name") or "").strip()
        model_name = hf_download.validate_ollama_name(name_raw) if name_raw else None
        push_ids = [int(x) for x in body.get("push_provider_ids") or []]
        refresh_ids = [int(x) for x in body.get("refresh_provider_ids") or []]
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    pm = deps.get_pm()
    db = deps.get_db()
    for pid in push_ids + refresh_ids:
        target = await db.get_provider(pid)
        if target is None:
            raise HTTPException(status_code=400, detail=f"unknown provider {pid}")
        if pid in push_ids and target.is_worker:
            # The worker link buffers whole request bodies; a multi-GB upload
            # through it would exhaust memory on the hub.
            raise HTTPException(
                status_code=400,
                detail=f"{target.name} is a worker; pull the model on the worker "
                "itself instead of pushing it through the hub",
            )

    pull_id = str(uuid.uuid4())
    label = model_name or f"{repo} ({pattern})"
    entry = {
        "model": label,
        "pull_api": "huggingface",
        "provider_ids": push_ids + refresh_ids,
        "status": "pulling",
        "completed": [],
        "failed": [],
        "progress": "starting download",
    }
    _active_pulls[pull_id] = entry
    lock = asyncio.Lock()

    async def _run() -> None:
        try:

            def on_progress(text: str) -> None:
                entry["progress"] = f"{repo}: {text}"

            paths = await hf_download.download(repo, pattern, revision, on_progress)
            models = [
                p
                for p in paths
                if p.name in hf_download.model_files([q.name for q in paths])
            ]
            if push_ids:
                # A refusal here fails only the Ollama targets; shared-volume
                # refreshes below still run, since the files are on disk.
                refusal: str | None = None
                if not models:
                    refusal = "no GGUF model file was downloaded"
                elif hf_download.split_parts([p.name for p in models]):
                    refusal = (
                        "split GGUF files cannot be pushed to Ollama; "
                        "use a single-file quant or the shared volume"
                    )
                elif len(models) > 1:
                    refusal = (
                        f"the pattern matched {len(models)} GGUF files; "
                        "narrow it to one quant to push to Ollama"
                    )
                if refusal is not None:
                    logger.warning("HF push of %s refused: %s", repo, refusal)
                    async with lock:
                        entry["failed"].extend(push_ids)
                        entry["progress"] = f"failed: {refusal}"
                else:
                    path = models[0]
                    name = model_name or hf_download.default_ollama_name(
                        repo, path.name
                    )
                    entry["model"] = name
                    entry["progress"] = f"hashing {path.name}"
                    digest = await hf_download.sha256_file(path)
                    await asyncio.gather(
                        *(
                            _hf_deliver_ollama(
                                pm=pm,
                                db=db,
                                pid=pid,
                                path=path,
                                digest=digest,
                                model_name=name,
                                entry=entry,
                                lock=lock,
                            )
                            for pid in push_ids
                        )
                    )
            for pid in refresh_ids:
                try:
                    await pm.refresh_provider(pid)
                    async with lock:
                        entry["completed"].append(pid)
                except Exception:
                    logger.exception("refresh after HF download failed (%s)", pid)
                    async with lock:
                        entry["failed"].append(pid)
            if not (push_ids or refresh_ids):
                entry["progress"] = (
                    f"saved {len(paths)} file(s) in the models directory"
                )
        except Exception as exc:
            logger.error("HF download of %s failed: %s", repo, exc)
            async with lock:
                done = set(entry["completed"]) | set(entry["failed"])
                entry["failed"].extend(
                    p for p in entry["provider_ids"] if p not in done
                )
                if not entry["provider_ids"]:
                    entry["failed"].append(0)
                entry["progress"] = f"failed: {exc}"
        finally:
            async with lock:
                entry["status"] = "done"
                entry["completed_at"] = time.monotonic()

    asyncio.create_task(_run())
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
        if not provider.is_ollama or provider.lacks("/api/pull") or provider.id is None:
            continue
        existing_ollama = {m.name for m in info.models}
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


@router.get("/api/auto-benchmark")
async def api_auto_benchmark_status():
    """Idle auto-benchmark settings, what is running, what is due, recent results."""
    return JSONResponse(jsonable_encoder(await deps.get_auto_benchmark().status()))


@router.post("/api/auto-benchmark")
async def api_auto_benchmark_configure(request: Request):
    """Body: any of ``enabled`` (bool), ``idle_minutes``, ``max_age_days``."""
    body = await request.json()
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
        raise HTTPException(status_code=400, detail=str(exc))
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
    return RedirectResponse(url=_provider_url(provider_id), status_code=303)


@router.post("/models/pull-all")
async def pull_model_all_legacy(model: str = Form(...)):
    """Legacy form-based pull-all — redirects immediately, pull runs in background."""
    pm = deps.get_pm()
    db = deps.get_db()
    infos = await pm.list_provider_infos()
    provider_ids = [
        i.provider.id
        for i in infos
        if i.provider.is_ollama
        and not i.provider.lacks("/api/pull")
        and i.provider.id is not None
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


def suggest_aliases(
    names_by_provider: dict[int, set[str]], aliases: dict[str, str]
) -> list[dict[str, str]]:
    """Likely aliases between providers' model names.

    ``vendor/model`` (OpenRouter, Together, ...) and ``model:latest``
    (Ollama) are suggested as aliases of ``model`` when another provider
    serves that bare name. Existing aliases are not suggested again.
    """
    everywhere: dict[str, set[int]] = {}
    for pid, names in names_by_provider.items():
        for n in names:
            everywhere.setdefault(n, set()).add(pid)
    out: list[dict[str, str]] = []
    seen: set[str] = set()
    for name, pids in sorted(everywhere.items()):
        candidates = []
        if "/" in name:
            candidates.append(name.rsplit("/", 1)[1])
        if name.endswith(":latest"):
            candidates.append(name[: -len(":latest")])
        for bare in candidates:
            if not bare or bare == name or name in aliases or name in seen:
                continue
            if everywhere.get(bare, set()) - pids:
                out.append({"alias": name, "canonical": bare})
                seen.add(name)
    return out


@router.get("/api/model-aliases")
async def list_model_aliases():
    """All aliases (alias -> canonical) and suggested new ones."""
    db = deps.get_db()
    aliases = await db.get_model_aliases()
    names: dict[int, set[str]] = {}
    for m in await db.get_all_provider_models():
        names.setdefault(m.provider_id, set()).add(m.name)
    return JSONResponse(
        {"aliases": aliases, "suggestions": suggest_aliases(names, aliases)}
    )


@router.post("/api/model-aliases")
async def set_model_alias(request: Request):
    """Body: {alias, canonical}. The alias then reaches every provider of the
    canonical name and of its other aliases."""
    body = await request.json()
    alias = str(body.get("alias") or "").strip()
    canonical = str(body.get("canonical") or "").strip()
    if not alias or not canonical:
        raise HTTPException(status_code=400, detail="alias and canonical are required")
    if alias == canonical:
        raise HTTPException(status_code=400, detail="a model cannot alias itself")
    if len(alias) > 255 or len(canonical) > 255:
        raise HTTPException(status_code=400, detail="model name too long")
    db = deps.get_db()
    aliases = await db.get_model_aliases()
    # Refuse a loop: following canonical's chain must never reach alias.
    cur, seen = canonical, set()
    while cur not in seen:
        if cur == alias:
            raise HTTPException(status_code=400, detail="that alias would form a loop")
        seen.add(cur)
        if cur not in aliases:
            break
        cur = aliases[cur]
    await db.set_model_alias(alias, canonical)
    return JSONResponse({"alias": alias, "canonical": canonical})


@router.delete("/api/model-aliases/{alias:path}")
async def delete_model_alias(alias: str):
    if not await deps.get_db().remove_model_alias(alias):
        raise HTTPException(status_code=404, detail="alias not found")
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
        if i.provider.is_ollama and any(m.name == model for m in i.models)
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


# --- Workers: member routers that dial in to this hub ---

_WORKER_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{1,63}$")


def _iso(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return value.isoformat()
    return str(value)


async def _worker_rows(db: Any, pm: Any) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for w in await db.list_workers():
        pid = w["provider_id"]
        conn = pm.worker_hub.get(pid) if pid is not None else None
        live = conn.summary() if conn is not None and not conn.closed else None
        models = await db.get_provider_models(pid) if pid is not None else []
        out.append(
            {
                "id": w["id"],
                "name": w["name"],
                "token_prefix": w["token_prefix"],
                "provider_id": pid,
                "enabled": w["enabled"],
                "created_at": _iso(w["created_at"]),
                "last_connected_at": _iso(w["last_connected_at"]),
                "last_remote_addr": w["last_remote_addr"],
                "connected": live is not None,
                "live": live,
                "model_count": len(models),
            }
        )
    return out


def _hub_url_hint(request: Request) -> str:
    host = request.url.hostname or "this-hub"
    scheme = "https" if request.url.scheme == "https" else "http"
    return f"{scheme}://{host}:{settings.api_port}"


class WorkerCreateBody(BaseModel):
    name: str


class WorkerEnabledBody(BaseModel):
    enabled: bool


class WorkerLinkBody(BaseModel):
    hub_url: str = ""
    token: Optional[str] = None


@router.get("/api/workers")
async def api_list_workers():
    return JSONResponse({"workers": await _worker_rows(deps.get_db(), deps.get_pm())})


@router.post("/api/workers")
async def api_create_worker(request: Request, body: WorkerCreateBody):
    db = deps.get_db()
    pm = deps.get_pm()
    name = body.name.strip().lower()
    if not _WORKER_NAME_RE.match(name):
        raise HTTPException(
            status_code=400,
            detail="Worker name must be 2-64 characters: letters, digits, '.', '_', '-'",
        )
    if await db.get_provider_by_name(name) is not None:
        raise HTTPException(
            status_code=409, detail=f"A provider named {name!r} already exists"
        )
    provider = await pm.add_worker_provider(name)
    assert provider.id is not None
    token = generate_worker_token()
    worker_id = await db.create_worker(
        name, worker_token_hash(token), worker_token_prefix(token), provider.id
    )
    hub_url = _hub_url_hint(request)
    return JSONResponse(
        {
            "id": worker_id,
            "name": name,
            "provider_id": provider.id,
            "token": token,
            "hub_url": hub_url,
            "env_example": (
                f"LLAMA_ROUTER_WORKER_HUB_URL={hub_url}\n"
                f"LLAMA_ROUTER_WORKER_TOKEN={token}"
            ),
        }
    )


@router.post("/api/workers/{worker_id}/rotate")
async def api_rotate_worker_token(worker_id: int):
    db = deps.get_db()
    pm = deps.get_pm()
    worker = await db.get_worker(worker_id)
    if not worker:
        raise HTTPException(status_code=404, detail="Worker not found")
    token = generate_worker_token()
    await db.set_worker_token(
        worker_id, worker_token_hash(token), worker_token_prefix(token)
    )
    if worker["provider_id"] is not None:
        await pm.worker_hub.disconnect(
            worker["provider_id"], CLOSE_REVOKED, "token rotated"
        )
    return JSONResponse({"id": worker_id, "token": token})


@router.post("/api/workers/{worker_id}/enabled")
async def api_set_worker_enabled(worker_id: int, body: WorkerEnabledBody):
    db = deps.get_db()
    pm = deps.get_pm()
    worker = await db.get_worker(worker_id)
    if not worker:
        raise HTTPException(status_code=404, detail="Worker not found")
    await db.set_worker_enabled(worker_id, body.enabled)
    if not body.enabled and worker["provider_id"] is not None:
        await pm.worker_hub.disconnect(
            worker["provider_id"], CLOSE_REVOKED, "worker disabled by hub admin"
        )
    return JSONResponse({"id": worker_id, "enabled": body.enabled})


@router.delete("/api/workers/{worker_id}")
async def api_delete_worker(worker_id: int):
    db = deps.get_db()
    pm = deps.get_pm()
    worker = await db.get_worker(worker_id)
    if not worker:
        raise HTTPException(status_code=404, detail="Worker not found")
    if worker["provider_id"] is not None:
        await pm.remove_provider(worker["provider_id"])
    await db.delete_worker(worker_id)
    return JSONResponse({"deleted": worker_id})


@router.get("/api/worker-link")
async def api_worker_link_status():
    return JSONResponse(await deps.get_worker_link().status())


@router.post("/api/worker-link")
async def api_worker_link_save(body: WorkerLinkBody):
    link = deps.get_worker_link()
    if link.env_configured():
        raise HTTPException(
            status_code=409,
            detail="The worker link is set by environment variables on this host",
        )
    hub_url = body.hub_url.strip()
    token = body.token.strip() if body.token is not None else None
    if hub_url:
        try:
            normalized = normalize_hub_url(hub_url)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        if not urlparse(normalized).hostname:
            raise HTTPException(status_code=400, detail="Hub URL needs a host name")
    else:
        token = ""
    await link.save(hub_url, token)
    return JSONResponse(await link.status())
