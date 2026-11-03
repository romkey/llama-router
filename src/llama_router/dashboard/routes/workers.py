"""Workers tab: member routers that dial in, and this router's own hub link."""

from __future__ import annotations

import re
from datetime import UTC, datetime
from typing import Any
from urllib.parse import urlparse

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import (
    JSONResponse,
)
from pydantic import BaseModel

from ... import router_identity
from ...config import settings
from ...worker_link.hub import (
    generate_worker_token,
    worker_token_expired,
    worker_token_hash,
    worker_token_prefix,
)
from ...worker_link.protocol import CLOSE_REVOKED, EXPOSE_INDIVIDUAL
from ...worker_link.worker import normalize_hub_url
from .. import deps, management_keys
from ..auth_core import _external_origin

router = APIRouter()


# --- Workers: member routers that dial in to this hub ---

_WORKER_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{1,63}$")


def _iso(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        if value.tzinfo is None:
            value = value.replace(tzinfo=UTC)
        return value.isoformat()
    return str(value)


async def _worker_rows(db: Any, pm: Any) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for w in await db.list_workers():
        pid = w["provider_id"]
        conn = pm.worker_hub.get(pid) if pid is not None else None
        live = conn.summary() if conn is not None and not conn.closed else None
        models = await db.get_provider_models(pid) if pid is not None else []
        individual = live is not None and live.get("expose") == EXPOSE_INDIVIDUAL
        out.append(
            {
                "id": w["id"],
                "name": w["name"],
                "token_prefix": w["token_prefix"],
                "token_expires_at": w["token_expires_at"],
                # UTC, as the ``localtime`` filter expects.
                "token_expires": (
                    None
                    if w["token_expires_at"] is None
                    else datetime.fromtimestamp(w["token_expires_at"], UTC).strftime(
                        "%Y-%m-%d %H:%M:%S"
                    )
                ),
                "token_expired": worker_token_expired(w),
                "provider_id": pid,
                "enabled": w["enabled"],
                "created_at": _iso(w["created_at"]),
                "last_connected_at": _iso(w["last_connected_at"]),
                "last_remote_addr": w["last_remote_addr"],
                "connected": live is not None,
                "live": live,
                "model_count": len(models),
                "models_label": (
                    f"{live['shared_providers'] or 0} providers"
                    if individual and live is not None
                    else str(len(models))
                ),
            }
        )
    return out


def _hub_url_hint(request: Request) -> str:
    if settings.api_on_dashboard_port:
        # Workers can use the address this page was opened on, proxy and all.
        scheme, netloc = _external_origin(request)
        return f"{scheme}://{netloc or 'this-hub'}"
    scheme = "https" if request.url.scheme == "https" else "http"
    host = request.url.hostname or "this-hub"
    return f"{scheme}://{host}:{settings.api_port}"


class WorkerCreateBody(BaseModel):
    name: str
    expires_in_days: int | None = None


class WorkerRotateBody(BaseModel):
    expires_in_days: int | None = None


class WorkerEnabledBody(BaseModel):
    enabled: bool


class WorkerExposeBody(BaseModel):
    expose: str


class WorkerLinkBody(BaseModel):
    hub_url: str = ""
    token: str | None = None


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
            detail=(
                "Worker name must be 2-64 characters: letters, digits, '.', '_', '-'"
            ),
        )
    if await db.get_provider_by_name(name) is not None:
        raise HTTPException(
            status_code=409, detail=f"A provider named {name!r} already exists"
        )
    expires_at = management_keys.expiry_from_days(body.expires_in_days)
    provider = await pm.add_worker_provider(name)
    token = generate_worker_token()
    worker_id = await db.create_worker(
        name,
        worker_token_hash(token),
        worker_token_prefix(token),
        provider.require_id(),
        expires_at,
    )
    hub_url = _hub_url_hint(request)
    return JSONResponse(
        {
            "id": worker_id,
            "name": name,
            "provider_id": provider.id,
            "token": token,
            "token_expires_at": expires_at,
            "hub_url": hub_url,
            "env_example": (
                f"LLAMA_ROUTER_WORKER_HUB_URL={hub_url}\n"
                f"LLAMA_ROUTER_WORKER_TOKEN={token}"
            ),
        }
    )


@router.post("/api/workers/{worker_id}/rotate")
async def api_rotate_worker_token(worker_id: int, body: WorkerRotateBody | None = None):
    db = deps.get_db()
    pm = deps.get_pm()
    worker = await db.get_worker(worker_id)
    if not worker:
        raise HTTPException(status_code=404, detail="Worker not found")
    expires_at = management_keys.expiry_from_days(
        body.expires_in_days if body else None
    )
    token = generate_worker_token()
    await db.set_worker_token(
        worker_id, worker_token_hash(token), worker_token_prefix(token), expires_at
    )
    if worker["provider_id"] is not None:
        await pm.worker_hub.disconnect(
            worker["provider_id"], CLOSE_REVOKED, "token rotated"
        )
    return JSONResponse(
        {"id": worker_id, "token": token, "token_expires_at": expires_at}
    )


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


class HubProvidersBody(BaseModel):
    provider_ids: list[int]


def _hub_kind(provider: Any) -> str:
    """Quick-pick group of a provider: local, cloud or remote (a worker)."""
    if provider.is_worker:
        return "remote"
    return str(provider.locality)


async def _hub_provider_rows(db: Any) -> list[dict[str, Any]]:
    return [
        {
            "id": p.id,
            "name": p.name,
            "status": p.status.value,
            "locality": p.locality,
            "is_worker": p.is_worker,
            "kind": _hub_kind(p),
            "hub_shared": p.hub_shared,
        }
        for p in await db.list_providers()
    ]


@router.get("/api/hub-providers")
async def api_hub_providers():
    """Which providers a hub this router works for may use."""
    return JSONResponse({"providers": await _hub_provider_rows(deps.get_db())})


@router.post("/api/hub-providers")
async def api_set_hub_providers(body: HubProvidersBody):
    """Share exactly *provider_ids* with the hub; every other provider is
    kept from it. Takes effect on the hub's next request and model listing."""
    db = deps.get_db()
    known = {p.id for p in await db.list_providers()}
    unknown = sorted(set(body.provider_ids) - known)
    if unknown:
        raise HTTPException(status_code=400, detail=f"unknown provider ids {unknown}")
    await db.set_hub_shared_providers(body.provider_ids)
    return JSONResponse({"providers": await _hub_provider_rows(db)})


@router.get("/api/worker-link")
async def api_worker_link_status():
    return JSONResponse(await deps.get_worker_link().status())


@router.post("/api/worker-link/router-id")
async def api_new_router_id():
    """Give this router a new ID (for a copy of another router's data).

    Both links reconnect so the other side learns it: the one to this
    router's hub, and every worker of this router, which reconnects on its own.
    """
    link = deps.get_worker_link()
    await router_identity.regenerate(deps.get_db())
    await link.reload()
    await deps.get_pm().worker_hub.close_all()
    return JSONResponse(await link.status())


@router.post("/api/worker-link/expose")
async def api_worker_link_expose(body: WorkerExposeBody):
    """Share this router's providers with the hub as one provider or each on
    its own. The link reconnects so the hub sees the change."""
    link = deps.get_worker_link()
    if link.expose_env_locked():
        raise HTTPException(
            status_code=409,
            detail="How providers are shared is set by LLAMA_ROUTER_WORKER_EXPOSE",
        )
    try:
        await link.save_expose(body.expose.strip().lower())
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return JSONResponse(await link.status())


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
