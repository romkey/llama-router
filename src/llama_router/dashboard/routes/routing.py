"""Routing tab: the routing table and routing explanations."""

from __future__ import annotations

import ipaddress

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import (
    JSONResponse,
)

from ...affinity import session_key
from ...auth import model_access
from ...router import (
    ENDPOINT_PROTOCOLS,
    ROUTING_MODES,
    RoutingPreferences,
)
from ...routing_table import DEFAULT_MODE
from .. import deps, management_keys
from ..form_values import _key_locality, _tristate
from ..pages import _is_admin

router = APIRouter()


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


def _management_key_may_read_keys(request: Request) -> bool:
    key = getattr(request.state, "management_key", None)
    return key is not None and management_keys.has_privilege(
        key.privileges, "keys:read"
    )


@router.get("/api/routing/explain")
async def api_explain_routing(request: Request):
    """How a request would be routed right now, and why.

    Query: ``model`` (required), ``endpoint`` (default /v1/chat/completions),
    and either ``mode`` (balanced, latency, throughput, chaos) with optional
    ``allow_fallback`` and ``session_affinity``, or ``key_id`` to use an API
    key's mode, fallback, session affinity setting and pins (admins only).
    ``session_id`` shows how a client sending that ``X-Session-ID`` would be
    routed, including its current binding. Without ``key_id``, ``client_ip``
    names the client address, which scopes anonymous sessions.
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
        if not (_is_admin(request) or _management_key_may_read_keys(request)):
            raise HTTPException(status_code=403, detail="Forbidden")
        try:
            key_id = int(raw_key_id)
        except ValueError as exc:
            raise HTTPException(
                status_code=400, detail="key_id must be a number"
            ) from exc
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
            access=await model_access(db, key_id),
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
    client = (q.get("client_ip") or "").strip() or None
    if client is not None:
        try:
            client = str(ipaddress.ip_address(client))
        except ValueError as exc:
            raise HTTPException(
                status_code=400, detail="client_ip must be an IP address"
            ) from exc
    session = (
        session_key({"x-session-id": session_id}, {"model": model}, key_id, client)
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
