"""API Keys tab."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta

from fastapi import APIRouter, HTTPException, Request
from fastapi.encoders import jsonable_encoder
from fastapi.responses import (
    JSONResponse,
)

from ...auth import generate_api_key, key_prefix
from ...provider_secrets import key_hash
from ...router import (
    ROUTING_MODES,
)
from ...routing_table import DEFAULT_MODE
from .. import deps
from ..form_values import _key_label, _key_locality, _tristate, _usd
from .workers import _iso

router = APIRouter()


async def _api_key_exists(db, key_id: int) -> bool:
    keys = await db.list_api_keys()
    return any(int(k["id"]) == int(key_id) for k in keys)


@router.get("/api/keys")
async def api_list_keys():
    """API keys with their settings; never the keys themselves."""
    keys = await deps.get_db().list_api_keys()
    return JSONResponse(jsonable_encoder({"keys": keys}))


# Periods the usage report covers; None is everything in the request log.
_USAGE_PERIODS: dict[str, timedelta | None] = {
    "24h": timedelta(hours=24),
    "7d": timedelta(days=7),
    "30d": timedelta(days=30),
    "all": None,
}


@router.get("/api/keys/usage")
async def api_key_usage(period: str = "7d", key_id: int | None = None):
    """Requests, tokens and cost per API key and provider over *period*
    (24h, 7d, 30d or all), from the request log."""
    if period not in _USAGE_PERIODS:
        raise HTTPException(
            status_code=400,
            detail=f"period must be one of {', '.join(_USAGE_PERIODS)}",
        )
    span = _USAGE_PERIODS[period]
    since = datetime.now(UTC).replace(tzinfo=None) - span if span else None
    rows = await deps.get_db().usage_by_key_and_provider(since, key_id)
    for row in rows:
        row["last_request_at"] = _iso(row["last_request_at"])
    return JSONResponse(
        {
            "period": period,
            "since": _iso(since),
            "key_id": key_id,
            "rows": rows,
        }
    )


@router.post("/api/keys/generate")
async def api_generate_key(request: Request):
    db = deps.get_db()
    body = await deps.json_body(request)
    routing_mode = (body.get("routing_mode") or DEFAULT_MODE).strip().lower()
    allow_fallback = bool(body.get("allow_fallback", True))
    session_affinity = _tristate(body.get("session_affinity"))
    locality = _key_locality(body.get("locality"))
    cap = _usd(body.get("daily_spend_cap_usd"), "daily spend cap")
    label = _key_label(body.get("label"))
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
        label=label,
    )
    return JSONResponse(
        {
            "api_key": plaintext,
            "label": label,
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
    body = await deps.json_body(request)
    cap = _usd(body.get("daily_spend_cap_usd"), "daily spend cap")
    if not await deps.get_db().set_api_key_spend_cap(key_id, cap):
        raise HTTPException(status_code=404, detail="API key not found")
    return JSONResponse({"id": key_id, "daily_spend_cap_usd": cap})


@router.post("/api/keys/{key_id}/locality")
async def api_set_key_locality(key_id: int, request: Request):
    """Limit a key to local or cloud providers, or allow either (``any``)."""
    body = await deps.json_body(request)
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


PIN_MODEL_NAME_MAX = 255


def validate_pin_model_name(raw: object) -> str:
    """Return a pinned route's model name, or raise 400.

    The name is shown in the dashboard and used as a path segment, so it must
    be printable (no control characters) and at most 255 characters.
    """
    name = str(raw or "").strip()
    if not name:
        raise HTTPException(status_code=400, detail="model_name is required")
    if len(name) > PIN_MODEL_NAME_MAX:
        raise HTTPException(
            status_code=400,
            detail=f"model_name must be at most {PIN_MODEL_NAME_MAX} characters",
        )
    if not name.isprintable():
        raise HTTPException(status_code=400, detail="model_name must be printable text")
    return name


@router.post("/api/keys/{key_id}/pins")
async def api_set_key_pin(key_id: int, request: Request):
    db = deps.get_db()
    if not await _api_key_exists(db, key_id):
        raise HTTPException(status_code=404, detail="API key not found")
    body = await deps.json_body(request)
    provider_id_raw = body.get("provider_id")
    if not body.get("model_name") or provider_id_raw is None:
        raise HTTPException(
            status_code=400, detail="model_name and provider_id are required"
        )
    model_name = validate_pin_model_name(body.get("model_name"))
    try:
        provider_id = int(provider_id_raw)
    except TypeError, ValueError:
        raise HTTPException(
            status_code=400, detail="provider_id must be an integer"
        ) from None
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
    body = await deps.json_body(request)
    allow = bool(body.get("allow", True))
    await db.set_allow_unauthenticated(allow)
    return JSONResponse({"allow_unauthenticated": allow})
