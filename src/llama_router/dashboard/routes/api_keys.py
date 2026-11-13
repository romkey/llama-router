"""API Keys tab."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from fastapi.encoders import jsonable_encoder
from fastapi.responses import (
    JSONResponse,
)

from ...auth import key_prefix, new_api_key
from ...config import settings
from ...key_lifecycle import (
    holds_by_default,
    now_utc,
    parse_expiry,
    renewal_mode,
    renewal_period,
    renewed_expiry,
    set_holds_by_default,
    set_renewal_mode,
    with_status,
)
from ...router import (
    ROUTING_MODES,
)
from ...routing_table import DEFAULT_MODE
from .. import deps, management_keys
from ..form_values import (
    ADMIN_KEY_FIELDS,
    _key_label,
    _key_locality,
    _tristate,
    _usd,
    key_edits,
    strict_pins_flag,
)
from ..key_access import UNRESTRICTED, parse_access, store_access, with_access
from ..pages import _require_admin
from .workers import _iso

router = APIRouter()


async def _api_key_exists(db, key_id: int) -> bool:
    return await db.get_api_key(key_id) is not None


@router.get("/api/keys")
async def api_list_keys():
    """API keys with their settings; never the keys themselves."""
    db = deps.get_db()
    keys = with_status(await db.list_api_keys())
    return JSONResponse(
        jsonable_encoder(
            {
                "keys": keys,
                "lifecycle": await _lifecycle_settings(db),
            }
        )
    )


async def _lifecycle_settings(db: Any) -> dict[str, Any]:
    return {
        "renewal_mode": await renewal_mode(db),
        "holds_by_default": await holds_by_default(db),
        "max_key_lifetime_days": settings.max_key_lifetime_days,
        "renewal_days": renewal_period().days,
    }


@router.put("/api/keys/lifecycle-settings")
async def api_set_lifecycle_settings(request: Request):
    """Body: {renewal_mode?: off|allowed|approval, holds_by_default?: bool}.
    Admins only (not management keys): system-wide options."""
    _require_admin(request)
    db = deps.get_db()
    body = await deps.json_body(request)
    if "renewal_mode" in body:
        await set_renewal_mode(db, str(body["renewal_mode"]).strip().lower())
    if "holds_by_default" in body:
        if not isinstance(body["holds_by_default"], bool):
            raise HTTPException(
                status_code=400, detail="holds_by_default must be true or false"
            )
        await set_holds_by_default(db, body["holds_by_default"])
    return JSONResponse(await _lifecycle_settings(db))


@router.get("/api/keys/renewal-requests")
async def api_renewal_requests():
    """Keys whose owners asked for a renewal an admin must approve."""
    keys = with_status(await deps.get_db().list_api_key_renewal_requests())
    return JSONResponse(jsonable_encoder({"keys": keys}))


async def _key_or_404(db: Any, key_id: int) -> dict[str, Any]:
    key = await db.get_api_key(key_id)
    if key is None:
        raise HTTPException(status_code=404, detail="API key not found")
    return dict(key)


async def _status_reply(db: Any, key_id: int) -> JSONResponse:
    key = await _key_or_404(db, key_id)
    return JSONResponse(jsonable_encoder(with_status([key])[0]))


@router.post("/api/keys/{key_id:int}/renew")
async def api_renew_key(key_id: int):
    """Give the key a new expiry (now plus the renewal period); the secret
    stays. Works on an expired key, and whatever the renewal option."""
    db = deps.get_db()
    key = await _key_or_404(db, key_id)
    expiry = renewed_expiry(key.get("owner_user_id") is not None)
    await db.set_api_key_expiry(key_id, expiry)
    return await _status_reply(db, key_id)


@router.post("/api/keys/{key_id:int}/renewal/approve")
async def api_approve_renewal(key_id: int):
    db = deps.get_db()
    key = await _key_or_404(db, key_id)
    if key.get("renewal_requested_at") is None:
        raise HTTPException(status_code=409, detail="No renewal was requested")
    await db.set_api_key_expiry(
        key_id, renewed_expiry(key.get("owner_user_id") is not None)
    )
    return await _status_reply(db, key_id)


@router.post("/api/keys/{key_id:int}/renewal/deny")
async def api_deny_renewal(key_id: int, request: Request):
    """Body: {reason?: text shown to the owner}."""
    db = deps.get_db()
    key = await _key_or_404(db, key_id)
    if key.get("renewal_requested_at") is None:
        raise HTTPException(status_code=409, detail="No renewal was requested")
    body = await deps.json_body(request)
    reason = str(body.get("reason") or "").strip()[:500] or None
    await db.deny_api_key_renewal(key_id, reason)
    return await _status_reply(db, key_id)


def _actor_id(request: Request) -> int | None:
    user = getattr(request.state, "dashboard_user", None)
    if user is None or user.auth_source == management_keys.AUTH_SOURCE:
        return None
    return int(user.id)


@router.post("/api/keys/{key_id:int}/pause")
async def api_pause_key(key_id: int, request: Request):
    """Body: {locked?: bool, reason?: text}. *locked* (default: the
    system-wide option) makes it a hold only an admin may lift; a hold
    replaces an owner's pause."""
    db = deps.get_db()
    await _key_or_404(db, key_id)
    body = await deps.json_body(request)
    locked = body.get("locked")
    if locked is None:
        locked = await holds_by_default(db)
    elif not isinstance(locked, bool):
        raise HTTPException(status_code=400, detail="locked must be true or false")
    reason = str(body.get("reason") or "").strip()[:500] or None
    await db.pause_api_key(
        key_id,
        by_user_id=_actor_id(request),
        locked=locked,
        reason=reason,
        when=now_utc(),
    )
    return await _status_reply(db, key_id)


@router.post("/api/keys/{key_id:int}/unpause")
async def api_unpause_key(key_id: int):
    db = deps.get_db()
    await _key_or_404(db, key_id)
    await db.unpause_api_key(key_id)
    return await _status_reply(db, key_id)


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
    strict_pins = strict_pins_flag(body)
    if routing_mode not in ROUTING_MODES:
        raise HTTPException(status_code=400, detail="Invalid routing_mode")
    # Admins may issue keys that never expire.
    expires_at = parse_expiry(body.get("expires_at"))
    access = await parse_access(db, body.get("allowed"))
    plaintext, hashed = await new_api_key()

    key_id = await db.create_api_key(
        key_prefix=key_prefix(plaintext),
        key_hash=hashed,
        routing_mode=routing_mode,
        allow_fallback=allow_fallback,
        session_affinity=session_affinity,
        locality=locality,
        daily_spend_cap_usd=cap,
        label=label,
        strict_pins=strict_pins,
    )
    if access is not UNRESTRICTED:
        await store_access(db, key_id, access)
    if expires_at is not None:
        await db.set_api_key_expiry(key_id, expires_at)
    return JSONResponse(
        {
            "id": key_id,
            "expires_at": expires_at.isoformat() if expires_at else None,
            "api_key": plaintext,
            "label": label,
            "daily_spend_cap_usd": cap,
            "routing_mode": routing_mode,
            "allow_fallback": allow_fallback,
            "session_affinity": session_affinity,
            "locality": locality,
            "strict_pins": strict_pins,
        }
    )


@router.get("/api/keys/{key_id:int}")
async def api_get_key(key_id: int):
    """One key's settings; never the key itself."""
    db = deps.get_db()
    key = await db.get_api_key(key_id)
    if key is None:
        raise HTTPException(status_code=404, detail="API key not found")
    return JSONResponse(
        jsonable_encoder(await with_access(db, with_status([dict(key)])[0]))
    )


@router.patch("/api/keys/{key_id:int}")
async def api_update_key(key_id: int, request: Request):
    """Change any of label, routing_mode, allow_fallback, session_affinity,
    locality, daily_spend_cap_usd, strict_pins and allowed (the models and
    groups the key may use; null for any); fields left out stay as they are.
    Works on keys users issued themselves too."""
    db = deps.get_db()
    body = await deps.json_body(request)
    edits = key_edits(body, ADMIN_KEY_FIELDS)
    access = await parse_access(db, body["allowed"]) if "allowed" in body else None
    if not await db.update_api_key(key_id, edits):
        raise HTTPException(status_code=404, detail="API key not found")
    if "expires_at" in body:
        await db.set_api_key_expiry(key_id, parse_expiry(body["expires_at"]))
    if access is not None:
        await store_access(db, key_id, access)
    key = await db.get_api_key(key_id)
    if key is None:
        raise HTTPException(status_code=404, detail="API key not found")
    return JSONResponse(jsonable_encoder(await with_access(db, key)))


@router.post("/api/keys/{key_id:int}/regenerate")
async def api_regenerate_key(key_id: int):
    """Give the key a new secret, shown once. Its settings, pins, policy and
    usage history stay; the old secret stops working at once."""
    plaintext, hashed = await new_api_key()
    db = deps.get_db()
    if not await db.regenerate_api_key(key_id, key_prefix(plaintext), hashed):
        raise HTTPException(status_code=404, detail="API key not found")
    return JSONResponse(
        {"id": key_id, "api_key": plaintext, "key_prefix": key_prefix(plaintext)}
    )


# Deprecated: PATCH /api/keys/{key_id} changes these and everything else.
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
