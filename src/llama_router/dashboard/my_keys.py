"""API keys dashboard users issue to themselves for model access.

Any signed-in user may issue, list and revoke their own keys on ``/my-keys``.
They choose how the key is routed (priority, fallback, session affinity);
which providers it may use and its daily spend cap come from defaults an
administrator sets, so a user can't give themselves cloud spending an admin
didn't allow. Admins see these keys, with their owner, on the API Keys tab
and can change them there like any other key. Deleting a user deletes the
keys they issued.
"""

from __future__ import annotations

import asyncio
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from fastapi.encoders import jsonable_encoder
from fastapi.responses import HTMLResponse, JSONResponse, Response

from ..auth import generate_api_key, key_hash, key_prefix
from ..config import settings
from ..database import Database
from ..router import ROUTING_MODES
from ..routing_table import DEFAULT_MODE
from . import deps, management_keys
from .routes import (
    _key_locality,
    _tristate,
    _usd,
    _utc_day_start,
    merge_dash_template_ctx,
    templates,
)

router = APIRouter()

LOCALITY_SETTING = "self_key_locality"
CAP_SETTING = "self_key_daily_spend_cap_usd"
# Until an admin says otherwise, self-issued keys can't spend money.
DEFAULT_LOCALITY = "local"


async def self_service_defaults(db: Database) -> dict[str, Any]:
    """The locality (None: any) and daily spend cap self-issued keys get."""
    raw_locality = await db.get_app_setting(LOCALITY_SETTING)
    locality = _key_locality(DEFAULT_LOCALITY if raw_locality is None else raw_locality)
    raw_cap = await db.get_app_setting(CAP_SETTING)
    cap = float(raw_cap) if raw_cap else None
    return {"locality": locality, "daily_spend_cap_usd": cap}


def _owner_id(request: Request) -> int:
    """The signed-in user's id; keys belong to a person, not a management
    key or an unconfigured dashboard."""
    user = getattr(request.state, "dashboard_user", None)
    if user is None or user.auth_source == management_keys.AUTH_SOURCE:
        raise HTTPException(
            status_code=403, detail="Sign in to issue yourself an API key"
        )
    return int(user.id)


async def _body(request: Request) -> dict[str, Any]:
    try:
        body = await request.json()
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="body must be JSON") from exc
    if not isinstance(body, dict):
        raise HTTPException(status_code=400, detail="body must be a JSON object")
    return body


@router.get("/my-keys", response_class=HTMLResponse)
async def my_keys_page(request: Request) -> Response:
    return templates.TemplateResponse(
        request, "my_keys.html", merge_dash_template_ctx(request, {})
    )


@router.get("/api/my-keys")
async def list_my_keys(request: Request) -> JSONResponse:
    """The user's own keys (never the keys themselves) and the limits new
    ones get."""
    owner = _owner_id(request)
    db = deps.get_db()
    keys = await db.list_api_keys(owner_user_id=owner)
    spent = await db.spend_by_key_since(_utc_day_start())
    for k in keys:
        k["spent_today_usd"] = spent.get(int(k["id"]), 0.0)
    return JSONResponse(
        jsonable_encoder({"keys": keys, "defaults": await self_service_defaults(db)})
    )


@router.post("/api/my-keys")
async def issue_my_key(request: Request) -> JSONResponse:
    owner = _owner_id(request)
    body = await _body(request)
    routing_mode = str(body.get("routing_mode") or DEFAULT_MODE).strip().lower()
    if routing_mode not in ROUTING_MODES:
        raise HTTPException(status_code=400, detail="Invalid routing_mode")
    allow_fallback = bool(body.get("allow_fallback", True))
    session_affinity = _tristate(body.get("session_affinity"))
    db = deps.get_db()
    limit = settings.max_api_keys_per_user
    if limit > 0 and len(await db.list_api_keys(owner_user_id=owner)) >= limit:
        raise HTTPException(
            status_code=409,
            detail=f"You already have {limit} API keys; revoke one to issue another",
        )
    defaults = await self_service_defaults(db)
    # PBKDF2 is slow; keep it off the event loop.
    plaintext = generate_api_key()
    hashed = await asyncio.to_thread(key_hash, plaintext)
    key_id = await db.create_api_key(
        key_prefix=key_prefix(plaintext),
        key_hash=hashed,
        routing_mode=routing_mode,
        allow_fallback=allow_fallback,
        session_affinity=session_affinity,
        locality=defaults["locality"],
        daily_spend_cap_usd=defaults["daily_spend_cap_usd"],
        owner_user_id=owner,
    )
    return JSONResponse(
        {
            "id": key_id,
            "api_key": plaintext,
            "routing_mode": routing_mode,
            "allow_fallback": allow_fallback,
            "session_affinity": session_affinity,
            **defaults,
        }
    )


@router.delete("/api/my-keys/{key_id}")
async def revoke_my_key(request: Request, key_id: int) -> JSONResponse:
    owner = _owner_id(request)
    if not await deps.get_db().delete_owned_api_key(key_id, owner):
        raise HTTPException(status_code=404, detail="API key not found")
    return JSONResponse({"ok": True})


# Under /api/keys, so only admins (and management keys with the keys
# privilege) reach these.
@router.get("/api/keys/self-service")
async def get_self_service_defaults() -> JSONResponse:
    return JSONResponse(await self_service_defaults(deps.get_db()))


@router.put("/api/keys/self-service")
async def set_self_service_defaults(request: Request) -> JSONResponse:
    """Body: {locality: local|cloud|any, daily_spend_cap_usd: number|null}.
    Applies to keys issued from now on, not to existing ones."""
    body = await _body(request)
    locality = _key_locality(body.get("locality"))
    cap = _usd(body.get("daily_spend_cap_usd"), "daily spend cap")
    db = deps.get_db()
    await db.set_app_setting(LOCALITY_SETTING, locality or "any")
    await db.set_app_setting(CAP_SETTING, "" if cap is None else str(cap))
    return JSONResponse(await self_service_defaults(db))
