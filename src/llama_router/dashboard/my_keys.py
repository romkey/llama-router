"""API keys dashboard users issue to themselves for model access.

Any signed-in user may issue, list, edit, regenerate and revoke their own
keys on ``/my-keys``. They choose how the key is routed (priority, fallback,
session affinity) and can change that later; which providers it may use and
its daily spend cap come from defaults an administrator sets, so a user
can't give themselves cloud spending an admin didn't allow. Admins see
these keys, with their owner, on the API Keys tab and can change them there
like any other key. Deleting a user deletes the keys they issued.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Request
from fastapi.encoders import jsonable_encoder
from fastapi.responses import HTMLResponse, JSONResponse, Response

from ..auth import key_prefix, new_api_key
from ..config import settings
from ..database import Database
from ..router import ROUTING_MODES
from ..routing_table import DEFAULT_MODE
from . import deps, management_keys
from .form_values import (
    OWNER_KEY_FIELDS,
    _key_label,
    _key_locality,
    _tristate,
    _usd,
    _utc_day_start,
    key_edits,
)
from .pages import dash_page_ctx, templates

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


@router.get("/my-keys", response_class=HTMLResponse)
async def my_keys_page(request: Request) -> Response:
    return templates.TemplateResponse(
        request, "my_keys.html", await dash_page_ctx(request, {})
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
    body = await deps.json_body(request)
    routing_mode = str(body.get("routing_mode") or DEFAULT_MODE).strip().lower()
    if routing_mode not in ROUTING_MODES:
        raise HTTPException(status_code=400, detail="Invalid routing_mode")
    allow_fallback = bool(body.get("allow_fallback", True))
    session_affinity = _tristate(body.get("session_affinity"))
    label = _key_label(body.get("label"))
    db = deps.get_db()
    limit = settings.max_api_keys_per_user
    if limit > 0 and len(await db.list_api_keys(owner_user_id=owner)) >= limit:
        raise HTTPException(
            status_code=409,
            detail=f"You already have {limit} API keys; revoke one to issue another",
        )
    defaults = await self_service_defaults(db)
    plaintext, hashed = await new_api_key()
    key_id = await db.create_api_key(
        key_prefix=key_prefix(plaintext),
        key_hash=hashed,
        routing_mode=routing_mode,
        allow_fallback=allow_fallback,
        session_affinity=session_affinity,
        locality=defaults["locality"],
        daily_spend_cap_usd=defaults["daily_spend_cap_usd"],
        owner_user_id=owner,
        label=label,
    )
    return JSONResponse(
        {
            "id": key_id,
            "api_key": plaintext,
            "label": label,
            "routing_mode": routing_mode,
            "allow_fallback": allow_fallback,
            "session_affinity": session_affinity,
            **defaults,
        }
    )


async def _own_key(owner: int, key_id: int) -> dict[str, Any]:
    key = await deps.get_db().get_api_key(key_id, owner_user_id=owner)
    if key is None:
        raise HTTPException(status_code=404, detail="API key not found")
    return key


@router.get("/api/my-keys/{key_id}")
async def get_my_key(request: Request, key_id: int) -> JSONResponse:
    return JSONResponse(jsonable_encoder(await _own_key(_owner_id(request), key_id)))


@router.patch("/api/my-keys/{key_id}")
async def edit_my_key(request: Request, key_id: int) -> JSONResponse:
    """Change the key's label, routing_mode, allow_fallback or
    session_affinity. Locality and the spend cap are an administrator's."""
    owner = _owner_id(request)
    edits = key_edits(await deps.json_body(request), OWNER_KEY_FIELDS)
    db = deps.get_db()
    if not await db.update_api_key(key_id, edits, owner_user_id=owner):
        raise HTTPException(status_code=404, detail="API key not found")
    return JSONResponse(jsonable_encoder(await _own_key(owner, key_id)))


@router.post("/api/my-keys/{key_id}/regenerate")
async def regenerate_my_key(request: Request, key_id: int) -> JSONResponse:
    """A new secret for the key, shown once; the old one stops working."""
    owner = _owner_id(request)
    plaintext, hashed = await new_api_key()
    db = deps.get_db()
    prefix = key_prefix(plaintext)
    if not await db.regenerate_api_key(key_id, prefix, hashed, owner_user_id=owner):
        raise HTTPException(status_code=404, detail="API key not found")
    return JSONResponse({"id": key_id, "api_key": plaintext, "key_prefix": prefix})


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
    body = await deps.json_body(request)
    locality = _key_locality(body.get("locality"))
    cap = _usd(body.get("daily_spend_cap_usd"), "daily spend cap")
    db = deps.get_db()
    await db.set_app_setting(LOCALITY_SETTING, locality or "any")
    await db.set_app_setting(CAP_SETTING, "" if cap is None else str(cap))
    return JSONResponse(await self_service_defaults(db))
