"""API keys dashboard users issue to themselves for model access.

Any signed-in user may issue, list, edit, regenerate and revoke their own
keys on ``/my-keys``. They choose how the key is routed (priority, fallback,
session affinity, strict pins, which provider serves a model) and can change
that later; which providers it may use and
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

from ..auth import key_prefix, model_access, new_api_key
from ..config import settings
from ..database import Database
from ..key_lifecycle import (
    now_utc,
    parse_expiry,
    renewal_mode,
    renewed_expiry,
    self_issued_expiry,
    with_status,
)
from ..models import Provider
from ..quotas import limits_for_user
from ..router import ROUTING_MODES
from ..routing_table import DEFAULT_MODE
from ..spend import next_day_start, tracker_for
from . import deps, management_keys
from .form_values import (
    OWNER_KEY_FIELDS,
    _key_label,
    _key_locality,
    _tristate,
    _usd,
    _utc_day_start,
    key_edits,
    strict_pins_flag,
)
from .key_access import (
    UNRESTRICTED,
    allowable,
    parse_access,
    store_access,
    with_access,
)
from .pages import dash_page_ctx, templates
from .routes.api_keys import validate_pin_model_name

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


def _owner_or_admin(request: Request) -> None:
    """A signed-in user, or an unconfigured dashboard's admin; not a
    management key."""
    user = getattr(request.state, "dashboard_user", None)
    if user is not None and user.auth_source == management_keys.AUTH_SOURCE:
        raise HTTPException(status_code=403, detail="Not available to management keys")


def _owner_id(request: Request) -> int:
    """The signed-in user's id; keys belong to a person, not a management
    key or an unconfigured dashboard."""
    user = getattr(request.state, "dashboard_user", None)
    if user is None or user.auth_source == management_keys.AUTH_SOURCE:
        raise HTTPException(
            status_code=403, detail="Sign in to issue yourself an API key"
        )
    return int(user.id)


_QUOTA_KEY_FIELDS = ("daily_token_limit", "requests_per_minute")


async def _only_tighter(
    db: Database, key_id: int, owner: int, edits: dict[str, Any]
) -> None:
    """An owner may lower a key's token or rate limit, or set one, but not
    raise or clear one (an admin may have set it)."""
    asked = {f: edits[f] for f in _QUOTA_KEY_FIELDS if f in edits}
    if not asked:
        return
    key = await db.get_api_key(key_id, owner_user_id=owner)
    if key is None:
        return  # the update reports 404
    for field, new in asked.items():
        current = key.get(field)
        if current is not None and (new is None or new > current):
            raise HTTPException(
                status_code=403,
                detail=f"You can lower {field} on your key, not raise or clear it",
            )


@router.get("/my-keys", response_class=HTMLResponse)
async def my_keys_page(request: Request) -> Response:
    return templates.TemplateResponse(
        request,
        "my_keys.html",
        await dash_page_ctx(
            request, {"max_key_lifetime_days": settings.max_key_lifetime_days}
        ),
    )


@router.get("/api/my-keys")
async def list_my_keys(request: Request) -> JSONResponse:
    """The user's own keys (never the keys themselves) and the limits new
    ones get."""
    owner = _owner_id(request)
    db = deps.get_db()
    keys = with_status(await db.list_api_keys(owner_user_id=owner))
    today = _utc_day_start()
    used = await db.usage_by_since(today, "api_key_id")
    tracker = tracker_for(db)
    for k in keys:
        cost, tokens = used.get(int(k["id"]), (0.0, 0))
        k["spent_today_usd"] = cost
        k["tokens_today"] = tokens
        k["requests_this_minute"] = tracker.requests_this_minute(key_id=int(k["id"]))
    cost, tokens = await db.usage_since(today, user_id=owner)
    limits = await limits_for_user(db, owner)
    return JSONResponse(
        jsonable_encoder(
            {
                "keys": keys,
                "defaults": await self_service_defaults(db),
                "lifecycle": {
                    "renewal_mode": await renewal_mode(db),
                    "max_key_lifetime_days": settings.max_key_lifetime_days,
                },
                "quota": {
                    "limits": limits.to_dict(),
                    "usage": {
                        "tokens": tokens,
                        "cost_usd": cost,
                        "requests_this_minute": tracker.requests_this_minute(
                            user_id=owner
                        ),
                    },
                    "resets_at": next_day_start().isoformat(),
                },
            }
        )
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
    strict_pins = strict_pins_flag(body)
    db = deps.get_db()
    limit = settings.max_api_keys_per_user
    if limit > 0 and len(await db.list_api_keys(owner_user_id=owner)) >= limit:
        raise HTTPException(
            status_code=409,
            detail=f"You already have {limit} API keys; revoke one to issue another",
        )
    defaults = await self_service_defaults(db)
    expires_at = self_issued_expiry(parse_expiry(body.get("expires_at")))
    # Limiting a key to some models only narrows it, so users may.
    access = await parse_access(db, body.get("allowed"))
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
        strict_pins=strict_pins,
    )
    if access is not UNRESTRICTED:
        await store_access(db, key_id, access)
    if expires_at is not None:
        await db.set_api_key_expiry(key_id, expires_at, owner_user_id=owner)
    return JSONResponse(
        {
            "id": key_id,
            "api_key": plaintext,
            "expires_at": expires_at.isoformat() if expires_at else None,
            "label": label,
            "routing_mode": routing_mode,
            "allow_fallback": allow_fallback,
            "session_affinity": session_affinity,
            "strict_pins": strict_pins,
            **defaults,
        }
    )


async def _own_key(owner: int, key_id: int) -> dict[str, Any]:
    db = deps.get_db()
    key = await db.get_api_key(key_id, owner_user_id=owner)
    if key is None:
        raise HTTPException(status_code=404, detail="API key not found")
    return await with_access(db, with_status([dict(key)])[0])


@router.get("/api/my-keys/allowable")
async def allowable_models(request: Request) -> JSONResponse:
    """What a key can be limited to: model groups and known model names,
    and for each model or group the providers serving it (id, name and
    locality only), for choosing a pin. For the key forms on My API keys
    and the API Keys tab."""
    _owner_or_admin(request)
    db = deps.get_db()
    served = await db.get_all_provider_models()
    out = await allowable(db, [m.name for m in served])
    providers = {p.id: p for p in await db.list_providers() if p.id is not None}
    serving: dict[str, set[int]] = {}
    for m in served:
        if m.provider_id in providers:
            serving.setdefault(m.name, set()).add(m.provider_id)
    for g in out["groups"]:
        ids = set().union(*(serving.get(n, set()) for n in g["members"]))
        if ids:
            serving[g["name"]] = ids
    out["providers"] = {
        name: [_provider_summary(providers[i]) for i in _by_name(ids, providers)]
        for name, ids in sorted(serving.items())
    }
    return JSONResponse(jsonable_encoder(out))


def _provider_summary(p: Provider) -> dict[str, Any]:
    """What a key's owner may see of a provider: never addresses or keys."""
    return {"id": p.id, "name": p.name, "locality": p.locality}


def _by_name(ids: set[int], providers: dict[int, Provider]) -> list[int]:
    return sorted(ids, key=lambda i: (providers[i].name, i))


@router.get("/api/my-keys/{key_id:int}")
async def get_my_key(request: Request, key_id: int) -> JSONResponse:
    return JSONResponse(jsonable_encoder(await _own_key(_owner_id(request), key_id)))


@router.patch("/api/my-keys/{key_id:int}")
async def edit_my_key(request: Request, key_id: int) -> JSONResponse:
    """Change the key's label, routing_mode, allow_fallback,
    session_affinity, strict_pins or allowed models. Locality and the spend cap are an
    administrator's."""
    owner = _owner_id(request)
    body = await deps.json_body(request)
    edits = key_edits(body, OWNER_KEY_FIELDS)
    db = deps.get_db()
    await _only_tighter(db, key_id, owner, edits)
    access = await parse_access(db, body["allowed"]) if "allowed" in body else None
    expiry = (
        self_issued_expiry(parse_expiry(body["expires_at"]))
        if "expires_at" in body
        else None
    )
    if not await db.update_api_key(key_id, edits, owner_user_id=owner):
        raise HTTPException(status_code=404, detail="API key not found")
    if "expires_at" in body:
        await db.set_api_key_expiry(key_id, expiry, owner_user_id=owner)
    if access is not None:
        await store_access(db, key_id, access)
    return JSONResponse(jsonable_encoder(await _own_key(owner, key_id)))


@router.post("/api/my-keys/{key_id:int}/renew")
async def renew_my_key(request: Request, key_id: int) -> JSONResponse:
    """Renew the key (a new expiry; the secret stays), or ask an admin to,
    as the system-wide renewal option says. Allowed before or after expiry;
    a pending request does not revive an expired key."""
    owner = _owner_id(request)
    db = deps.get_db()
    await _own_key(owner, key_id)
    mode = await renewal_mode(db)
    if mode == "off":
        raise HTTPException(
            status_code=403, detail="Only an administrator can renew API keys"
        )
    if mode == "approval":
        await db.request_api_key_renewal(key_id, owner, now_utc())
    else:
        await db.set_api_key_expiry(key_id, renewed_expiry(True), owner_user_id=owner)
    return JSONResponse(jsonable_encoder(await _own_key(owner, key_id)))


@router.post("/api/my-keys/{key_id:int}/pause")
async def pause_my_key(request: Request, key_id: int) -> JSONResponse:
    """Pause the key (403 on requests until unpaused). An admin's hold
    can't be replaced."""
    owner = _owner_id(request)
    key = await _own_key(owner, key_id)
    if key.get("pause_locked"):
        raise HTTPException(
            status_code=403, detail="An administrator holds this key paused"
        )
    if not await deps.get_db().owner_set_pause(
        key_id, owner, paused=True, when=now_utc()
    ):
        raise HTTPException(
            status_code=403, detail="An administrator holds this key paused"
        )
    return JSONResponse(jsonable_encoder(await _own_key(owner, key_id)))


@router.post("/api/my-keys/{key_id:int}/unpause")
async def unpause_my_key(request: Request, key_id: int) -> JSONResponse:
    """Lift your own pause; an admin's hold only an admin can lift."""
    owner = _owner_id(request)
    key = await _own_key(owner, key_id)
    if key.get("pause_locked"):
        raise HTTPException(
            status_code=403,
            detail="An administrator holds this key paused; ask them to lift it",
        )
    if not await deps.get_db().owner_set_pause(
        key_id, owner, paused=False, when=now_utc()
    ):
        raise HTTPException(
            status_code=403,
            detail="An administrator holds this key paused; ask them to lift it",
        )
    return JSONResponse(jsonable_encoder(await _own_key(owner, key_id)))


@router.get("/api/my-keys/{key_id:int}/pins")
async def list_my_key_pins(request: Request, key_id: int) -> JSONResponse:
    """The key's pins, including any an administrator set."""
    key = await _own_key(_owner_id(request), key_id)
    return JSONResponse(
        {"pins": await deps.get_db().list_api_key_model_pins(key["id"])}
    )


@router.post("/api/my-keys/{key_id:int}/pins")
async def set_my_key_pin(request: Request, key_id: int) -> JSONResponse:
    """Body: {model_name, provider_id}. Pins the model (or model group) to
    the provider for this key. Only a model the key may use and a provider
    inside the key's locality can be pinned; the key's locality and model
    limits are an administrator's, so a pin can't widen them."""
    key = await _own_key(_owner_id(request), key_id)
    body = await deps.json_body(request)
    model_name = validate_pin_model_name(body.get("model_name"))
    provider_id = _provider_id(body.get("provider_id"))
    db = deps.get_db()
    provider = await db.get_provider(provider_id)
    if provider is None:
        raise HTTPException(status_code=400, detail="Provider not found")
    locality = key.get("locality")
    if locality is not None and provider.locality != locality:
        raise HTTPException(
            status_code=400,
            detail=f"Provider {provider.name} is {provider.locality}; "
            f"this key may only use {locality} providers",
        )
    if not await _key_may_use(db, key_id, model_name):
        raise HTTPException(
            status_code=400, detail=f"This API key may not use model '{model_name}'"
        )
    pins = await db.list_api_key_model_pins(key_id)
    limit = settings.max_pins_per_key
    if (
        limit > 0
        and len(pins) >= limit
        and all(p["model_name"] != model_name for p in pins)
    ):
        raise HTTPException(
            status_code=409,
            detail=f"This key already has {limit} pins; remove one to add another",
        )
    await db.set_api_key_model_pin(key_id, model_name, provider_id)
    return JSONResponse({"ok": True})


@router.delete("/api/my-keys/{key_id:int}/pins/{model_name:path}")
async def delete_my_key_pin(
    request: Request, key_id: int, model_name: str
) -> JSONResponse:
    """Remove a pin, including one an administrator set on the key."""
    await _own_key(_owner_id(request), key_id)
    await deps.get_db().remove_api_key_model_pin(key_id, model_name)
    return JSONResponse({"ok": True})


def _provider_id(raw: Any) -> int:
    if raw is None or isinstance(raw, bool):
        raise HTTPException(status_code=400, detail="provider_id is required")
    try:
        return int(raw)
    except TypeError, ValueError:
        raise HTTPException(
            status_code=400, detail="provider_id must be an integer"
        ) from None


async def _key_may_use(db: Database, key_id: int, model: str) -> bool:
    access = await model_access(db, key_id)
    if access is None:
        return True
    rt = deps.get_pm().router
    if rt is not None:
        return (await rt.current_table()).allows(access, model)
    return model in access.models or model in access.groups


@router.post("/api/my-keys/{key_id:int}/regenerate")
async def regenerate_my_key(request: Request, key_id: int) -> JSONResponse:
    """A new secret for the key, shown once; the old one stops working."""
    owner = _owner_id(request)
    plaintext, hashed = await new_api_key()
    db = deps.get_db()
    prefix = key_prefix(plaintext)
    if not await db.regenerate_api_key(key_id, prefix, hashed, owner_user_id=owner):
        raise HTTPException(status_code=404, detail="API key not found")
    return JSONResponse({"id": key_id, "api_key": plaintext, "key_prefix": prefix})


@router.delete("/api/my-keys/{key_id:int}")
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
