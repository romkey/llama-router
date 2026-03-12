from __future__ import annotations

import hashlib
from secrets import token_urlsafe

from fastapi import HTTPException, Request

from .database import Database
from .router import RoutingPreferences


def generate_api_key() -> str:
    return "lrk_" + token_urlsafe(32)


def key_prefix(key: str) -> str:
    if len(key) <= 10:
        return key
    return key[:10] + "…"


def key_hash(key: str) -> str:
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


async def routing_preferences_from_request(
    db: Database, request: Request
) -> RoutingPreferences | None:
    allow_unauthenticated = await db.get_allow_unauthenticated()
    raw_key = request.headers.get("API-KEY")
    key = (raw_key or "").strip()

    if not key:
        if allow_unauthenticated:
            return None
        raise HTTPException(status_code=401, detail="Missing API-KEY header")

    record = await db.lookup_api_key(key_hash(key))
    if not record:
        if allow_unauthenticated:
            return None
        raise HTTPException(status_code=401, detail="Invalid API-KEY")

    mode = str(record["routing_mode"]).lower()
    if mode not in {"latency", "throughput", "chaos"}:
        mode = "latency"
    pins = await db.list_api_key_model_pins(int(record["id"]))
    pinned_providers = {str(p["model_name"]): int(p["provider_id"]) for p in pins}
    return RoutingPreferences(
        mode=mode,
        allow_fallback=bool(record["allow_fallback"]),
        pinned_providers=pinned_providers,
    )
