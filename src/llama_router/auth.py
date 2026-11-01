from __future__ import annotations

import asyncio
import hashlib
import logging
import time
import weakref
from dataclasses import replace
from secrets import token_urlsafe
from typing import Any

from fastapi import HTTPException, Request

from .background import spawn
from .config import settings
from .database import AUTH_CHANGES, CHANGE_PROVIDERS, Database
from .decision_policy import DecisionPolicy, parse_policy
from .provider_secrets import server_secret
from .router import ROUTING_MODES, RoutingPreferences
from .routing_table import DEFAULT_MODE
from .spend import tracker_for
from .upstream_url import LOCALITIES
from .worker_link.protocol import HUB_HEADER, HUB_HEADER_VALUE, PROVIDER_HEADER

logger = logging.getLogger(__name__)

_PBKDF2_ROUNDS = 390_000
# Resolved keys are reused for this long unless an edit invalidates them.
_PREFS_TTL_S = 300.0
# last_used_at is written at most this often per key.
_TOUCH_INTERVAL_S = 60.0
_MAX_CACHED_KEYS = 1024


def generate_api_key() -> str:
    return "lrk_" + token_urlsafe(32)


def key_prefix(key: str) -> str:
    if len(key) <= 10:
        return key
    return key[:10] + "…"


def key_hash_legacy(key: str) -> str:
    """Legacy storage format (plain SHA-256); used for backward-compatible lookup."""
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


# The fixed pepper older releases used without a session secret. Keys
# hashed with it still match and are re-hashed with the current pepper.
_LEGACY_DEV_PEPPER = b"llama-router-api-key-pepper-dev"


def _pbkdf2(key: str, pepper: bytes) -> str:
    return hashlib.pbkdf2_hmac(
        "sha256",
        key.encode("utf-8"),
        pepper,
        _PBKDF2_ROUNDS,
    ).hex()


def key_hash(key: str) -> str:
    """Fingerprint for API key storage using PBKDF2-HMAC-SHA256.

    The pepper is ``LLAMA_ROUTER_SESSION_SECRET``, or the generated secret
    file's value when that is not set.
    """
    return _pbkdf2(key, server_secret())


def key_hash_legacy_pepper(key: str) -> str:
    """PBKDF2 fingerprint with the old fixed development pepper."""
    return _pbkdf2(key, _LEGACY_DEV_PEPPER)


class _AuthCache:
    """Resolved API keys and the unauthenticated-access flag for one database.

    Keys are cached by their SHA-256, never in plain text. Entries expire after
    a few minutes and are dropped at once when keys, pins, providers or the
    auth setting change, so dashboard edits apply to the next request.
    """

    def __init__(self, db: Database) -> None:
        self.allow_unauthenticated: bool | None = None
        self.prefs: dict[str, tuple[float, int, RoutingPreferences]] = {}
        self.touched: dict[int, float] = {}
        # Bumped on every clear. A lookup that started before a clear must
        # not store its (possibly stale) answer afterwards.
        self.generation = 0
        db.add_change_listener(self._on_change)

    def _on_change(self, reason: str) -> None:
        if reason in AUTH_CHANGES or reason == CHANGE_PROVIDERS:
            self.clear()

    def clear(self) -> None:
        self.generation += 1
        self.allow_unauthenticated = None
        self.prefs.clear()


_caches: weakref.WeakKeyDictionary[Database, _AuthCache] = weakref.WeakKeyDictionary()


def _cache_for(db: Database) -> _AuthCache:
    cache = _caches.get(db)
    if cache is None:
        cache = _caches[db] = _AuthCache(db)
    return cache


def invalidate_auth_cache(db: Database) -> None:
    cache = _caches.get(db)
    if cache is not None:
        cache.clear()


async def _allow_unauthenticated(db: Database, cache: _AuthCache) -> bool:
    cached = cache.allow_unauthenticated
    if cached is not None:
        return cached
    generation = cache.generation
    value = await db.get_allow_unauthenticated()
    if cache.generation == generation:
        cache.allow_unauthenticated = value
    return value


async def _resolve_key(
    db: Database, key: str, legacy: str
) -> tuple[RoutingPreferences, int] | None:
    """Look the key up in the database. Slow path: hashing and queries.

    *legacy* is ``key_hash_legacy(key)``, already computed by the caller.
    """
    record = await db.lookup_api_key(legacy)
    if not record:
        # PBKDF2 takes tens of milliseconds; keep it off the event loop.
        modern = await asyncio.to_thread(key_hash, key)
        if modern != legacy:
            record = await db.lookup_api_key(modern)
        if not record and not settings.session_secret.strip():
            # Created before the generated secret replaced the fixed pepper.
            old = await asyncio.to_thread(key_hash_legacy_pepper, key)
            record = await db.lookup_api_key(old)
            if record:
                await db.set_api_key_hash(int(record["id"]), modern)
    if not record:
        return None
    mode = str(record["routing_mode"]).lower()
    if mode not in ROUTING_MODES:
        mode = DEFAULT_MODE
    pins = await db.list_api_key_model_pins(int(record["id"]))
    prefs = RoutingPreferences(
        mode=mode,
        allow_fallback=bool(record["allow_fallback"]),
        pinned_providers={str(p["model_name"]): int(p["provider_id"]) for p in pins},
        session_affinity=record.get("session_affinity"),
        key_id=int(record["id"]),
        locality=_locality(record.get("locality")),
        daily_spend_cap_usd=record.get("daily_spend_cap_usd"),
        decision_policy=await _decision_policy(db, int(record["id"])),
    )
    return prefs, int(record["id"])


async def _decision_policy(db: Database, key_id: int) -> DecisionPolicy | None:
    raw = await db.get_api_key_decision_policy(key_id)
    if raw is None:
        return None
    try:
        return parse_policy(raw)
    except ValueError as exc:
        # Written through the dashboard, which validates; this is a policy
        # from an older version or edited by hand. Route as if it had none.
        logger.warning(
            "Ignoring invalid decision policy of API key %s: %s", key_id, exc
        )
        return None


LOCALITY_HEADER = "X-Llama-Router-Locality"


def from_hub(request: Request) -> bool:
    """The request was relayed by the worker link from this router's hub."""
    value = (request.headers.get(HUB_HEADER) or "").strip().lower()
    return value == HUB_HEADER_VALUE


def hub_provider(request: Request) -> int | None:
    """The one provider a job from this router's hub is for, when the worker
    shares providers individually; None for any other request."""
    if not from_hub(request):
        return None
    raw = (request.headers.get(PROVIDER_HEADER) or "").strip()
    return int(raw) if raw.isdigit() else None


def _locality(value: Any) -> str | None:
    v = str(value or "").strip().lower()
    return v if v in LOCALITIES else None


def _narrow_locality(
    request: Request, prefs: RoutingPreferences | None
) -> RoutingPreferences | None:
    """Apply the request's locality header, which may narrow the key's
    setting (from either to local or cloud) but never widen or flip it."""
    wanted = _locality(request.headers.get(LOCALITY_HEADER))
    if wanted is None:
        return prefs
    if prefs is None:
        return RoutingPreferences(locality=wanted)
    if prefs.locality is not None:
        return prefs
    return replace(prefs, locality=wanted)


async def routing_preferences_from_request(
    db: Database, request: Request
) -> RoutingPreferences | None:
    """The request's routing preferences: its API key's, narrowed by the
    X-Llama-Router-Locality header and the key's daily spend cap.

    Also records the key on ``request.state.api_key_id`` for the request log.
    """
    prefs = _narrow_locality(request, await _key_preferences(db, request))
    if from_hub(request):
        prefs = replace(
            prefs or RoutingPreferences(),
            hub_only=True,
            hub_provider=hub_provider(request),
        )
    if prefs is not None and prefs.key_id is not None:
        request.state.api_key_id = prefs.key_id
        prefs = await _apply_spend_cap(db, prefs)
    return prefs


async def _apply_spend_cap(
    db: Database, prefs: RoutingPreferences
) -> RoutingPreferences:
    """A key over its daily cap is limited to local providers, which cost
    nothing; a cloud-only key over its cap is refused until the next UTC day."""
    cap = prefs.daily_spend_cap_usd
    if cap is None or prefs.key_id is None:
        return prefs
    spent = await tracker_for(db).spent_today(prefs.key_id)
    if spent < cap:
        return prefs
    if prefs.locality == "cloud":
        raise HTTPException(
            status_code=429,
            detail=f"daily spend cap of ${cap:.2f} reached for this API key",
        )
    return replace(prefs, locality="local", spend_capped=True)


def client_api_key(request: Request) -> str:
    """The client's key from ``API-KEY``, else ``Authorization: Bearer``.

    OpenAI-compatible clients such as Open WebUI and the OpenAI SDKs can only
    send Bearer. Neither header is forwarded upstream.
    """
    key = (request.headers.get("API-KEY") or "").strip()
    if key:
        return key
    raw = (request.headers.get("authorization") or "").strip()
    scheme, _, token = raw.partition(" ")
    return token.strip() if scheme.lower() == "bearer" else ""


async def _key_preferences(db: Database, request: Request) -> RoutingPreferences | None:
    cache = _cache_for(db)
    key = client_api_key(request)

    if not key:
        if await _allow_unauthenticated(db, cache):
            return None
        raise HTTPException(
            status_code=401,
            detail="Missing API key: send API-KEY or Authorization: Bearer",
        )

    # The legacy digest is needed for the lookup anyway; it doubles as the
    # in-memory cache key so the plain key is never kept.
    fingerprint = key_hash_legacy(key)
    now = time.monotonic()
    hit = cache.prefs.get(fingerprint)
    if hit is not None and now - hit[0] < _PREFS_TTL_S:
        _, key_id, prefs = hit
        if now - cache.touched.get(key_id, 0.0) >= _TOUCH_INTERVAL_S:
            cache.touched[key_id] = now
            spawn(db.touch_api_key(key_id), name="touch-api-key")
        return prefs

    generation = cache.generation
    resolved = await _resolve_key(db, key, fingerprint)
    if resolved is None:
        if await _allow_unauthenticated(db, cache):
            return None
        raise HTTPException(status_code=401, detail="Invalid API-KEY")
    prefs, key_id = resolved
    # lookup_api_key just wrote last_used_at.
    cache.touched[key_id] = now
    if cache.generation != generation:
        # A key, pin or policy changed while we looked it up; answer this
        # request but do not cache what may already be stale.
        return prefs
    if len(cache.prefs) >= _MAX_CACHED_KEYS:
        cache.prefs.clear()
    cache.prefs[fingerprint] = (now, key_id, prefs)
    return prefs
