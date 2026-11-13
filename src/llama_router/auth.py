from __future__ import annotations

import asyncio
import logging
import re
import time
import weakref
from collections.abc import Callable
from dataclasses import replace
from secrets import token_urlsafe
from typing import Any

from fastapi import HTTPException, Request

from .background import spawn
from .database import AUTH_CHANGES, CHANGE_GROUPS, CHANGE_PROVIDERS, Database
from .decision_policy import DecisionPolicy, parse_policy
from .provider_secrets import (
    key_hash,
    key_hash_legacy,
    key_hash_pbkdf2,
    key_hash_previous,
    pbkdf2_peppers,
)
from .quotas import (
    check_rate,
    check_tokens_left,
    limits_for_user,
    reserve_tokens,
    token_estimate,
)
from .router import (
    ROUTING_MODES,
    PinUnavailableError,
    Router,
    RouteResult,
    RoutingPreferences,
)
from .routing_table import DEFAULT_MODE, ModelAccess
from .spend import RESERVATION_STATE, cap_reached, tracker_for
from .upstream_url import LOCALITIES
from .worker_link.relay import SCOPE_KEY as _RELAY_SCOPE_KEY
from .worker_link.relay import is_own_relay

logger = logging.getLogger(__name__)

# Resolved keys are reused for this long unless an edit invalidates them.
_PREFS_TTL_S = 300.0
# last_used_at is written at most this often per key.
_TOUCH_INTERVAL_S = 60.0
_MAX_CACHED_KEYS = 1024
# Keys the database did not know are refused from memory for this long, so
# a client retrying a wrong key does not cost a PBKDF2 hash per request
# (older fingerprints are still checked for keys the fast forms miss).
_INVALID_TTL_S = 30.0
_MAX_INVALID_KEYS = 10_000
# At most this many PBKDF2 hashes run at once (per event loop).
_MAX_CONCURRENT_HASHES = 4

# What generate_api_key returns: "lrk_" and 32 random bytes in URL-safe
# base64. Anything else cannot be a key, so it is refused without hashing.
_KEY_FORMAT = re.compile(r"lrk_[A-Za-z0-9_-]{43}")


def generate_api_key() -> str:
    return "lrk_" + token_urlsafe(32)


async def new_api_key() -> tuple[str, str]:
    """A new key and the fingerprint to store for it. PBKDF2 is slow, so
    the hash runs off the event loop."""
    plaintext = generate_api_key()
    return plaintext, await asyncio.to_thread(key_hash, plaintext)


def well_formed_key(key: str) -> bool:
    """Whether *key* has the shape of a key :func:`generate_api_key` makes."""
    return _KEY_FORMAT.fullmatch(key) is not None


def key_prefix(key: str) -> str:
    if len(key) <= 10:
        return key
    return key[:10] + "…"


class _AuthCache:
    """Resolved API keys and the unauthenticated-access flag for one database.

    Keys are cached by their SHA-256, never in plain text. Entries expire after
    a few minutes (unknown keys and the auth setting sooner) and are dropped at
    once when keys, pins, providers, model groups or the auth setting change
    here, so dashboard edits apply to the next request. Other replicas
    sharing the database pick them up when their entries expire.
    """

    def __init__(self, db: Database) -> None:
        # (fetched at, value)
        self.allow_unauthenticated: tuple[float, bool] | None = None
        self.prefs: dict[str, tuple[float, int, RoutingPreferences]] = {}
        # SHA-256 of keys the database did not know -> when that expires.
        self.invalid: dict[str, float] = {}
        self.touched: dict[int, float] = {}
        # Bumped on every clear. A lookup that started before a clear must
        # not store its (possibly stale) answer afterwards.
        self.generation = 0
        db.add_change_listener(self._on_change)

    def _on_change(self, reason: str) -> None:
        # Groups too: a key allowed a group holds its name, which a rename
        # or delete changes.
        if reason in AUTH_CHANGES or reason in (CHANGE_PROVIDERS, CHANGE_GROUPS):
            self.clear()

    def clear(self) -> None:
        self.generation += 1
        self.allow_unauthenticated = None
        self.prefs.clear()
        self.invalid.clear()

    def is_invalid(self, fingerprint: str, now: float) -> bool:
        expires = self.invalid.get(fingerprint)
        if expires is None:
            return False
        if now < expires:
            return True
        del self.invalid[fingerprint]
        return False

    def remember_invalid(self, fingerprint: str, now: float) -> None:
        if len(self.invalid) >= _MAX_INVALID_KEYS:
            self.invalid = {k: t for k, t in self.invalid.items() if t > now}
            if len(self.invalid) >= _MAX_INVALID_KEYS:
                self.invalid.clear()
        self.invalid[fingerprint] = now + _INVALID_TTL_S


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
    now = time.monotonic()
    if cached is not None and now - cached[0] < _PREFS_TTL_S:
        return cached[1]
    generation = cache.generation
    value = await db.get_allow_unauthenticated()
    if cache.generation == generation:
        cache.allow_unauthenticated = (now, value)
    return value


_hash_slots: weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, asyncio.Semaphore] = (
    weakref.WeakKeyDictionary()
)


async def _hash_off_loop(
    fn: Callable[[str, bytes], str], key: str, pepper: bytes
) -> str:
    """Run a PBKDF2 hash in a thread, a few at a time.

    PBKDF2 takes tens of milliseconds; a flood of keys must not tie up every
    worker thread the event loop has.
    """
    loop = asyncio.get_running_loop()
    slots = _hash_slots.get(loop)
    if slots is None:
        slots = _hash_slots[loop] = asyncio.Semaphore(_MAX_CONCURRENT_HASHES)
    async with slots:
        return await asyncio.to_thread(fn, key, pepper)


async def _upgrade_hash(db: Database, key_id: int, modern: str) -> None:
    try:
        await db.set_api_key_hash(key_id, modern)
        logger.info("Upgraded API key %s to an HMAC-SHA256 hash", key_id)
    except Exception:
        logger.warning(
            "Could not upgrade the hash of API key %s", key_id, exc_info=True
        )


async def _lookup_record(db: Database, key: str, legacy: str) -> dict[str, Any] | None:
    """The key's database row, upgrading an older fingerprint to HMAC-SHA256.

    Tries the current fingerprint, then plain SHA-256 (*legacy*) and the HMAC
    under the previous secret, which are cheap, and only then PBKDF2 under
    each secret older releases may have used.
    """
    modern = key_hash(key)
    record = await db.lookup_api_key(modern)
    if record:
        return record
    for old in (legacy, key_hash_previous(key)):
        if old is None or old == modern:
            continue
        record = await db.lookup_api_key(old)
        if record:
            await _upgrade_hash(db, int(record["id"]), modern)
            return record
    for pepper in pbkdf2_peppers():
        old = await _hash_off_loop(key_hash_pbkdf2, key, pepper)
        record = await db.lookup_api_key(old)
        if record:
            await _upgrade_hash(db, int(record["id"]), modern)
            return record
    # A concurrent request may have just re-hashed it.
    return await db.lookup_api_key(modern)


async def _resolve_key(
    db: Database, key: str, legacy: str
) -> tuple[RoutingPreferences, int] | None:
    """Look the key up in the database. Slow path: hashing and queries.

    *legacy* is ``key_hash_legacy(key)``, already computed by the caller.
    """
    record = await _lookup_record(db, key, legacy)
    if not record:
        return None
    mode = str(record["routing_mode"]).lower()
    if mode not in ROUTING_MODES:
        mode = DEFAULT_MODE
    pins = await db.list_api_key_model_pins(int(record["id"]))
    owner = record.get("owner_user_id")
    user_limits = await limits_for_user(db, owner)
    prefs = RoutingPreferences(
        mode=mode,
        allow_fallback=bool(record["allow_fallback"]),
        pinned_providers={str(p["model_name"]): int(p["provider_id"]) for p in pins},
        strict_pins=bool(record.get("strict_pins")),
        session_affinity=record.get("session_affinity"),
        key_id=int(record["id"]),
        locality=_locality(record.get("locality")),
        daily_spend_cap_usd=record.get("daily_spend_cap_usd"),
        decision_policy=await _decision_policy(db, int(record["id"])),
        access=await model_access(db, int(record["id"])),
        owner_user_id=owner,
        daily_token_limit=record.get("daily_token_limit"),
        requests_per_minute=record.get("requests_per_minute"),
        user_daily_token_limit=user_limits.daily_token_limit,
        user_requests_per_minute=user_limits.requests_per_minute,
        user_daily_spend_cap_usd=user_limits.daily_spend_cap_usd,
    )
    return prefs, int(record["id"])


async def model_access(db: Database, key_id: int) -> ModelAccess | None:
    """The models and groups the key may use, or None for any."""
    access = await db.get_api_key_access(key_id)
    if access is None or not access["restricted"]:
        return None
    return ModelAccess(
        models=frozenset(access["models"]),
        groups=frozenset(str(g["name"]) for g in access["groups"]),
    )


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
    """The request was relayed by the worker link from this router's hub.

    Set by :class:`~.worker_link.relay.HubRelayMiddleware` only for jobs the
    worker link in this process sent; the headers alone mean nothing.
    """
    relay = request.scope.get(_RELAY_SCOPE_KEY)
    return isinstance(relay, dict) and bool(relay.get("hub"))


def hub_provider(request: Request) -> int | None:
    """The one provider a job from this router's hub is for, when the worker
    shares providers individually; None for any other request."""
    relay = request.scope.get(_RELAY_SCOPE_KEY)
    if not isinstance(relay, dict):
        return None
    pid = relay.get("provider")
    return pid if isinstance(pid, int) else None


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
        if prefs.owner_user_id is not None:
            request.state.quota_user_id = prefs.owner_user_id
        prefs = await _apply_spend_cap(db, prefs)
    return prefs


_CHARGED_STATE = "quota_charged"


async def charge_quotas(
    db: Database, request: Request, prefs: RoutingPreferences | None
) -> None:
    """Count a model request against its key's and owner's per-minute
    limits, and refuse it (429) when either has used its daily tokens.

    Only requests that run a model are charged, once each (a failover's
    second routing is not); listing models or reading the version is free.
    Arithmetic only once today's totals are loaded.
    """
    if prefs is None or prefs.key_id is None:
        return
    if getattr(request.state, _CHARGED_STATE, False):
        return
    owner = prefs.owner_user_id
    check_rate(
        db,
        key_id=prefs.key_id,
        key_rpm=prefs.requests_per_minute,
        user_id=owner,
        user_rpm=prefs.user_requests_per_minute,
    )
    request.state.quota_charged = True
    await check_tokens_left(
        db,
        key_id=prefs.key_id,
        key_limit=prefs.daily_token_limit,
        user_id=owner,
        user_limit=prefs.user_daily_token_limit,
    )


def spend_cap_of(prefs: RoutingPreferences) -> float | None:
    """The tighter of the key's and its owner's daily spend caps."""
    caps = [
        c
        for c in (
            prefs.daily_spend_cap_usd,
            prefs.user_daily_spend_cap_usd if prefs.owner_user_id is not None else None,
        )
        if c is not None
    ]
    return min(caps) if caps else None


async def _apply_spend_cap(
    db: Database, prefs: RoutingPreferences
) -> RoutingPreferences:
    """A key over its daily cap, or whose owner is over theirs, is limited
    to local providers, which cost nothing; a cloud-only key is refused
    until the next UTC day."""
    tracker = tracker_for(db)
    cap = prefs.daily_spend_cap_usd
    over: tuple[float, str] | None = None
    if (
        cap is not None
        and prefs.key_id is not None
        and await tracker.committed_today(prefs.key_id) >= cap
    ):
        over = (cap, "this API key")
    user_cap = prefs.user_daily_spend_cap_usd
    if (
        over is None
        and user_cap is not None
        and prefs.owner_user_id is not None
        and await tracker.user_committed_today(prefs.owner_user_id) >= user_cap
    ):
        over = (user_cap, "this user")
    if over is None:
        return prefs
    if prefs.locality == "cloud":
        which = "key_spend" if over[1] == "this API key" else "user_spend"
        tracker.refusals[which] += 1
        raise cap_reached(over[0], whose=over[1])
    return replace(prefs, locality="local", spend_capped=True)


async def model_allowed(
    rt: Router, prefs: RoutingPreferences | None, model: str
) -> bool:
    """Whether the request's API key may use *model* (keys not limited to
    some models may use any)."""
    if prefs is None or prefs.access is None:
        return True
    return (await rt.current_table()).allows(prefs.access, model)


async def require_model_access(
    rt: Router, prefs: RoutingPreferences | None, model: str
) -> None:
    """Refuse (HTTP 403) a model the request's API key may not use."""
    if not await model_allowed(rt, prefs, model):
        raise HTTPException(
            status_code=403, detail=f"This API key may not use model '{model}'"
        )


def _is_media_endpoint(path: str) -> bool:
    return "/images/" in path or "/audio/" in path


async def route_within_cap(
    rt: Router,
    db: Database,
    request: Request,
    model: str,
    prefs: RoutingPreferences | None,
    body: Any,
    *,
    session: str | None = None,
    exclude: frozenset[int] = frozenset(),
) -> RouteResult | None:
    """Route the request, holding its estimated cost against the key's cap.

    On a key with a daily spend cap, a request routed to a priced cloud model
    reserves its estimated cost (see spend.py); the request log settles the
    reservation (``request.state.spend_reservation``). If the estimate does
    not fit under the cap, or an image or audio request lands on a cloud model
    with no price (its cost could not be counted), the request is routed to
    local providers instead, or refused (HTTP 429 over the cap, 403 for an
    unpriced model) when that is not possible. A key with strict pins is
    refused rather than moved off a pinned model's provider. *exclude* names
    providers that already failed this request.
    """
    path = request.url.path
    await require_model_access(rt, prefs, model)
    await charge_quotas(db, request, prefs)
    if prefs is not None:
        await reserve_tokens(
            db,
            request,
            body,
            key_id=prefs.key_id,
            key_limit=prefs.daily_token_limit,
            user_id=prefs.owner_user_id,
            user_limit=prefs.user_daily_token_limit,
        )
    # Remembered with the session's binding: the prompt its provider caches.
    prompt_tokens = token_estimate(request, body)[0] if session is not None else 0
    try:
        result = await rt.route(
            model,
            path,
            prefs,
            session=session,
            prompt_tokens=prompt_tokens,
            exclude=exclude,
        )
    except PinUnavailableError as exc:
        if (
            prefs is not None
            and prefs.spend_capped
            and exc.outside_locality
            and spend_cap_of(prefs) is not None
        ):
            # Over the cap the key was narrowed to local providers, which
            # left its strict cloud pin out: the cap refused it, not the pin.
            raise cap_reached(spend_cap_of(prefs) or 0.0) from exc
        raise
    if (
        result is None
        or prefs is None
        or prefs.key_id is None
        or spend_cap_of(prefs) is None
        or not result.provider.is_cloud
    ):
        return result
    cap = prefs.daily_spend_cap_usd
    user_cap = (
        prefs.user_daily_spend_cap_usd if prefs.owner_user_id is not None else None
    )
    price = rt.price_for(result.provider.require_id(), result.model_key)
    if price is None:
        if not _is_media_endpoint(path):
            return result  # unknown cost, as without a cap
        status = 403
        reason = f"model '{model}' has no price, so its cost cannot be capped"
    else:
        estimate = price.cost(*token_estimate(request, body))
        if estimate <= 0:
            return result
        tracker = tracker_for(db)
        held = await tracker.reserve(
            prefs.key_id,
            cap,
            estimate,
            user_id=prefs.owner_user_id,
            user_cap=user_cap,
        )
        if held is not None:
            setattr(request.state, RESERVATION_STATE, held)
            return result
        status = 429
        key_over = (
            cap is not None
            and await tracker.committed_today(prefs.key_id) + estimate > cap
        )
        shown = cap if key_over or user_cap is None else user_cap
        whose = "this API key" if key_over or user_cap is None else "this user"
        reason = f"daily spend cap of ${shown or 0.0:.2f} reached for {whose}"
    local = None
    # A strict pin is never rerouted away from its provider, even to local.
    if prefs.locality != "cloud" and not _strictly_pinned(prefs, result):
        try:
            local = await rt.route(
                model,
                path,
                replace(prefs, locality="local", spend_capped=True),
                session=session,
                prompt_tokens=prompt_tokens,
                exclude=exclude,
            )
        except PinUnavailableError:
            # A strict cloud pin further down the fallback chain: the cap
            # is what refused the request, so say so.
            local = None
    if local is None:
        raise HTTPException(status_code=status, detail=reason)
    return local


def _strictly_pinned(prefs: RoutingPreferences, result: RouteResult) -> bool:
    return prefs.strict_pins and result.resolved_model in prefs.pinned_providers


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
        # This router's own worker relaying a hub job; the hub authenticated
        # the job, and HUB_HEADER still narrows it to the shared providers.
        if is_own_relay(request) or await _allow_unauthenticated(db, cache):
            return None
        raise HTTPException(
            status_code=401,
            detail="Missing API key: send API-KEY or Authorization: Bearer",
        )

    if not well_formed_key(key):
        raise HTTPException(status_code=401, detail="Invalid API-KEY")

    # The legacy digest is needed for the lookup anyway; it doubles as the
    # in-memory cache key so the plain key is never kept.
    fingerprint = key_hash_legacy(key)
    now = time.monotonic()
    if cache.is_invalid(fingerprint, now):
        raise HTTPException(status_code=401, detail="Invalid API-KEY")
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
        if cache.generation == generation:
            cache.remember_invalid(fingerprint, now)
        # Even with anonymous access on: a client that sends a wrong or
        # revoked key should find out rather than silently lose its limits.
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
