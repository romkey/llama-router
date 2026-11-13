"""API key expiry, renewal, pausing and what happens to keys when their
owner loses access.

- A key with ``expires_at`` stops working then (401 "API key expired") and
  is kept. Keys users issue themselves must expire within
  ``LLAMA_ROUTER_MAX_KEY_LIFETIME_DAYS`` (0: no limit); admins may issue
  keys that never expire.
- Renewal gives a key a new expiry, now plus ``LLAMA_ROUTER_KEY_RENEWAL_DAYS``
  (not added to what is left), with the same secret. An admin chooses in the
  dashboard whether owners may renew: ``off`` (admins only), ``allowed``
  (default) or ``approval`` (owners ask, admins approve or deny).
- A paused key gets 403 "API key paused"; its settings, pins and history
  stay. An owner's pause the owner or an admin lifts; an admin hold
  (``pause_locked``) only an admin lifts. The router places holds itself
  when an owner is un-approved, refused at single sign-on for group
  membership, or (``LLAMA_ROUTER_OIDC_KEY_MAX_IDLE_DAYS``) idle too long.
  Signing in again never lifts a hold.

Expiry and pause state are in the cached key record, so a request pays one
comparison; every change clears the auth cache.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from datetime import UTC, datetime, timedelta
from typing import Any

from fastapi import HTTPException

from .config import settings
from .database import Database
from .repositories.base import _utc_datetime

logger = logging.getLogger(__name__)

RENEWAL_SETTING = "key_renewal_mode"
RENEWAL_MODES = ("off", "allowed", "approval")
DEFAULT_RENEWAL_MODE = "allowed"
HOLD_DEFAULT_SETTING = "admin_pause_locks_by_default"

REASON_UNAPPROVED = "owner not approved"
REASON_NOT_MEMBER = "no longer a member"
REASON_INACTIVE = "owner inactive"

SWEEP_INTERVAL_S = 3600.0
# Expiry within this many days is shown as "expires in N days".
EXPIRY_WARNING_DAYS = 7


def now_utc() -> datetime:
    return datetime.now(UTC)


def max_lifetime() -> timedelta | None:
    days = settings.max_key_lifetime_days
    return timedelta(days=days) if days > 0 else None


def renewal_period() -> timedelta:
    days = settings.key_renewal_days or settings.max_key_lifetime_days or 90
    return timedelta(days=days)


async def renewal_mode(db: Database) -> str:
    raw = (await db.get_app_setting(RENEWAL_SETTING) or "").strip().lower()
    return raw if raw in RENEWAL_MODES else DEFAULT_RENEWAL_MODE


async def set_renewal_mode(db: Database, mode: str) -> None:
    if mode not in RENEWAL_MODES:
        raise HTTPException(
            status_code=400, detail=f"renewal mode must be one of {RENEWAL_MODES}"
        )
    await db.set_app_setting(RENEWAL_SETTING, mode)


async def holds_by_default(db: Database) -> bool:
    """Whether "Require an admin to unpause" starts ticked (default on)."""
    raw = await db.get_app_setting(HOLD_DEFAULT_SETTING)
    return raw is None or raw.strip() not in ("0", "false", "off")


async def set_holds_by_default(db: Database, value: bool) -> None:
    await db.set_app_setting(HOLD_DEFAULT_SETTING, "1" if value else "0")


def parse_expiry(value: Any) -> datetime | None:
    """An ``expires_at`` from JSON: ISO 8601 (a date alone means the end of
    that UTC day); null or blank for never."""
    if value is None or (isinstance(value, str) and not value.strip()):
        return None
    if not isinstance(value, str):
        raise HTTPException(status_code=400, detail="expires_at must be a date")
    raw = value.strip().replace("Z", "+00:00")
    try:
        when = datetime.fromisoformat(raw)
    except ValueError as exc:
        raise HTTPException(
            status_code=400, detail="expires_at must be an ISO 8601 date"
        ) from exc
    if len(raw) == 10:
        # A date alone (the forms' date fields) lasts through that UTC day.
        when += timedelta(days=1)
    return when if when.tzinfo is not None else when.replace(tzinfo=UTC)


def self_issued_expiry(
    requested: datetime | None, *, now: datetime | None = None
) -> datetime | None:
    """The expiry for a key a user issues or edits themselves: *requested*,
    which must fall within the maximum lifetime; with no request, the end of
    the maximum lifetime (or never, without one)."""
    now = now or now_utc()
    if requested is not None and requested <= now:
        raise HTTPException(status_code=400, detail="expires_at must be in the future")
    limit = max_lifetime()
    if limit is None:
        return requested
    latest = now + limit
    if requested is None:
        return latest
    if requested > latest + timedelta(minutes=1):
        raise HTTPException(
            status_code=400,
            detail=(
                f"keys you issue yourself must expire within "
                f"{settings.max_key_lifetime_days} days"
            ),
        )
    return requested


def renewed_expiry(owner_issued: bool, *, now: datetime | None = None) -> datetime:
    """The expiry a renewal gives: now plus the renewal period, within the
    maximum lifetime for a key its owner issued."""
    now = now or now_utc()
    when = now + renewal_period()
    limit = max_lifetime()
    if owner_issued and limit is not None:
        when = min(when, now + limit)
    return when


def key_status(key: dict[str, Any], *, now: datetime | None = None) -> dict[str, Any]:
    """What to show for a key: ``status`` is active, expiring, expired,
    renewal_requested, paused or held; ``label`` the words."""
    now = now or now_utc()
    expires = key.get("expires_at")
    if key.get("paused_at") is not None:
        held = bool(key.get("pause_locked"))
        reason = key.get("pause_reason")
        label = "held" if held else "paused"
        if reason:
            label += f" ({reason})"
        return {"status": "held" if held else "paused", "label": label}
    if expires is not None and expires <= now:
        if key.get("renewal_requested_at") is not None:
            return {
                "status": "renewal_requested",
                "label": "expired, renewal requested",
            }
        return {"status": "expired", "label": "expired"}
    if key.get("renewal_requested_at") is not None:
        return {"status": "renewal_requested", "label": "renewal requested"}
    if expires is not None:
        days = (expires - now).total_seconds() / 86400
        if days <= EXPIRY_WARNING_DAYS:
            n = max(0, int(days))
            return {
                "status": "expiring",
                "label": f"expires in {n} day{'' if n == 1 else 's'}",
                "days_left": n,
            }
    return {"status": "active", "label": "active"}


def with_status(keys: list[dict[str, Any]]) -> list[dict[str, Any]]:
    now = now_utc()
    for k in keys:
        k.update(
            {f"key_{name}": value for name, value in key_status(k, now=now).items()}
        )
    return keys


async def hold_user_keys(
    db: Database, user_id: int, reason: str, *, by_user_id: int | None = None
) -> int:
    """Put an admin hold on every key *user_id* issued themselves."""
    n = await db.hold_user_api_keys(
        user_id, reason, by_user_id=by_user_id, when=now_utc()
    )
    if n:
        logger.warning(
            "Held %d API key%s of dashboard user %d: %s",
            n,
            "" if n == 1 else "s",
            user_id,
            reason,
        )
    return n


async def sweep_idle_sso_users(db: Database, *, now: datetime | None = None) -> int:
    """Hold the keys of single sign-on users idle for
    ``oidc_key_max_idle_days``; returns how many keys were held."""
    days = settings.oidc_key_max_idle_days
    if days <= 0:
        return 0
    now = now or now_utc()
    cutoff = now - timedelta(days=days)
    held = 0
    for user in await db.list_dashboard_users():
        if user.get("auth_source") != "oidc":
            continue
        last = _utc_datetime(user.get("last_login_at") or user.get("created_at"))
        if last is not None and last > cutoff:
            continue
        held += await hold_user_keys(db, int(user["id"]), REASON_INACTIVE)
    return held


class IdleSweep:
    """Runs :func:`sweep_idle_sso_users` hourly while the setting is on."""

    def __init__(self, db: Database) -> None:
        self._db = db
        self._task: asyncio.Task[None] | None = None

    def start(self) -> None:
        if self._task is None and settings.oidc_key_max_idle_days > 0:
            self._task = asyncio.create_task(self._run(), name="idle-key-sweep")

    async def _run(self) -> None:
        while True:
            try:
                await sweep_idle_sso_users(self._db)
            except Exception:
                logger.exception("Idle key sweep failed")
            await asyncio.sleep(SWEEP_INTERVAL_S)

    async def stop(self) -> None:
        task, self._task = self._task, None
        if task is not None:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
