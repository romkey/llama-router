"""Per-user and per-key quotas: daily tokens, requests per minute and a
daily spend cap across all of a user's keys.

A user's limit is their own setting, else the admin's default for users
(stored in ``app_settings`` beside the self-service key defaults; unset, the
default is unlimited). A key may carry a tighter token or rate limit of its
own; whichever is hit first applies. Keys without an owner have key-level
limits only.

Limits are resolved when the key is (the auth cache's slow path) and carried
on :class:`~.router.RoutingPreferences`; usage lives in memory in
:class:`~.spend.SpendTracker`, seeded from the request log, so a warm
request only does arithmetic.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from fastapi import Request

from .alerts import alerts
from .config import settings
from .database import Database
from .pricing import _max_output_tokens, estimate_tokens
from .spend import (
    TOKEN_RESERVATION_STATE,
    TokenReservation,
    rate_limit_reached,
    token_limit_reached,
    tracker_for,
)

DEFAULT_TOKENS_SETTING = "user_default_daily_token_limit"
DEFAULT_RPM_SETTING = "user_default_requests_per_minute"
DEFAULT_SPEND_SETTING = "user_default_daily_spend_cap_usd"
QUOTA_FIELDS = ("daily_token_limit", "requests_per_minute", "daily_spend_cap_usd")

# Where a request's token estimate is kept (``request.state.token_estimate``).
_ESTIMATE_STATE = "token_estimate"


@dataclass(frozen=True, slots=True)
class UserLimits:
    daily_token_limit: int | None = None
    requests_per_minute: int | None = None
    daily_spend_cap_usd: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "daily_token_limit": self.daily_token_limit,
            "requests_per_minute": self.requests_per_minute,
            "daily_spend_cap_usd": self.daily_spend_cap_usd,
        }


def _int(raw: str | None) -> int | None:
    return int(raw) if raw else None


async def default_limits(db: Database) -> UserLimits:
    """The admin's default limits for users (unlimited until set)."""
    raw_spend = await db.get_app_setting(DEFAULT_SPEND_SETTING)
    return UserLimits(
        daily_token_limit=_int(await db.get_app_setting(DEFAULT_TOKENS_SETTING)),
        requests_per_minute=_int(await db.get_app_setting(DEFAULT_RPM_SETTING)),
        daily_spend_cap_usd=float(raw_spend) if raw_spend else None,
    )


async def set_default_limits(db: Database, limits: UserLimits) -> None:
    """Store the default limits; the auth cache must be cleared after."""
    values = {
        DEFAULT_TOKENS_SETTING: limits.daily_token_limit,
        DEFAULT_RPM_SETTING: limits.requests_per_minute,
        DEFAULT_SPEND_SETTING: limits.daily_spend_cap_usd,
    }
    for key, value in values.items():
        await db.set_app_setting(key, "" if value is None else str(value))


def effective_limits(user: dict[str, Any] | None, defaults: UserLimits) -> UserLimits:
    """A user's own limits, each falling back to the default when unset."""
    if user is None:
        return UserLimits()

    def pick(name: str) -> Any:
        own = user.get(name)
        return own if own is not None else getattr(defaults, name)

    return UserLimits(
        daily_token_limit=pick("daily_token_limit"),
        requests_per_minute=pick("requests_per_minute"),
        daily_spend_cap_usd=pick("daily_spend_cap_usd"),
    )


async def limits_for_user(db: Database, user_id: int | None) -> UserLimits:
    if user_id is None:
        return UserLimits()
    user = await db.get_dashboard_user_by_id(user_id)
    return effective_limits(user, await default_limits(db))


def token_estimate(request: Request | None, body: Any) -> tuple[int, int]:
    """The request's estimated (prompt, completion) tokens, worked out once
    per request (it is O(body size)) and shared by the spend cap, quotas
    and session affinity."""
    if request is not None:
        cached = getattr(request.state, _ESTIMATE_STATE, None)
        if isinstance(cached, tuple):
            return int(cached[0]), int(cached[1])
    estimate = estimate_tokens(body, settings.spend_estimate_completion_tokens)
    if request is not None:
        setattr(request.state, _ESTIMATE_STATE, estimate)
    return estimate


def _whose(name: str) -> str:
    return "this API key" if name.startswith("key") else "this user"


def check_rate(
    db: Database,
    *,
    key_id: int | None,
    key_rpm: int | None,
    user_id: int | None,
    user_rpm: int | None,
) -> None:
    """Count the request against its key's and user's per-minute limits,
    or refuse it (429 with ``Retry-After``). No I/O."""
    if key_rpm is None and user_rpm is None:
        return
    hit = tracker_for(db).hit(
        key_id=key_id, key_rpm=key_rpm, user_id=user_id, user_rpm=user_rpm
    )
    if hit is not None:
        name, limit, retry = hit
        who = key_id if name.startswith("key") else user_id
        alerts.quota_reached(
            "requests_per_minute",
            f"{'API key' if name.startswith('key') else 'User'} {who}",
            f"{name}:{who}",
        )
        raise rate_limit_reached(limit, _whose(name), retry)


async def check_tokens_left(
    db: Database,
    *,
    key_id: int | None,
    key_limit: int | None,
    user_id: int | None,
    user_limit: int | None,
) -> None:
    """Refuse (429 until the next UTC day) a request whose key or user has
    used its daily tokens."""
    tracker = tracker_for(db)
    if (
        key_id is not None
        and key_limit is not None
        and await tracker.key_tokens_committed(key_id) >= key_limit
    ):
        tracker.refusals["key_tokens"] += 1
        alerts.quota_reached("tokens", f"API key {key_id}", f"key:{key_id}")
        raise token_limit_reached(key_limit, "this API key")
    if (
        user_id is not None
        and user_limit is not None
        and await tracker.user_tokens_committed(user_id) >= user_limit
    ):
        tracker.refusals["user_tokens"] += 1
        alerts.quota_reached("tokens", f"User {user_id}", f"user:{user_id}")
        raise token_limit_reached(user_limit, "this user")


async def reserve_tokens(
    db: Database,
    request: Request | None,
    body: Any,
    *,
    key_id: int | None,
    key_limit: int | None,
    user_id: int | None,
    user_limit: int | None,
) -> TokenReservation | None:
    """Hold the request's estimated tokens against the limits (once per
    request; failover reuses it), or refuse it with 429.

    The hold is kept on ``request.state`` and settled by the request log.
    Requests with no token limit hold nothing.
    """
    if key_limit is None and user_limit is None:
        return None
    if request is not None:
        held = getattr(request.state, TOKEN_RESERVATION_STATE, None)
        if isinstance(held, TokenReservation):
            return held
    prompt, completion = token_estimate(request, body)
    tracker = tracker_for(db)
    held_or_why = await tracker.reserve_tokens(
        prompt + completion,
        key_id=key_id,
        key_limit=key_limit,
        user_id=user_id,
        user_limit=user_limit,
        # A request that sets its output limit must fit whole; one without
        # (its completion is a guess) may run on what is left.
        minimum=(
            None if isinstance(body, dict) and _max_output_tokens(body) else prompt + 1
        ),
    )
    if isinstance(held_or_why, str):
        tracker.refusals[held_or_why] += 1
        limit = key_limit if held_or_why == "key_tokens" else user_limit
        raise token_limit_reached(int(limit or 0), _whose(held_or_why))
    if request is not None:
        setattr(request.state, TOKEN_RESERVATION_STATE, held_or_why)
    return held_or_why
