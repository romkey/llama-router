"""Daily spend and tokens per API key and per key owner, for spend caps
and quotas, and per-minute request counts for rate limits.

Totals are kept in memory per UTC day. The first check of a key reads its
spend so far from the request log; every logged request with a cost adds to
it. A restart re-reads, so a cap holds across restarts.

A priced cloud request on a capped key first reserves its estimated cost
(:meth:`SpendTracker.reserve`). Reservations count against the cap until the
request is logged, when the reservation is swapped for the actual cost (or
released if the request failed), so concurrent requests cannot all pass a cap
they would exceed together. :class:`SpendReservationMiddleware` releases a
reservation still held when its request ends without being logged.
"""

from __future__ import annotations

import math
import time
import weakref
from collections import Counter
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any

from fastapi import HTTPException

from .database import Database


def _day_start(now: datetime | None = None) -> datetime:
    now = now or datetime.now(UTC)
    return now.replace(hour=0, minute=0, second=0, microsecond=0, tzinfo=None)


def next_day_start(now: datetime | None = None) -> datetime:
    """When the UTC day after *now* starts (aware)."""
    return _day_start(now).replace(tzinfo=UTC) + timedelta(days=1)


def seconds_to_next_day(now: datetime | None = None) -> int:
    now = now or datetime.now(UTC)
    return max(1, math.ceil((next_day_start(now) - now).total_seconds()))


@dataclass(slots=True)
class Usage:
    """What a key or user has used today."""

    cost: float = 0.0
    tokens: int = 0


@dataclass(slots=True)
class Reservation:
    """An estimated cost held against a key's (and its owner's) cap while
    its request runs."""

    key_id: int | None
    amount: float
    day: datetime
    tracker: SpendTracker = field(repr=False)
    released: bool = False
    charged: bool = False
    user_id: int | None = None

    def release(self) -> None:
        """Stop holding the amount against the cap. Idempotent."""
        if self.released:
            return
        self.released = True
        self.tracker._unhold(self)


@dataclass(slots=True)
class TokenReservation:
    """Estimated tokens held against a key's and its owner's daily token
    limits while the request runs; settled from the logged counts."""

    key_id: int | None
    user_id: int | None
    tokens: int
    day: datetime
    tracker: SpendTracker = field(repr=False)
    released: bool = False

    def release(self) -> None:
        if self.released:
            return
        self.released = True
        self.tracker._unhold_tokens(self)


def cap_reached(cap: float, *, whose: str = "this API key") -> HTTPException:
    """The error a key (or owner) past its daily spend cap is refused with."""
    reset = next_day_start().strftime("%Y-%m-%dT%H:%M:%SZ")
    return HTTPException(
        status_code=429,
        detail=f"daily spend cap of ${cap:.2f} reached for {whose}; resets at {reset}",
        headers={"Retry-After": str(seconds_to_next_day())},
    )


def token_limit_reached(limit: int, whose: str) -> HTTPException:
    reset = next_day_start().strftime("%Y-%m-%dT%H:%M:%SZ")
    return HTTPException(
        status_code=429,
        detail=(f"daily token limit of {limit} reached for {whose}; resets at {reset}"),
        headers={"Retry-After": str(seconds_to_next_day())},
    )


def rate_limit_reached(limit: int, whose: str, retry_after: int) -> HTTPException:
    return HTTPException(
        status_code=429,
        detail=(
            f"rate limit of {limit} requests per minute reached for {whose}; "
            f"try again in {retry_after}s"
        ),
        headers={"Retry-After": str(retry_after)},
    )


class SpendTracker:
    """Today's spend and tokens per API key and per key owner, held
    reservations, and per-minute request counts. In memory, for the one
    router process; seeded from the request log."""

    def __init__(self, db: Database) -> None:
        self._db = db
        self._day: datetime | None = None
        self._keys: dict[int, Usage] = {}
        self._users: dict[int, Usage] = {}
        self._reserved: dict[int, float] = {}
        self._user_reserved: dict[int, float] = {}
        self._key_tokens_held: dict[int, int] = {}
        self._user_tokens_held: dict[int, int] = {}
        # id -> (minute number, requests in it): fixed one-minute windows.
        self._key_minute: dict[int, tuple[int, int]] = {}
        self._user_minute: dict[int, tuple[int, int]] = {}
        # Requests refused, by limit ("key_tokens", "user_rpm", ...).
        self.refusals: Counter[str] = Counter()

    def _roll(self) -> datetime:
        today = _day_start()
        if self._day != today:
            self._day = today
            self._keys.clear()
            self._users.clear()
            self._reserved.clear()
            self._user_reserved.clear()
            self._key_tokens_held.clear()
            self._user_tokens_held.clear()
        return today

    # --- seeding ---

    async def seed_today(self) -> None:
        """Load today's totals of every key and user active today (startup)."""
        today = self._roll()
        keys = await self._db.usage_by_since(today, "api_key_id")
        users = await self._db.usage_by_since(today, "user_id")
        if self._roll() != today:
            return
        for table, rows in ((self._keys, keys), (self._users, users)):
            for who, (cost, tokens) in rows.items():
                known = table.get(who)
                table[who] = Usage(
                    max(cost, known.cost if known else 0.0),
                    max(tokens, known.tokens if known else 0),
                )

    async def _key_usage(self, key_id: int) -> Usage:
        today = self._roll()
        usage = self._keys.get(key_id)
        if usage is None:
            cost, tokens = await self._db.usage_since(today, key_id=key_id)
            # A request logged while that query ran was already added.
            known = self._keys.get(key_id)
            usage = Usage(
                max(cost, known.cost if known else 0.0),
                max(tokens, known.tokens if known else 0),
            )
            self._keys[key_id] = usage
        return usage

    async def _user_usage(self, user_id: int) -> Usage:
        today = self._roll()
        usage = self._users.get(user_id)
        if usage is None:
            cost, tokens = await self._db.usage_since(today, user_id=user_id)
            known = self._users.get(user_id)
            usage = Usage(
                max(cost, known.cost if known else 0.0),
                max(tokens, known.tokens if known else 0),
            )
            self._users[user_id] = usage
        return usage

    # --- reading ---

    async def spent_today(self, key_id: int) -> float:
        return (await self._key_usage(key_id)).cost

    async def committed_today(self, key_id: int) -> float:
        """Spend so far today plus what running requests have reserved."""
        spent = await self.spent_today(key_id)
        return spent + self._reserved.get(key_id, 0.0)

    async def user_committed_today(self, user_id: int) -> float:
        usage = await self._user_usage(user_id)
        return usage.cost + self._user_reserved.get(user_id, 0.0)

    async def key_tokens_committed(self, key_id: int) -> int:
        usage = await self._key_usage(key_id)
        return usage.tokens + self._key_tokens_held.get(key_id, 0)

    async def user_tokens_committed(self, user_id: int) -> int:
        usage = await self._user_usage(user_id)
        return usage.tokens + self._user_tokens_held.get(user_id, 0)

    def usage(self, *, key_id: int | None = None, user_id: int | None = None) -> Usage:
        """Today's known usage (zero when not loaded yet). No I/O."""
        self._roll()
        if key_id is not None:
            return self._keys.get(key_id) or Usage()
        if user_id is not None:
            return self._users.get(user_id) or Usage()
        return Usage()

    # --- spend reservations ---

    async def reserve(
        self,
        key_id: int | None,
        cap: float | None,
        amount: float,
        *,
        user_id: int | None = None,
        user_cap: float | None = None,
    ) -> Reservation | None:
        """Hold *amount* against the key's *cap* and its owner's *user_cap*
        (None: no cap; no key for a dashboard chat); None if it would not
        fit under either."""
        spent = await self.spent_today(key_id) if key_id is not None else 0.0
        user_spent = (
            await self._user_usage(user_id)
            if user_id is not None and user_cap is not None
            else None
        )
        # No await from here on: the check and the hold happen together.
        today = self._roll()
        held = self._reserved.get(key_id, 0.0) if key_id is not None else 0.0
        if cap is not None and spent + held + amount > cap:
            return None
        if user_spent is not None and user_cap is not None and user_id is not None:
            user_held = self._user_reserved.get(user_id, 0.0)
            if user_spent.cost + user_held + amount > user_cap:
                return None
        if key_id is not None:
            self._reserved[key_id] = held + amount
        if user_id is not None:
            self._user_reserved[user_id] = (
                self._user_reserved.get(user_id, 0.0) + amount
            )
        return Reservation(key_id, amount, today, self, user_id=user_id)

    def settle(self, reservation: Reservation, cost: float | None) -> None:
        """Swap *reservation* for the request's actual *cost* (None or 0 when
        it failed or cost nothing). The cost is charged once, even if the
        reservation was already released."""
        reservation.release()
        if reservation.charged:
            return
        reservation.charged = True
        if cost:
            self.add(reservation.key_id, cost, user_id=reservation.user_id)

    def _unhold(self, reservation: Reservation) -> None:
        today = self._roll()
        if reservation.day != today:
            return
        if reservation.key_id is not None:
            _take(self._reserved, reservation.key_id, reservation.amount)
        if reservation.user_id is not None:
            _take(self._user_reserved, reservation.user_id, reservation.amount)

    def add(
        self,
        key_id: int | None,
        cost: float,
        *,
        user_id: int | None = None,
        tokens: int = 0,
    ) -> None:
        """Count a logged request's *cost* and *tokens* against the key and
        user, where their totals are loaded (else the next load reads it)."""
        self._roll()
        if key_id is not None and key_id in self._keys:
            self._keys[key_id].cost += cost
            self._keys[key_id].tokens += tokens
        if user_id is not None and user_id in self._users:
            self._users[user_id].cost += cost
            self._users[user_id].tokens += tokens

    def known(self) -> dict[int, float]:
        self._roll()
        return {k: u.cost for k, u in self._keys.items()}

    def known_users(self) -> dict[int, Usage]:
        self._roll()
        return dict(self._users)

    # --- token reservations ---

    async def reserve_tokens(
        self,
        tokens: int,
        *,
        key_id: int | None,
        key_limit: int | None,
        user_id: int | None,
        user_limit: int | None,
        minimum: int | None = None,
    ) -> TokenReservation | str:
        """Hold *tokens* against the key's and the user's daily token limits.

        When *tokens* does not fit but *minimum* (the part the request is
        sure to use, its prompt) does, everything left is held instead, so
        a request without an output limit may still run on the remainder
        while no other request can share it.

        Returns the reservation, or which limit it would pass
        (``"key_tokens"`` or ``"user_tokens"``).
        """
        key_usage = (
            await self._key_usage(key_id)
            if key_id is not None and key_limit is not None
            else None
        )
        user_usage = (
            await self._user_usage(user_id)
            if user_id is not None and user_limit is not None
            else None
        )
        # No await from here on: the checks and the holds happen together.
        today = self._roll()
        rooms: list[tuple[str, int]] = []
        if key_usage is not None and key_id is not None and key_limit is not None:
            held = self._key_tokens_held.get(key_id, 0)
            rooms.append(("key_tokens", key_limit - key_usage.tokens - held))
        if user_usage is not None and user_id is not None and user_limit is not None:
            held = self._user_tokens_held.get(user_id, 0)
            rooms.append(("user_tokens", user_limit - user_usage.tokens - held))
        floor = tokens if minimum is None else min(tokens, minimum)
        for name, room in rooms:
            if room < floor or room <= 0:
                return name
        if rooms:
            tokens = min(tokens, *(room for _, room in rooms))
        if key_id is not None:
            held_now = self._key_tokens_held.get(key_id, 0)
            self._key_tokens_held[key_id] = held_now + tokens
        if user_id is not None:
            self._user_tokens_held[user_id] = (
                self._user_tokens_held.get(user_id, 0) + tokens
            )
        return TokenReservation(key_id, user_id, tokens, today, self)

    def _unhold_tokens(self, reservation: TokenReservation) -> None:
        today = self._roll()
        if reservation.day != today:
            return
        if reservation.key_id is not None:
            _take(self._key_tokens_held, reservation.key_id, reservation.tokens)
        if reservation.user_id is not None:
            _take(self._user_tokens_held, reservation.user_id, reservation.tokens)

    # --- requests per minute ---

    def hit(
        self,
        *,
        key_id: int | None,
        key_rpm: int | None,
        user_id: int | None,
        user_rpm: int | None,
        now: float | None = None,
    ) -> tuple[str, int, int] | None:
        """Count one request in this minute's windows.

        Returns None, or (the limit hit, the limit, seconds until the window
        ends); a refused request is not counted.
        """
        t = time.time() if now is None else now
        minute = int(t // 60)
        retry = max(1, 60 - int(t % 60))
        checks = (
            ("key_rpm", self._key_minute, key_id, key_rpm),
            ("user_rpm", self._user_minute, user_id, user_rpm),
        )
        for name, windows, who, limit in checks:
            if who is None or limit is None:
                continue
            window = windows.get(who)
            used = window[1] if window is not None and window[0] == minute else 0
            if used >= limit:
                self.refusals[name] += 1
                return name, limit, retry
        for _name, windows, who, limit in checks:
            if who is None or limit is None:
                continue
            window = windows.get(who)
            used = window[1] if window is not None and window[0] == minute else 0
            windows[who] = (minute, used + 1)
        return None

    def requests_this_minute(
        self, *, key_id: int | None = None, user_id: int | None = None
    ) -> int:
        minute = int(time.time() // 60)
        if key_id is not None:
            windows, who = self._key_minute, key_id
        elif user_id is not None:
            windows, who = self._user_minute, user_id
        else:
            return 0
        window = windows.get(who)
        return window[1] if window is not None and window[0] == minute else 0


def _take(held: dict[int, Any], who: int, amount: float) -> None:
    left = held.get(who, 0) - amount
    if left > 1e-12:
        held[who] = left
    else:
        held.pop(who, None)


_trackers: weakref.WeakKeyDictionary[Database, SpendTracker] = (
    weakref.WeakKeyDictionary()
)


def tracker_for(db: Database) -> SpendTracker:
    t = _trackers.get(db)
    if t is None:
        t = _trackers[db] = SpendTracker(db)
    return t


# Where a request's reservation is kept (``request.state.spend_reservation``).
RESERVATION_STATE = "spend_reservation"
# And its token reservation (``request.state.token_reservation``).
TOKEN_RESERVATION_STATE = "token_reservation"  # noqa: S105 - a state name


# Further holds a request took (decision fan-outs), released at its end.
EXTRA_RESERVATIONS_STATE = "extra_reservations"


def keep_for_release(request: Any, held: Reservation | TokenReservation) -> None:
    """Release *held* when *request* ends, if nothing settled it first."""
    if request is None:
        return
    state = getattr(request, "state", None)
    if state is None:
        return
    kept = getattr(state, EXTRA_RESERVATIONS_STATE, None)
    if not isinstance(kept, list):
        kept = []
        setattr(state, EXTRA_RESERVATIONS_STATE, kept)
    kept.append(held)


class SpendReservationMiddleware:
    """Releases a request's spend reservation if it is still held when the
    request ends, for example when the handler failed before the request was
    logged. A pure ASGI middleware: it neither buffers nor wraps streams, and
    it runs after a streamed body has been sent."""

    def __init__(self, app: Any) -> None:
        self.app = app

    async def __call__(self, scope: dict[str, Any], receive: Any, send: Any) -> None:
        try:
            await self.app(scope, receive, send)
        finally:
            state = scope.get("state")
            if isinstance(state, dict):
                held = state.pop(RESERVATION_STATE, None)
                if isinstance(held, Reservation):
                    held.release()
                tokens = state.pop(TOKEN_RESERVATION_STATE, None)
                if isinstance(tokens, TokenReservation):
                    tokens.release()
                for extra in state.pop(EXTRA_RESERVATIONS_STATE, None) or ():
                    if isinstance(extra, (Reservation, TokenReservation)):
                        extra.release()
