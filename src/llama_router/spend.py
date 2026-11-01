"""Daily spend per API key, for spend caps.

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

import weakref
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from fastapi import HTTPException

from .database import Database


def _day_start(now: datetime | None = None) -> datetime:
    now = now or datetime.now(UTC)
    return now.replace(hour=0, minute=0, second=0, microsecond=0, tzinfo=None)


@dataclass(slots=True)
class Reservation:
    """An estimated cost held against a key's cap while its request runs."""

    key_id: int
    amount: float
    day: datetime
    tracker: SpendTracker = field(repr=False)
    released: bool = False
    charged: bool = False

    def release(self) -> None:
        """Stop holding the amount against the cap. Idempotent."""
        if self.released:
            return
        self.released = True
        self.tracker._unhold(self)


def cap_reached(cap: float) -> HTTPException:
    """The error a key past its daily spend cap is refused with."""
    return HTTPException(
        status_code=429,
        detail=f"daily spend cap of ${cap:.2f} reached for this API key",
    )


class SpendTracker:
    def __init__(self, db: Database) -> None:
        self._db = db
        self._day: datetime | None = None
        self._totals: dict[int, float] = {}
        self._reserved: dict[int, float] = {}

    def _roll(self) -> datetime:
        today = _day_start()
        if self._day != today:
            self._day = today
            self._totals.clear()
            self._reserved.clear()
        return today

    async def spent_today(self, key_id: int) -> float:
        today = self._roll()
        total = self._totals.get(key_id)
        if total is None:
            total = await self._db.spend_since(key_id, today)
            # A request logged while that query ran was already added.
            total = max(total, self._totals.get(key_id, 0.0))
            self._totals[key_id] = total
        return total

    async def committed_today(self, key_id: int) -> float:
        """Spend so far today plus what running requests have reserved."""
        spent = await self.spent_today(key_id)
        return spent + self._reserved.get(key_id, 0.0)

    async def reserve(
        self, key_id: int, cap: float, amount: float
    ) -> Reservation | None:
        """Hold *amount* against the key's *cap*; None if it would not fit."""
        spent = await self.spent_today(key_id)
        # No await from here on: the check and the hold happen together.
        today = self._roll()
        held = self._reserved.get(key_id, 0.0)
        if spent + held + amount > cap:
            return None
        self._reserved[key_id] = held + amount
        return Reservation(key_id, amount, today, self)

    def settle(self, reservation: Reservation, cost: float | None) -> None:
        """Swap *reservation* for the request's actual *cost* (None or 0 when
        it failed or cost nothing). The cost is charged once, even if the
        reservation was already released."""
        reservation.release()
        if reservation.charged:
            return
        reservation.charged = True
        if cost:
            self.add(reservation.key_id, cost)

    def _unhold(self, reservation: Reservation) -> None:
        today = self._roll()
        if reservation.day == today:
            key = reservation.key_id
            left = self._reserved.get(key, 0.0) - reservation.amount
            if left > 1e-12:
                self._reserved[key] = left
            else:
                self._reserved.pop(key, None)

    def add(self, key_id: int, cost: float) -> None:
        self._roll()
        if key_id in self._totals:
            self._totals[key_id] += cost

    def known(self) -> dict[int, float]:
        self._roll()
        return dict(self._totals)


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
