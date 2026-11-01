"""Daily spend per API key, for spend caps.

Totals are kept in memory per UTC day. The first check of a key reads its
spend so far from the request log; every logged request with a cost adds to
it. A restart re-reads, so a cap holds across restarts.
"""

from __future__ import annotations

import weakref
from datetime import UTC, datetime

from fastapi import HTTPException

from .database import Database


def _day_start(now: datetime | None = None) -> datetime:
    now = now or datetime.now(UTC)
    return now.replace(hour=0, minute=0, second=0, microsecond=0, tzinfo=None)


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

    def _roll(self) -> datetime:
        today = _day_start()
        if self._day != today:
            self._day = today
            self._totals.clear()
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
