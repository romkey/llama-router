"""Per-provider concurrency limits (``providers.max_concurrent``).

A provider with a limit gets a :class:`SlotGate`: at most *limit* requests
run on it at once, and the rest wait in line at the queue level their key's
routing mode sets (:func:`queue_level`), first come first served within a
level. The same gate, without a provider, is the global and per-key upstream
limit in ``limits.py``. A provider without one
has no gate at all, so an unconfigured limit costs one dictionary lookup.

Taking a free slot never suspends: with no one waiting and a slot free,
:meth:`SlotGate.acquire` returns without awaiting anything. A request that
has to wait is bounded by ``upstream_queue_timeout_seconds`` and its request
deadline, then refused with :class:`ProviderAtCapacity` (429 and
``Retry-After``). A released slot is handed straight to the next waiter, so a
newcomer cannot overtake the queue at its level.
"""

from __future__ import annotations

import asyncio
import contextlib
import time
from collections import deque
from dataclasses import dataclass, field

from fastapi import HTTPException

from .config import settings

# Upper bounds (seconds) of the queue-wait histogram's buckets.
WAIT_BUCKETS: tuple[float, ...] = (0.01, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60)


class ProviderAtCapacity(HTTPException):
    """Every slot on the provider stayed taken for as long as the request
    could wait."""

    def __init__(self, provider_id: int, limit: int) -> None:
        self.provider_id = provider_id
        self.limit = limit
        super().__init__(
            status_code=429,
            detail=(
                f"provider is at capacity ({limit} concurrent requests); "
                "try again shortly"
            ),
            headers={"Retry-After": "1"},
        )


@dataclass(slots=True)
class WaitHistogram:
    """How long requests waited for a slot (not cumulative; +Inf last)."""

    counts: list[int] = field(default_factory=lambda: [0] * (len(WAIT_BUCKETS) + 1))
    total: float = 0.0
    count: int = 0

    def observe(self, seconds: float) -> None:
        seconds = max(0.0, seconds)
        for i, bound in enumerate(WAIT_BUCKETS):
            if seconds <= bound:
                self.counts[i] += 1
                break
        else:
            self.counts[-1] += 1
        self.total += seconds
        self.count += 1

    def cumulative(self) -> list[tuple[float, int]]:
        out: list[tuple[float, int]] = []
        running = 0
        for bound, n in zip((*WAIT_BUCKETS, float("inf")), self.counts, strict=True):
            running += n
            out.append((bound, running))
        return out


# Queue levels: a waiter at a lower level is served first; FIFO within one.
LATENCY_LEVEL = 1
DEFAULT_LEVEL = 2
BULK_LEVEL = 3
LEVELS = (LATENCY_LEVEL, DEFAULT_LEVEL, BULK_LEVEL)
_MODE_LEVELS = {
    "latency": LATENCY_LEVEL,
    "throughput": BULK_LEVEL,
    "cheapest": BULK_LEVEL,
}


def queue_level(mode: str | None) -> int:
    """The queue level a routing mode waits at: latency first; balanced,
    chaos and requests without a key next; throughput and cheapest last."""
    return _MODE_LEVELS.get(mode or "", DEFAULT_LEVEL)


class QueueTimeout(Exception):  # noqa: N818  (a condition, not an error kind)
    """No slot came free while the request could wait."""


class SlotGate:
    """At most *limit* holders at once; the rest wait in three queues.

    A released slot goes to the oldest waiter at the lowest level. Level 3
    (throughput and cheapest) never takes the last free slot of a gate whose
    limit is 2 or more, so a latency or balanced request arriving next still
    finds one. Nothing that holds a slot is pre-empted.
    """

    __slots__ = ("_waiters", "in_use", "limit", "refused", "waits")

    def __init__(self, limit: int) -> None:
        self.limit = max(1, limit)
        self.in_use = 0
        self._waiters: tuple[deque[asyncio.Future[None]], ...] = tuple(
            deque() for _ in LEVELS
        )
        self.waits = WaitHistogram()
        self.refused = 0

    @property
    def queued(self) -> int:
        return sum(self.queued_at(level) for level in LEVELS)

    def queued_at(self, level: int) -> int:
        return sum(1 for w in self._waiters[level - 1] if not w.done())

    def _may_take(self, level: int) -> bool:
        free = self.limit - self.in_use
        if level >= BULK_LEVEL and self.limit >= 2:
            return free >= 2
        return free >= 1

    def has_room(self, level: int = DEFAULT_LEVEL) -> bool:
        """A request at *level* arriving now would get a slot without waiting."""
        free = self.limit - self.in_use
        if free <= 0:
            return False
        w = self._waiters
        if level <= LATENCY_LEVEL:
            return not w[0]
        if level == DEFAULT_LEVEL:
            return not (w[0] or w[1])
        if free < 2 and self.limit >= 2:
            return False
        return not (w[0] or w[1] or w[2])

    def try_acquire(self, level: int = DEFAULT_LEVEL) -> bool:
        if self.has_room(level):
            self.in_use += 1
            return True
        return False

    async def acquire(
        self, deadline: float | None, level: int = DEFAULT_LEVEL
    ) -> float:
        """Take a slot, waiting in line at *level* when there is none.

        Returns the seconds waited; raises :class:`QueueTimeout` once the
        wait passes ``upstream_queue_timeout_seconds`` or *deadline*. Returns
        without suspending when a slot is free for *level*.
        """
        level = min(max(level, LEVELS[0]), LEVELS[-1])
        if self.try_acquire(level):
            self.waits.observe(0.0)
            return 0.0
        wait = max(0.0, settings.upstream_queue_timeout_seconds)
        if deadline is not None:
            wait = min(wait, max(0.0, deadline - time.monotonic()))
        if wait <= 0:
            self.refused += 1
            raise QueueTimeout
        fut: asyncio.Future[None] = asyncio.get_running_loop().create_future()
        self._waiters[level - 1].append(fut)
        started = time.monotonic()
        try:
            async with asyncio.timeout(wait):
                await fut
        except TimeoutError:
            if fut.done() and not fut.cancelled():
                # Handed a slot just as the wait ran out: take it.
                waited = time.monotonic() - started
                self.waits.observe(waited)
                return waited
            self._forget(fut, level)
            self.refused += 1
            raise QueueTimeout from None
        except BaseException:
            if fut.done() and not fut.cancelled():
                self.release()  # handed a slot, then cancelled: pass it on
            else:
                self._forget(fut, level)
            raise
        waited = time.monotonic() - started
        self.waits.observe(waited)
        return waited

    def _forget(self, fut: asyncio.Future[None], level: int) -> None:
        fut.cancel()
        with contextlib.suppress(ValueError):
            self._waiters[level - 1].remove(fut)
        # Leaving may unblock a waiter behind it (a level 3 behind nothing).
        self._dispatch()

    def release(self) -> None:
        """Free a slot and hand out what the waiters may take."""
        if self.in_use > 0:
            self.in_use -= 1
        self._dispatch()

    def _dispatch(self) -> None:
        for level in LEVELS:
            queue = self._waiters[level - 1]
            while queue:
                if queue[0].done():
                    queue.popleft()
                    continue
                if not self._may_take(level):
                    return  # higher levels wait their turn; none skip ahead
                self.in_use += 1
                queue.popleft().set_result(None)
            # This level is empty: the next may go.

    def resize(self, limit: int) -> None:
        """Change the limit; waiters get slots that a larger limit frees."""
        self.limit = max(1, limit)
        self._dispatch()


class CapacityLimits:
    """The gates of the providers that have a limit, by provider id."""

    def __init__(self) -> None:
        self._gates: dict[int, SlotGate] = {}

    def configure(self, limits: dict[int, int | None]) -> None:
        """Apply every provider's ``max_concurrent`` (None: unlimited).

        A gate whose provider loses its limit is dropped; requests holding
        its slots release into the old gate harmlessly.
        """
        for pid in list(self._gates):
            if not limits.get(pid):
                dropped = self._gates.pop(pid)
                dropped.resize(dropped.in_use + dropped.queued + 2)
        for pid, limit in limits.items():
            if not limit or limit <= 0:
                continue
            gate = self._gates.get(pid)
            if gate is None:
                self._gates[pid] = SlotGate(limit)
            elif gate.limit != limit:
                gate.resize(limit)

    def gate(self, provider_id: int) -> SlotGate | None:
        return self._gates.get(provider_id)

    def has_room(self, provider_id: int, level: int = DEFAULT_LEVEL) -> bool:
        gate = self._gates.get(provider_id)
        return gate is None or gate.has_room(level)

    def limit(self, provider_id: int) -> int | None:
        gate = self._gates.get(provider_id)
        return gate.limit if gate is not None else None

    def any_limited(self) -> bool:
        return bool(self._gates)

    def gates(self) -> dict[int, SlotGate]:
        return dict(self._gates)
