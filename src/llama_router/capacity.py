"""Per-provider concurrency limits (``providers.max_concurrent``).

A provider with a limit gets a :class:`SlotGate`: at most *limit* requests
run on it at once, and the rest wait in a FIFO queue. A provider without one
has no gate at all, so an unconfigured limit costs one dictionary lookup.

Taking a free slot never suspends: with no one waiting and a slot free,
:meth:`SlotGate.acquire` returns without awaiting anything. A request that
has to wait is bounded by ``upstream_queue_timeout_seconds`` and its request
deadline, then refused with :class:`ProviderAtCapacity` (429 and
``Retry-After``). A released slot is handed straight to the next waiter, so a
newcomer cannot overtake the queue.
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


class SlotGate:
    """At most *limit* holders at once; the rest wait in FIFO order."""

    __slots__ = ("_waiters", "in_use", "limit", "refused", "waits")

    def __init__(self, limit: int) -> None:
        self.limit = max(1, limit)
        self.in_use = 0
        self._waiters: deque[asyncio.Future[None]] = deque()
        self.waits = WaitHistogram()
        self.refused = 0

    @property
    def queued(self) -> int:
        return sum(1 for w in self._waiters if not w.done())

    def has_room(self) -> bool:
        """A request arriving now would get a slot without waiting."""
        return self.in_use < self.limit and not self._waiters

    def try_acquire(self) -> bool:
        if self.has_room():
            self.in_use += 1
            return True
        return False

    async def acquire(self, deadline: float | None, provider_id: int = 0) -> None:
        """Take a slot, waiting in line when there is none (429 on timeout).

        Returns without suspending when a slot is free and no one waits.
        """
        if self.try_acquire():
            self.waits.observe(0.0)
            return
        wait = max(0.0, settings.upstream_queue_timeout_seconds)
        if deadline is not None:
            wait = min(wait, max(0.0, deadline - time.monotonic()))
        if wait <= 0:
            self.refused += 1
            raise ProviderAtCapacity(provider_id, self.limit)
        fut: asyncio.Future[None] = asyncio.get_running_loop().create_future()
        self._waiters.append(fut)
        started = time.monotonic()
        try:
            async with asyncio.timeout(wait):
                await fut
        except TimeoutError:
            if fut.done() and not fut.cancelled():
                # Handed a slot just as the wait ran out: take it.
                self.waits.observe(time.monotonic() - started)
                return
            self._forget(fut)
            self.refused += 1
            raise ProviderAtCapacity(provider_id, self.limit) from None
        except BaseException:
            if fut.done() and not fut.cancelled():
                self.release()  # handed a slot, then cancelled: pass it on
            else:
                self._forget(fut)
            raise
        self.waits.observe(time.monotonic() - started)

    def _forget(self, fut: asyncio.Future[None]) -> None:
        fut.cancel()
        with contextlib.suppress(ValueError):
            self._waiters.remove(fut)

    def release(self) -> None:
        """Free a slot, handing it to the longest waiter if there is one."""
        while self._waiters:
            fut = self._waiters.popleft()
            if not fut.done():
                fut.set_result(None)  # the slot changes hands; in_use stays
                return
        if self.in_use > 0:
            self.in_use -= 1

    def resize(self, limit: int) -> None:
        """Change the limit; waiters get slots that a larger limit frees."""
        self.limit = max(1, limit)
        while self._waiters and self.in_use < self.limit:
            fut = self._waiters.popleft()
            if not fut.done():
                self.in_use += 1
                fut.set_result(None)


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
                dropped.resize(dropped.in_use + len(dropped._waiters) + 1)
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

    def has_room(self, provider_id: int) -> bool:
        gate = self._gates.get(provider_id)
        return gate is None or gate.has_room()

    def limit(self, provider_id: int) -> int | None:
        gate = self._gates.get(provider_id)
        return gate.limit if gate is not None else None

    def any_limited(self) -> bool:
        return bool(self._gates)

    def gates(self) -> dict[int, SlotGate]:
        return dict(self._gates)
