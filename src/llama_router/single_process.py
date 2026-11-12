"""One router process: how busy its event loop is, and a guard against two.

The router keeps a lot of state in memory (see :data:`PER_PROCESS_STATE`),
so it runs as one process; docs/configuration.md, "One router process",
says why and when to revisit that. Two things here back the decision up:

- :class:`LoopLag` measures how late the event loop wakes a task that
  sleeps a fixed interval: the time any request may wait behind others.
  Exported as ``llama_router_event_loop_lag_seconds``.
- :class:`ProcessLease` keeps a lease row with a heartbeat in the database.
  A second process started on the same database finds a fresh lease and
  logs an error naming what the two would not share. It only warns.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import socket
import time
import uuid
from collections.abc import Awaitable, Callable
from typing import Any

from .database import Database

logger = logging.getLogger(__name__)

# How often the lag probe wakes up.
LAG_INTERVAL_S = 0.5
# Histogram buckets (seconds): from unnoticeable to a stalled loop.
LAG_BUCKETS = (0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0)

LEASE_KEY = "router_process_lease"
HEARTBEAT_S = 30.0
# A lease not renewed for this long belongs to a process that is gone.
LEASE_STALE_S = 3 * HEARTBEAT_S

# What each process keeps for itself and two processes would not share.
PER_PROCESS_STATE = (
    "the API key cache (300 s)",
    "daily spend totals and reservations",
    "in-flight request counts",
    "session affinity",
    "per-key concurrency and stream limits",
    "429 cooldowns",
    "the dashboard login throttle",
    "worker hub sockets",
)


class LoopLag:
    """A histogram of how late the event loop runs a task sleeping
    *interval* seconds."""

    def __init__(
        self,
        interval: float = LAG_INTERVAL_S,
        buckets: tuple[float, ...] = LAG_BUCKETS,
        *,
        clock: Callable[[], float] = time.perf_counter,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        self.interval = interval
        self.buckets = buckets
        self._clock = clock
        self._sleep = sleep
        # Observations in each bucket (not cumulative); the last is +Inf.
        self._counts = [0] * (len(buckets) + 1)
        self.sum = 0.0
        self.count = 0
        self._task: asyncio.Task[None] | None = None

    def observe(self, seconds: float) -> None:
        seconds = max(0.0, seconds)
        for i, bound in enumerate(self.buckets):
            if seconds <= bound:
                self._counts[i] += 1
                break
        else:
            self._counts[-1] += 1
        self.sum += seconds
        self.count += 1

    def cumulative(self) -> list[tuple[float, int]]:
        """(upper bound, observations at or below it), +Inf last."""
        out: list[tuple[float, int]] = []
        running = 0
        for bound, n in zip((*self.buckets, float("inf")), self._counts, strict=True):
            running += n
            out.append((bound, running))
        return out

    async def tick(self) -> float:
        """Sleep one interval and record how late the wake-up came."""
        started = self._clock()
        await self._sleep(self.interval)
        lag = self._clock() - started - self.interval
        self.observe(lag)
        return max(0.0, lag)

    async def run(self) -> None:
        while True:
            await self.tick()

    def start(self) -> None:
        if self._task is None:
            self._task = asyncio.create_task(self.run(), name="event-loop-lag")

    async def stop(self) -> None:
        task, self._task = self._task, None
        if task is not None:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task


LOOP_LAG = LoopLag()


class ProcessLease:
    """A heartbeat in ``app_settings`` saying which process uses the
    database. Finding another live process's lease logs an error once."""

    def __init__(
        self,
        db: Database,
        *,
        clock: Callable[[], float] = time.time,
        heartbeat_s: float = HEARTBEAT_S,
        host: str | None = None,
        pid: int | None = None,
    ) -> None:
        self._db = db
        self._clock = clock
        self._heartbeat_s = heartbeat_s
        self.id = uuid.uuid4().hex
        self.host = host if host is not None else socket.gethostname()
        self.pid = pid if pid is not None else os.getpid()
        self.warned = False
        self._task: asyncio.Task[None] | None = None

    def _mine(self, now: float) -> str:
        return json.dumps(
            {"id": self.id, "host": self.host, "pid": self.pid, "at": now}
        )

    async def _other(self, now: float) -> dict[str, Any] | None:
        """Another live process's lease, if the stored one is one."""
        raw = await self._db.get_app_setting(LEASE_KEY)
        if not raw:
            return None
        try:
            lease = json.loads(raw)
        except ValueError:
            return None
        if not isinstance(lease, dict) or lease.get("id") == self.id:
            return None
        at = lease.get("at")
        if not isinstance(at, (int, float)) or now - at >= LEASE_STALE_S:
            return None
        if lease.get("host") == self.host and lease.get("pid") == self.pid:
            # This process's previous run (a restarted container keeps its
            # hostname and PID 1), not a second process.
            return None
        return lease

    async def beat(self) -> None:
        """Check for another process, then renew this one's lease."""
        now = self._clock()
        other = await self._other(now)
        if other is not None and not self.warned:
            self.warned = True
            logger.error(
                "Another llama-router process (host %s, pid %s) is using this "
                "database. Run one router process per database: each process "
                "keeps its own %s, so limits, caps and affinity would not hold "
                "across them. See 'One router process' in docs/configuration.md.",
                other.get("host"),
                other.get("pid"),
                ", ".join(PER_PROCESS_STATE),
            )
        await self._db.set_app_setting(LEASE_KEY, self._mine(now))

    async def _run(self) -> None:
        while True:
            await asyncio.sleep(self._heartbeat_s)
            try:
                await self.beat()
            except Exception:
                logger.warning(
                    "Could not renew the router process lease", exc_info=True
                )

    async def start(self) -> None:
        try:
            await self.beat()
        except Exception:
            logger.warning("Could not take the router process lease", exc_info=True)
        if self._task is None:
            self._task = asyncio.create_task(self._run(), name="process-lease")

    async def stop(self) -> None:
        """Stop renewing and give the lease up, so a restart is not taken
        for a second process."""
        task, self._task = self._task, None
        if task is not None:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
        with contextlib.suppress(Exception):
            now = self._clock()
            if await self._other(now) is None:
                await self._db.set_app_setting(LEASE_KEY, "")
