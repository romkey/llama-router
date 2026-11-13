"""Webhook alerts: a JSON POST to ``LLAMA_ROUTER_ALERT_WEBHOOK_URL`` when
something an operator should know about happens.

Events (``LLAMA_ROUTER_ALERT_EVENTS``, comma-separated; default all):

- ``provider_offline`` / ``provider_online``: a provider changed state.
- ``model_unavailable``: a model no online provider serves any more.
- ``worker_disconnected``; ``worker_token_expiring`` (within 7 days).
- ``quota_reached``: a user or key reached a daily limit (once a day each).
- ``key_expiring``: an API key expires within 7 days.
- ``renewal_requested``: an owner asked for a key renewal.
- ``keys_held``: the router held a user's keys.
- ``event_loop_lag``: the event loop's p99 lag over the last check passed
  50 ms.

The payload is ``{"event", "time", "text", "details"}``: a one-line text and
a small details object, never request content or secrets.

Callers only :func:`emit`, which does a ``put_nowait`` on a bounded
in-memory queue (an alert is dropped when it is full). A background task
delivers: a short timeout, 3 retries, and a 10-minute debounce per event and
subject. Failures are logged and counted, never raised. With no URL set,
:func:`emit` returns after one check.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime, timedelta
from typing import Any

import httpx

from .config import settings

logger = logging.getLogger(__name__)

EVENTS = (
    "provider_offline",
    "provider_online",
    "model_unavailable",
    "worker_disconnected",
    "worker_token_expiring",
    "quota_reached",
    "key_expiring",
    "renewal_requested",
    "keys_held",
    "event_loop_lag",
)
QUEUE_MAX = 1000
DEBOUNCE_S = 600.0
# Alerts about a state that lasts (a limit reached today, a key expiring
# this week) repeat at most daily.
DAILY_S = 86400.0
TIMEOUT_S = 5.0
RETRIES = 3
RETRY_DELAY_S = 1.0
EXPIRY_WARNING = timedelta(days=7)
LAG_THRESHOLD_S = 0.05
CHECK_INTERVAL_S = 300.0


def _enabled_events() -> frozenset[str]:
    raw = (settings.alert_events or "all").strip().lower()
    if raw in ("", "all", "*"):
        return frozenset(EVENTS)
    return frozenset(e.strip() for e in raw.split(",") if e.strip())


class Alerts:
    """The queue, the debounce memory and the delivery task."""

    def __init__(self) -> None:
        self._queue: asyncio.Queue[dict[str, Any]] | None = None
        self._task: asyncio.Task[None] | None = None
        self._checks: asyncio.Task[None] | None = None
        # (event, subject) -> when it was last sent (monotonic).
        self._sent: dict[tuple[str, str], float] = {}
        # Last provider states seen and models served, to report changes.
        self._provider_online: dict[int, bool] = {}
        self._unknown: set[int] = set()
        self._served: set[str] | None = None
        self.delivered = 0
        self.failed = 0
        self.dropped = 0
        self.clock: Callable[[], float] = time.monotonic
        self.post: Callable[[str, dict[str, Any]], Awaitable[None]] = _post

    @property
    def configured(self) -> bool:
        return bool(settings.alert_webhook_url)

    def reset(self) -> None:
        self._sent.clear()
        self._provider_online.clear()
        self._unknown.clear()
        self._served = None
        self.delivered = self.failed = self.dropped = 0
        self._queue = None

    # --- producing ---

    def emit(
        self,
        event: str,
        subject: str,
        text: str,
        details: dict[str, Any] | None = None,
        *,
        force: bool = False,
        debounce_s: float = DEBOUNCE_S,
    ) -> bool:
        """Queue an alert without waiting; False when off, not chosen,
        debounced (sent for the same event and subject within *debounce_s*)
        or dropped. *force* skips the debounce (test alerts)."""
        if not settings.alert_webhook_url:
            return False
        if event != "test" and event not in _enabled_events():
            return False
        key = (event, subject)
        now = self.clock()
        last = self._sent.get(key)
        if not force and last is not None and now - last < debounce_s:
            return False
        queue = self._ensure_queue()
        payload = {
            "event": event,
            "time": datetime.now(UTC).isoformat(timespec="seconds"),
            "text": text,
            "details": details or {},
        }
        try:
            queue.put_nowait(payload)
        except asyncio.QueueFull:
            self.dropped += 1
            return False
        self._sent[key] = now
        self._ensure_task()
        return True

    def _ensure_queue(self) -> asyncio.Queue[dict[str, Any]]:
        if self._queue is None:
            self._queue = asyncio.Queue(maxsize=QUEUE_MAX)
        return self._queue

    def _ensure_task(self) -> None:
        if self._task is None or self._task.done():
            with contextlib.suppress(RuntimeError):
                self._task = asyncio.get_running_loop().create_task(
                    self._deliver(), name="alert-delivery"
                )

    # --- delivering ---

    async def _deliver(self) -> None:
        queue = self._ensure_queue()
        while True:
            payload = await queue.get()
            try:
                await self._send(payload)
            finally:
                queue.task_done()

    async def _send(self, payload: dict[str, Any]) -> bool:
        url = settings.alert_webhook_url
        if not url:
            return False
        for attempt in range(1, RETRIES + 1):
            try:
                await self.post(url, payload)
            except Exception as exc:  # never reaches a request
                if attempt == RETRIES:
                    self.failed += 1
                    logger.warning(
                        "Alert %s not delivered after %d attempts: %s",
                        payload["event"],
                        RETRIES,
                        type(exc).__name__,
                    )
                    return False
                await asyncio.sleep(RETRY_DELAY_S * attempt)
            else:
                self.delivered += 1
                return True
        return False

    async def drain(self) -> None:
        """Wait until queued alerts are delivered (tests, shutdown)."""
        if self._queue is not None and self._task is not None:
            await self._queue.join()

    async def send_test(self) -> bool:
        """Send a test alert now, past the queue; whether it was delivered."""
        if not self.configured:
            return False
        return await self._send(
            {
                "event": "test",
                "time": datetime.now(UTC).isoformat(timespec="seconds"),
                "text": "Test alert from llama-router",
                "details": {},
            }
        )

    # --- observations ---

    def provider_status(
        self, provider_id: int, name: str, online: bool, *, was: bool | None = None
    ) -> None:
        """A provider's state; alerts on a change. *was* is the stored state
        before this write (None: never checked), so the first change after a
        restart is reported too; a provider seen for the first time sets the
        baseline."""
        if provider_id in self._unknown:
            # Its state is unknown (a worker at start-up): being marked
            # offline until it dials in is no outage, and dialling in sets
            # the baseline.
            if online:
                self._unknown.discard(provider_id)
                self._provider_online[provider_id] = online
            return
        before = self._provider_online.get(provider_id, was)
        self._provider_online[provider_id] = online
        if before is None or before == online:
            return
        event = "provider_online" if online else "provider_offline"
        self.emit(
            event,
            str(provider_id),
            f"Provider {name} is {'back online' if online else 'offline'}",
            {"provider_id": provider_id, "provider": name},
        )

    def forget_provider(self, provider_id: int) -> None:
        """Treat the provider's state as unknown: its next state (a worker
        marked offline at start-up, then online when it dials in) only sets
        the baseline."""
        self._provider_online.pop(provider_id, None)
        self._unknown.add(provider_id)

    def models_served(self, served: set[str]) -> None:
        """The models some online provider serves; alerts on each one that
        stopped being served (the first call sets the baseline)."""
        before, self._served = self._served, set(served)
        if before is None:
            return
        for model in sorted(before - served):
            self.emit(
                "model_unavailable",
                model,
                f"No online provider serves {model}",
                {"model": model},
            )

    def quota_reached(self, limit: str, who: str, subject: Any) -> None:
        day = datetime.now(UTC).date().isoformat()
        self.emit(
            "quota_reached",
            f"{limit}:{subject}:{day}",
            f"{who} reached its daily {limit.replace('_', ' ')} limit",
            {"limit": limit, "subject": str(subject)},
            # Once a day each: the subject carries the day.
            debounce_s=DAILY_S,
        )

    # --- periodic checks ---

    def start_checks(self, check: Callable[[], Awaitable[None]]) -> None:
        if not self.configured or self._checks is not None:
            return

        async def run() -> None:
            while True:
                try:
                    await check()
                except Exception:
                    logger.exception("Alert checks failed")
                await asyncio.sleep(CHECK_INTERVAL_S)

        self._checks = asyncio.create_task(run(), name="alert-checks")

    async def stop(self) -> None:
        for task in (self._checks, self._task):
            if task is not None:
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task
        self._checks = self._task = None


async def _post(url: str, payload: dict[str, Any]) -> None:
    async with httpx.AsyncClient(timeout=TIMEOUT_S) as client:
        resp = await client.post(url, json=payload)
        resp.raise_for_status()


alerts = Alerts()


class LagWatch:
    """p99 of the event loop lag observed since the previous check."""

    def __init__(self) -> None:
        self._last: list[tuple[float, int]] | None = None

    def p99_since_last(self, cumulative: list[tuple[float, int]]) -> float | None:
        prev, self._last = self._last, cumulative
        if prev is None:
            return None
        deltas = [(b, n - p) for (b, n), (_, p) in zip(cumulative, prev, strict=True)]
        total = deltas[-1][1]
        if total <= 0:
            return None
        target = total * 0.99
        for bound, n in deltas:
            if n >= target:
                return bound
        return deltas[-1][0]


def _label(key: dict[str, Any]) -> str:
    return f" ({key['label']})" if key.get("label") else ""


async def periodic_checks(db: Any, lag: Any, watch: LagWatch) -> None:
    """Expiring worker tokens and API keys, and event loop lag."""
    now = datetime.now(UTC)
    soon = now + EXPIRY_WARNING
    for w in await db.list_workers():
        exp = w.get("token_expires_at")
        if exp is None:
            continue
        when = datetime.fromtimestamp(int(exp), UTC)
        if now < when <= soon:
            alerts.emit(
                "worker_token_expiring",
                str(w["id"]),
                f"Worker {w['name']}'s token expires {when:%Y-%m-%d}",
                {
                    "worker_id": w["id"],
                    "worker": w["name"],
                    "expires_at": when.isoformat(),
                },
                debounce_s=DAILY_S,
            )
    for k in await db.list_api_keys():
        when = k.get("expires_at")
        if when is not None and now < when <= soon:
            alerts.emit(
                "key_expiring",
                str(k["id"]),
                f"API key {k['id']}{_label(k)} expires {when:%Y-%m-%d}",
                {
                    "key_id": k["id"],
                    "label": k.get("label"),
                    "owner": k.get("owner_username"),
                    "expires_at": when.isoformat(),
                },
                debounce_s=DAILY_S,
            )
    p99 = watch.p99_since_last(lag.cumulative())
    if p99 is not None and p99 > LAG_THRESHOLD_S:
        alerts.emit(
            "event_loop_lag",
            "loop",
            f"Event loop p99 lag {p99 * 1000:.0f} ms over the last check",
            {"p99_ms": round(p99 * 1000, 1)},
        )
