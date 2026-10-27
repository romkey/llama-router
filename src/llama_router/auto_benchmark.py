"""Benchmark models automatically while their providers are idle.

The scheduler keeps every model's benchmarks current without anyone pressing
a button, and without getting in the way of real traffic:

- A provider is benchmarked only after it has served no request for the
  idle threshold (five minutes by default). By then Ollama has usually
  unloaded its models (default keep_alive is five minutes), so a benchmark
  measures a real cold start and does not evict a model someone is using.
- Each provider runs one benchmark at a time, and only a few providers run
  at once.
- A real request to a provider cancels its running benchmark at once; the
  benchmark is retried the next time the provider is idle.
- Missing benchmarks run first, then the oldest past the maximum age.
- A benchmark that fails is not retried for a day.

Results are stored like manual benchmarks, marked ``source = "auto"``, and
the routing table picks them up through the database change notification.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import deque
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any

from . import decision_service
from . import decisions as d
from .config import settings
from .database import Database
from .endpoints import DECIDE, DECISION_PATHS
from .models import ProviderStatus
from .provider_manager import ProviderManager
from .routing_table import DECISION_UPSTREAM, decides_natively

logger = logging.getLogger(__name__)

SETTING_ENABLED = "auto_benchmark_enabled"
SETTING_IDLE_SECONDS = "auto_benchmark_idle_seconds"
SETTING_MAX_AGE_DAYS = "auto_benchmark_max_age_days"

# A failed benchmark is not retried for this long.
FAILURE_BACKOFF_S = 24 * 3600.0
# Results kept for the dashboard.
_RECENT = 20

# Asked after each benchmark: learns whether the provider returns logprobs
# (so routing knows if it can answer decisions) and seeds decision latency.
DECISION_CHECK = d.DecisionRequest(
    model="check",
    state="The sky in the photo is a clear, bright blue.",
    questions={
        "colour": {
            "type": "choice",
            "instructions": "What colour is the sky?",
            "criteria": {"blue": "Blue", "green": "Green", "red": "Red"},
        }
    },
)


@dataclass(slots=True)
class AutoBenchmarkConfig:
    enabled: bool
    idle_seconds: int
    max_age_days: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True, frozen=True)
class Job:
    provider_id: int
    provider_name: str
    model: str
    api: str
    reason: str  # "missing" or "stale"
    age_days: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _age_days(created_at: datetime | None, now: datetime) -> float | None:
    if created_at is None:
        return None
    if created_at.tzinfo is None:
        created_at = created_at.replace(tzinfo=timezone.utc)
    return max(0.0, (now - created_at).total_seconds() / 86400)


def _parse_bool(value: str | None, default: bool) -> bool:
    if value is None:
        return default
    return value.strip().lower() in ("1", "true", "yes", "on")


class AutoBenchmarker:
    def __init__(
        self,
        db: Database,
        pm: ProviderManager,
        *,
        clock: Callable[[], float] = time.monotonic,
        wall: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
        check_seconds: float | None = None,
        concurrency: int | None = None,
    ) -> None:
        self._db = db
        self._pm = pm
        self._clock = clock
        self._wall = wall
        self._check_seconds = (
            settings.auto_benchmark_check_seconds
            if check_seconds is None
            else check_seconds
        )
        self._sem = asyncio.Semaphore(
            max(1, concurrency or settings.auto_benchmark_concurrency)
        )
        self._config: AutoBenchmarkConfig | None = None
        self._task: asyncio.Task | None = None
        self._wake = asyncio.Event()
        # Provider runs in progress, and the benchmark each is running now.
        self._runs: dict[int, asyncio.Task] = {}
        self._current: dict[int, tuple[Job, asyncio.Task]] = {}
        self._backoff: dict[tuple[int, str, str], float] = {}
        self._recent: deque[dict[str, Any]] = deque(maxlen=_RECENT)
        self._last_check: float | None = None
        self.completed = 0
        self.failed = 0
        self.interrupted = 0
        pm.add_activity_listener(self._on_activity)

    # --- Settings ---

    async def config(self) -> AutoBenchmarkConfig:
        if self._config is None:
            get = self._db.get_app_setting
            enabled = _parse_bool(
                await get(SETTING_ENABLED), settings.auto_benchmark_enabled
            )
            idle = await get(SETTING_IDLE_SECONDS)
            age = await get(SETTING_MAX_AGE_DAYS)
            self._config = AutoBenchmarkConfig(
                enabled=enabled,
                idle_seconds=(
                    int(idle) if idle else settings.auto_benchmark_idle_seconds
                ),
                max_age_days=(
                    float(age) if age else settings.auto_benchmark_max_age_days
                ),
            )
        return self._config

    async def set_config(
        self,
        *,
        enabled: bool | None = None,
        idle_seconds: int | None = None,
        max_age_days: float | None = None,
    ) -> AutoBenchmarkConfig:
        if idle_seconds is not None and idle_seconds < 30:
            raise ValueError("idle time must be at least 30 seconds")
        if max_age_days is not None and max_age_days <= 0:
            raise ValueError("maximum age must be more than 0 days")
        if enabled is not None:
            await self._db.set_app_setting(
                SETTING_ENABLED, "true" if enabled else "false"
            )
        if idle_seconds is not None:
            await self._db.set_app_setting(SETTING_IDLE_SECONDS, str(int(idle_seconds)))
        if max_age_days is not None:
            await self._db.set_app_setting(
                SETTING_MAX_AGE_DAYS, str(float(max_age_days))
            )
        self._config = None
        cfg = await self.config()
        if not cfg.enabled:
            self._cancel_all()
        self.wake()
        return cfg

    # --- Lifecycle ---

    async def start(self) -> None:
        if self._task is None:
            self._task = asyncio.create_task(self._loop())

    async def stop(self) -> None:
        task, self._task = self._task, None
        if task is not None:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        self._cancel_all()
        runs = list(self._runs.values())
        for t in runs:
            t.cancel()
        if runs:
            await asyncio.gather(*runs, return_exceptions=True)

    def wake(self) -> None:
        self._wake.set()

    async def _loop(self) -> None:
        while True:
            try:
                await self.run_once()
            except Exception:
                logger.exception("Auto-benchmark check failed")
            try:
                await asyncio.wait_for(self._wake.wait(), timeout=self._check_seconds)
            except asyncio.TimeoutError:
                pass
            self._wake.clear()

    # --- Planning ---

    async def plan(self) -> list[Job]:
        """Every benchmark that is due, missing first, then oldest first."""
        cfg = await self.config()
        now_wall = self._wall()
        now = self._clock()
        latest: dict[tuple[int, str, str], datetime | None] = {}
        for b in await self._db.get_all_benchmark_results():
            key = (b.provider_id, b.model_name, (b.protocol or "").lower())
            latest.setdefault(key, b.created_at)  # rows are newest first
        models_by_provider: dict[int, list] = {}
        for m in await self._db.get_all_provider_models():
            models_by_provider.setdefault(m.provider_id, []).append(m)

        missing: list[Job] = []
        stale: list[Job] = []
        for p in await self._db.list_providers():
            if p.id is None or p.status == ProviderStatus.OFFLINE:
                continue
            if not p.auto_benchmarks:
                continue  # cloud providers cost money; see Provider.auto_benchmark
            api = p.protocol.value
            for m in models_by_provider.get(p.id, []):
                key = (p.id, m.name, api)
                if self._backoff.get(key, 0.0) > now:
                    continue
                if key not in latest:
                    missing.append(Job(p.id, p.name, m.name, api, "missing"))
                    continue
                age = _age_days(latest[key], now_wall)
                if age is not None and age >= cfg.max_age_days:
                    stale.append(Job(p.id, p.name, m.name, api, "stale", age))
        stale.sort(key=lambda j: -(j.age_days or 0.0))
        return missing + stale

    def _is_idle(self, provider_id: int, cfg: AutoBenchmarkConfig) -> bool:
        return (
            self._pm.active_requests(provider_id) == 0
            and self._pm.idle_seconds(provider_id, self._clock()) >= cfg.idle_seconds
        )

    async def run_once(self) -> None:
        """Start benchmarking every idle provider that has work."""
        self._last_check = self._clock()
        cfg = await self.config()
        if not cfg.enabled:
            return
        by_provider: dict[int, list[Job]] = {}
        for job in await self.plan():
            by_provider.setdefault(job.provider_id, []).append(job)
        for pid, jobs in by_provider.items():
            if pid in self._runs or not self._is_idle(pid, cfg):
                continue
            self._runs[pid] = asyncio.create_task(self._run_provider(pid, jobs))

    # --- Running ---

    async def _run_provider(self, pid: int, jobs: list[Job]) -> None:
        try:
            for job in jobs:
                async with self._sem:
                    cfg = await self.config()
                    if not cfg.enabled or not self._is_idle(pid, cfg):
                        return
                    if not await self._run_job(job):
                        return  # interrupted by traffic: wait for idle again
        finally:
            self._runs.pop(pid, None)

    async def _run_job(self, job: Job) -> bool:
        """Run one benchmark. False if real traffic interrupted it."""
        task = asyncio.create_task(self._benchmark(job))
        self._current[job.provider_id] = (job, task)
        started = self._clock()
        try:
            await asyncio.wait({task})
        except asyncio.CancelledError:
            task.cancel()  # the scheduler is stopping: stop the benchmark too
            raise
        finally:
            self._current.pop(job.provider_id, None)
        entry = {**job.to_dict(), "seconds": round(self._clock() - started, 2)}
        if task.cancelled():
            self.interrupted += 1
            self._recent.appendleft({**entry, "outcome": "interrupted"})
            logger.info(
                "Auto-benchmark of %s on %s interrupted by a request",
                job.model,
                job.provider_name,
            )
            return False
        exc = task.exception()
        if exc is not None:
            self.failed += 1
            self._backoff[(job.provider_id, job.model, job.api)] = (
                self._clock() + FAILURE_BACKOFF_S
            )
            self._recent.appendleft(
                {**entry, "outcome": "failed", "error": str(exc)[:300]}
            )
            logger.warning(
                "Auto-benchmark of %s on %s failed: %s",
                job.model,
                job.provider_name,
                exc,
            )
            return True
        result, decisions = task.result()
        self.completed += 1
        self._recent.appendleft(
            {
                **entry,
                "outcome": "ok",
                "startup_time_ms": result.startup_time_ms,
                "tokens_per_second": result.tokens_per_second,
                "decisions": decisions,
            }
        )
        logger.info(
            "Auto-benchmarked %s on %s (%s): %.1f tok/s",
            job.model,
            job.provider_name,
            job.api,
            result.tokens_per_second or 0.0,
        )
        return True

    async def _benchmark(self, job: Job) -> tuple[Any, str]:
        result = await self._pm.benchmark_provider(
            job.provider_id, job.model, benchmark_api=job.api, source="auto"
        )
        return result, await self.check_decisions(job.provider_id, job.model)

    async def check_decisions(self, provider_id: int, model: str) -> str:
        """Ask :data:`DECISION_CHECK` of *model*; returns the outcome.

        ``ok``, ``no_logprobs``, ``error``, or ``skipped`` for a provider
        that serves decisions natively or has no chat completions. Runs
        without request accounting, so it never cancels itself.
        """
        provider = await self._db.get_provider(provider_id)
        if provider is None or provider.lacks(DECISION_UPSTREAM):
            return "skipped"
        if any(decides_natively(provider, p) for p in DECISION_PATHS):
            return "skipped"
        backend = await self._db.get_backend_model_name(provider_id, model)
        stats = self._pm.stats
        started = self._clock()
        try:
            await decision_service.emulate(
                self._pm, provider_id, backend or model, DECISION_CHECK, acquire=False
            )
        except d.NoLogprobs:
            stats.record_logprobs(provider_id, ok=False)
            return "no_logprobs"
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # a side check never fails its benchmark
            logger.info(
                "Decision check of %s on %s failed: %s", model, provider.name, exc
            )
            return "error"
        stats.record_logprobs(provider_id, ok=True)
        stats.record_request(
            provider_id,
            model,
            ok=True,
            endpoint_class=DECIDE,
            duration_ms=(self._clock() - started) * 1000,
        )
        return "ok"

    def _on_activity(self, provider_id: int) -> None:
        """A real request started: real traffic always wins."""
        current = self._current.get(provider_id)
        if current is not None and not current[1].done():
            current[1].cancel()

    def _cancel_all(self) -> None:
        for _job, task in list(self._current.values()):
            if not task.done():
                task.cancel()

    # --- Status ---

    async def status(self) -> dict[str, Any]:
        cfg = await self.config()
        pending = await self.plan()
        return {
            "config": cfg.to_dict(),
            "running": [job.to_dict() for job, _t in self._current.values()],
            "pending": len(pending),
            "next": [j.to_dict() for j in pending[:10]],
            "recent": list(self._recent),
            "completed": self.completed,
            "failed": self.failed,
            "interrupted": self.interrupted,
            "backed_off": sum(1 for t in self._backoff.values() if t > self._clock()),
        }
