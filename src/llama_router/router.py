"""Routes requests to the best available provider for a given model.

Routing decisions come from a precomputed :class:`RoutingTable`. The table is
rebuilt in the background whenever something that affects routing changes:
the database reports a write (providers, status, addresses, models,
benchmarks, fallbacks, probe results), the hot-model set changes, or live
request and round-trip samples arrive. Rebuilds are coalesced, so a burst of
changes produces one rebuild. ``route()`` itself only reads the table and the
in-memory queue depth; it never touches the database once the table exists.
"""

from __future__ import annotations

import asyncio
import logging
import random
import time
from dataclasses import dataclass, field

from .database import ROUTING_CHANGES, Database
from .models import Provider
from .provider_manager import ProviderManager
from .routing_table import (
    DEFAULT_MODE,
    MODES,
    Candidate,
    RoutingInputs,
    RoutingTable,
    build_table,
    load_inputs,
    normalize_endpoint,
)

logger = logging.getLogger(__name__)

ROUTING_MODES = frozenset({*MODES, "chaos"})

# Wait this long after the first change before rebuilding, to batch a burst.
REBUILD_DEBOUNCE_S = 0.25
# Never rebuild more often than this.
REBUILD_MIN_INTERVAL_S = 1.0
# Reasons kept for the dashboard.
_MAX_REASONS = 20


class RouteResult:
    """Wraps a routing decision, including which model was actually resolved."""

    __slots__ = ("provider", "resolved_model", "backend_model")

    def __init__(
        self,
        provider: Provider,
        resolved_model: str,
        backend_model: str | None = None,
    ):
        self.provider = provider
        self.resolved_model = resolved_model
        # The name the chosen backend knows the model by.
        self.backend_model = backend_model or resolved_model


@dataclass(slots=True)
class RoutingPreferences:
    mode: str = DEFAULT_MODE  # "latency", "throughput", "balanced" or "chaos"
    allow_fallback: bool = True
    pinned_providers: dict[str, int] = field(default_factory=dict)


class Router:
    def __init__(
        self,
        db: Database,
        provider_manager: ProviderManager,
        *,
        debounce_s: float = REBUILD_DEBOUNCE_S,
        min_interval_s: float = REBUILD_MIN_INTERVAL_S,
    ):
        self._db = db
        self._pm = provider_manager
        self._debounce_s = debounce_s
        self._min_interval_s = min_interval_s
        self._table: RoutingTable | None = None
        self._inputs: RoutingInputs | None = None
        self._need_reload = True
        self._need_rescore = True
        self._pending_reasons: list[str] = []
        self._version = 0
        self._wake = asyncio.Event()
        self._lock = asyncio.Lock()
        self._task: asyncio.Task | None = None
        db.add_change_listener(self._on_db_change)
        provider_manager.stats.add_listener(self._on_stats_change)
        provider_manager.attach_router(self)

    # --- Table maintenance ---

    @property
    def table(self) -> RoutingTable | None:
        return self._table

    def mark_dirty(self, reason: str, *, reload: bool = True) -> None:
        """Schedule a rebuild. *reload* re-reads the database first.

        Cheap and synchronous; callers never wait for the rebuild.
        """
        if reload:
            self._need_reload = True
        self._need_rescore = True
        if len(self._pending_reasons) < _MAX_REASONS:
            self._pending_reasons.append(reason)
        self._wake.set()

    def _on_db_change(self, reason: str) -> None:
        if reason in ROUTING_CHANGES:
            self.mark_dirty(reason)

    def _on_stats_change(self, reason: str) -> None:
        # Live samples only change scores; the database inputs are unchanged.
        self.mark_dirty(reason, reload=False)

    async def rebuild(self) -> RoutingTable:
        """Rebuild the table now and swap it in."""
        async with self._lock:
            reasons = tuple(dict.fromkeys(self._pending_reasons)) or ("initial",)
            self._pending_reasons = []
            reload = self._need_reload or self._inputs is None
            # Clear flags before reading so a change during the build marks
            # the table dirty again.
            self._need_reload = False
            self._need_rescore = False
            started = time.perf_counter()
            try:
                if reload:
                    self._inputs = await load_inputs(self._db)
                assert self._inputs is not None
                self._version += 1
                table = build_table(
                    self._inputs,
                    hot_models=self._pm.hot_model_names,
                    stats=self._pm.stats,
                    version=self._version,
                    reasons=reasons,
                )
            except Exception:
                if reload:
                    self._need_reload = True
                self._need_rescore = True
                raise
            self._table = table
            elapsed = (time.perf_counter() - started) * 1000
            log = logger.info if reload else logger.debug
            log(
                "Routing table v%d built in %.1f ms (%s; %s)",
                table.version,
                elapsed,
                "reloaded" if reload else "rescored",
                ", ".join(reasons),
            )
            return table

    async def start(self) -> None:
        """Build the first table and keep it current in the background."""
        await self.rebuild()
        if self._task is None:
            self._task = asyncio.create_task(self._run())

    async def stop(self) -> None:
        task, self._task = self._task, None
        if task is not None:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        self._db.remove_change_listener(self._on_db_change)

    async def _run(self) -> None:
        while True:
            await self._wake.wait()
            await asyncio.sleep(self._debounce_s)
            self._wake.clear()
            try:
                await self.rebuild()
            except Exception:
                logger.exception("Routing table rebuild failed")
            await asyncio.sleep(self._min_interval_s)

    async def _current_table(self) -> RoutingTable:
        table = self._table
        if table is None or (
            self._task is None and (self._need_reload or self._need_rescore)
        ):
            # No background task (tests, or before start): build inline.
            table = await self.rebuild()
        return table

    # --- Routing ---

    async def route(
        self,
        model_name: str,
        protocol: str | None = None,
        preferences: RoutingPreferences | None = None,
        endpoint: str | None = None,
    ) -> RouteResult | None:
        """Pick the best provider for the requested model, following fallbacks.

        Walks the fallback chain if the requested model has no available
        provider. *endpoint* is the request path; providers whose last probe
        found it missing are skipped while another candidate remains.
        """
        table = await self._current_table()
        return self.route_with(table, model_name, protocol, preferences, endpoint)

    def route_with(
        self,
        table: RoutingTable,
        model_name: str,
        protocol: str | None = None,
        preferences: RoutingPreferences | None = None,
        endpoint: str | None = None,
    ) -> RouteResult | None:
        """Route against *table*. Synchronous: no I/O, no awaits."""
        pins = preferences.pinned_providers if preferences else None
        mode = preferences.mode if preferences else DEFAULT_MODE
        if mode == "chaos" and not (pins and model_name in pins):
            return self._route_chaos(table, model_name, protocol)
        if mode not in MODES:
            mode = DEFAULT_MODE

        if preferences and not preferences.allow_fallback:
            chain: tuple[str, ...] = (model_name,)
        else:
            chain = table.chain(model_name)
        path = normalize_endpoint(endpoint)
        active = self._pm.active_requests

        for candidate_model in chain:
            if pins:
                pid = pins.get(candidate_model)
                if pid is not None:
                    pinned = self._pinned(table, candidate_model, protocol, pid)
                    if pinned is not None:
                        if candidate_model != model_name:
                            logger.info(
                                "Model %s unavailable; fell back to pinned model %s",
                                model_name,
                                candidate_model,
                            )
                        return RouteResult(
                            pinned.provider, candidate_model, pinned.backend_model
                        )
            ranked = table.ranked.get((candidate_model, protocol, mode))
            if not ranked:
                continue
            chosen = _pick(ranked, mode, path, active)
            if candidate_model != model_name:
                logger.info(
                    "Model %s unavailable; fell back to %s",
                    model_name,
                    candidate_model,
                )
            logger.debug(
                "Routing model %s (%s, %s) to provider %s",
                candidate_model,
                protocol or "any",
                mode,
                chosen.provider.name,
            )
            return RouteResult(chosen.provider, candidate_model, chosen.backend_model)
        return None

    @staticmethod
    def _pinned(
        table: RoutingTable, model: str, protocol: str | None, provider_id: int
    ) -> Candidate | None:
        for c in table.all_candidates.get((model, protocol), ()):
            if c.provider.id == provider_id:
                logger.debug(
                    "Pinned routing model %s (%s) -> provider %s",
                    model,
                    protocol or "any",
                    c.provider.name,
                )
                return c
        logger.warning(
            "Pinned route unavailable for model %s (%s) provider_id=%d",
            model,
            protocol or "any",
            provider_id,
        )
        return None

    @staticmethod
    def _route_chaos(
        table: RoutingTable, requested_model: str, protocol: str | None
    ) -> RouteResult | None:
        """Chaos mode: pick a random online provider and random eligible model."""
        entries = table.chaos.get(protocol)
        if not entries:
            return None
        provider, eligible = random.choice(entries)
        resolved_model = random.choice(eligible)
        assert provider.id is not None
        logger.info(
            "Chaos routing requested=%s protocol=%s -> provider=%s model=%s",
            requested_model,
            protocol or "any",
            provider.name,
            resolved_model,
        )
        return RouteResult(
            provider, resolved_model, table.backend_name(provider.id, resolved_model)
        )


def _pick(
    ranked: tuple[Candidate, ...], mode: str, path: str | None, active
) -> Candidate:
    """Lowest static score plus queue cost; *ranked* is sorted by static score."""
    if path is not None:
        serving = [c for c in ranked if path not in c.missing_endpoints]
        if serving:
            ranked = tuple(serving)
    first = ranked[0]
    if len(ranked) == 1:
        return first
    first_active = active(first.provider_id)
    if first_active == 0:
        # Queue costs are never negative, so nothing can beat an idle leader.
        return first
    best = first
    best_score = first.scores[mode] + first_active * first.queue_weight[mode]
    for c in ranked[1:]:
        s = c.scores[mode]
        if s >= best_score:
            break  # sorted: every later static score is at least this high
        s += active(c.provider_id) * c.queue_weight[mode]
        if s < best_score:
            best, best_score = c, s
    return best
