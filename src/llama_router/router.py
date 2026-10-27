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
from typing import Any

from .database import ROUTING_CHANGES, Database
from .models import Provider, ProviderStatus
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

# Every proxied endpoint and the protocol its handler routes with. Ollama
# native routes need an Ollama-capable provider; OpenAI-style routes take any.
ENDPOINT_PROTOCOLS: dict[str, str | None] = {
    "/api/chat": "ollama",
    "/api/generate": "ollama",
    "/api/embed": "ollama",
    "/api/embeddings": "ollama",
    "/v1/chat/completions": None,
    "/v1/completions": None,
    "/v1/responses": None,
    "/v1/embeddings": None,
    "/v1/images/generations": None,
    "/v1/images/edits": None,
    "/v1/audio/speech": None,
    "/v1/audio/transcriptions": None,
}


def protocol_for_endpoint(endpoint: str | None) -> str | None:
    """The protocol a request to *endpoint* is routed with."""
    path = normalize_endpoint(endpoint)
    return ENDPOINT_PROTOCOLS.get(path or "")


class RouteResult:
    """Wraps a routing decision, including which model was actually resolved."""

    __slots__ = ("provider", "resolved_model", "backend_model", "model_key")

    def __init__(
        self,
        provider: Provider,
        resolved_model: str,
        backend_model: str | None = None,
        model_key: str | None = None,
    ):
        self.provider = provider
        self.resolved_model = resolved_model
        # The name the chosen backend knows the model by.
        self.backend_model = backend_model or resolved_model
        # The model's discovered name, which live stats are keyed by. Differs
        # from resolved_model when the client asked by raw (backend) name.
        self.model_key = model_key or resolved_model


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
                            pinned.provider,
                            candidate_model,
                            pinned.backend_model,
                            pinned.model,
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
            return RouteResult(
                chosen.provider, candidate_model, chosen.backend_model, chosen.model
            )
        return None

    async def explain(
        self,
        model_name: str,
        protocol: str | None = None,
        preferences: RoutingPreferences | None = None,
        endpoint: str | None = None,
    ) -> dict[str, Any]:
        """How a request would be routed right now, and why.

        Walks the same fallback chain, pins, circuit breaker, endpoint filter
        and scoring as :meth:`route`, against the same table and live queue
        depths, and reports every provider considered. The decision itself is
        taken from :meth:`route_with`, so the explanation cannot drift from
        what a real request would get.
        """
        table = await self._current_table()
        decision = self.route_with(table, model_name, protocol, preferences, endpoint)
        return self.explain_with(
            table, decision, model_name, protocol, preferences, endpoint
        )

    def explain_with(
        self,
        table: RoutingTable,
        decision: RouteResult | None,
        model_name: str,
        protocol: str | None,
        preferences: RoutingPreferences | None,
        endpoint: str | None,
    ) -> dict[str, Any]:
        pins = dict(preferences.pinned_providers) if preferences else {}
        mode = preferences.mode if preferences else DEFAULT_MODE
        allow_fallback = preferences.allow_fallback if preferences else True
        path = normalize_endpoint(endpoint)
        out: dict[str, Any] = {
            "request": {
                "model": model_name,
                "mode": mode,
                "endpoint": path,
                "protocol": protocol or "any",
                "allow_fallback": allow_fallback,
                "pins": pins,
            },
            "table_version": table.version,
            "table_built_at": table.built_at,
            "chosen": None,
            "steps": [],
        }
        if decision is not None:
            out["chosen"] = {
                "provider_id": decision.provider.id,
                "provider": decision.provider.name,
                "model": decision.resolved_model,
                "backend_model": decision.backend_model,
                "fell_back": decision.resolved_model != model_name,
            }

        if mode == "chaos" and model_name not in pins:
            out["chaos_pool"] = [
                {"provider_id": p.id, "provider": p.name, "models": list(models)}
                for p, models in table.chaos.get(protocol, ())
            ]
            if not out["chaos_pool"]:
                out["summary"] = (
                    "No provider can serve this request: chaos mode has no "
                    "online provider with an eligible model."
                )
            else:
                sample = out["chosen"] or {}
                out["summary"] = (
                    "Chaos mode picks a random provider from the pool and a random "
                    "model from it, so each request can land somewhere different "
                    f"(this time: {sample.get('provider')}, {sample.get('model')})."
                )
            return out
        if mode not in MODES:
            mode = DEFAULT_MODE

        chain = table.chain(model_name) if allow_fallback else (model_name,)
        out["fallback_chain"] = list(chain)
        active = self._pm.active_requests
        for candidate_model in chain:
            chosen_id = (
                decision.provider.id
                if decision is not None and decision.resolved_model == candidate_model
                else None
            )
            step = self._explain_step(
                table, candidate_model, protocol, mode, path, pins, active, chosen_id
            )
            out["steps"].append(step)
            if decision is not None and decision.resolved_model == candidate_model:
                break
        out["summary"] = _summary(out, model_name)
        return out

    def _explain_step(
        self,
        table: RoutingTable,
        model: str,
        protocol: str | None,
        mode: str,
        path: str | None,
        pins: dict[str, int],
        active,
        chosen_id: int | None = None,
    ) -> dict[str, Any]:
        step: dict[str, Any] = {"model": model, "pinned_provider_id": None}
        rows: list[dict[str, Any]] = []
        all_cands = table.all_candidates.get((model, protocol), ())
        ranked = table.ranked.get((model, protocol, mode), ())
        usable_ids = {c.provider_id for c in ranked}
        serving = [c for c in ranked if path is None or path not in c.missing_endpoints]
        serving_ids = {c.provider_id for c in (serving or ranked)}
        pid = pins.get(model)
        pinned = None
        if pid is not None:
            step["pinned_provider_id"] = pid
            pinned = next((c for c in all_cands if c.provider_id == pid), None)

        for c in all_cands:
            q = active(c.provider_id)
            row = c.to_dict()
            row.update(
                {
                    "static_score": round(c.scores[mode], 2),
                    "active_requests": q,
                    "queue_cost": round(q * c.queue_weight[mode], 2),
                    "total_score": round(c.scores[mode] + q * c.queue_weight[mode], 2),
                    "excluded": None,
                }
            )
            if pinned is not None and c is not pinned:
                row["excluded"] = "another provider is pinned for this model"
            elif pinned is None and c.provider_id not in usable_ids:
                row["excluded"] = (
                    "recent failures (circuit open); other providers are healthy"
                )
            elif (
                pinned is None and c.provider_id not in serving_ids and path is not None
            ):
                row["excluded"] = f"last probe found {path} missing"
            rows.append(row)

        rows.extend(
            self._unroutable(model, protocol, {c.provider_id for c in all_cands})
        )
        eligible = [r for r in rows if r["excluded"] is None]
        # The real decision ranks first, so ties read the same way it broke them.
        eligible.sort(key=lambda r: (r["provider_id"] != chosen_id, r["total_score"]))
        if pinned is not None:
            step["outcome"] = f"pinned to {pinned.provider.name}"
        elif eligible:
            step["outcome"] = f"lowest {mode} score: {eligible[0]['provider']}"
        else:
            step["outcome"] = "no provider can serve this model"
            if pid is not None:
                step["outcome"] = (
                    "pinned provider is unavailable and no other provider serves it"
                )
        rank = {id(r): i + 1 for i, r in enumerate(eligible)}
        for r in rows:
            r["rank"] = rank.get(id(r))
        rows.sort(key=lambda r: (r["rank"] is None, r["rank"] or 0, r["provider"]))
        step["candidates"] = rows
        return step

    def _unroutable(
        self, model: str, protocol: str | None, seen: set[int]
    ) -> list[dict[str, Any]]:
        """Providers that list the model but cannot take this request."""
        inputs = self._inputs
        if inputs is None:
            return []
        rows = []
        for p in inputs.providers:
            if p.id is None or p.id in seen:
                continue
            names = {
                n
                for m in inputs.models.get(p.id, [])
                for n in (m.name, m.raw_name)
                if n
            }
            if model not in names:
                continue
            if p.status == ProviderStatus.OFFLINE:
                reason = "provider is offline"
            elif protocol == "ollama" and not p.supports_ollama:
                reason = "provider does not serve the Ollama API"
            elif protocol == "llamacpp" and not p.supports_llamacpp:
                reason = "provider does not serve the llama.cpp API"
            else:
                continue
            rows.append({"provider_id": p.id, "provider": p.name, "excluded": reason})
        return rows

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


def _summary(out: dict[str, Any], requested: str) -> str:
    chosen = out.get("chosen")
    if chosen is None:
        tried = ", ".join(out.get("fallback_chain") or [requested])
        return f"No provider can serve this request (tried {tried})."
    via = (
        f" after falling back from {requested} to {chosen['model']}"
        if chosen["fell_back"]
        else ""
    )
    last = out["steps"][-1] if out["steps"] else {}
    return f"Routes to {chosen['provider']}{via}: {last.get('outcome', '')}."


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
