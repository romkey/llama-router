"""Routes requests to the best available provider for a given model.

Routing decisions come from a precomputed :class:`RoutingTable` that ranks,
for every model and every proxied endpoint, the providers per routing mode.
The table is rebuilt in the background whenever something that affects
routing changes: the database reports a write (providers, status, addresses,
models, benchmarks, fallbacks, probe results), the hot-model set changes, or
live request and health-check samples arrive. Rebuilds are coalesced, so a
burst of changes produces one rebuild. ``route()`` itself only reads the table
and the in-memory queue depth; it never touches the database once the table
exists.
"""

from __future__ import annotations

import asyncio
import logging
import random
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from .affinity import Binding, SessionAffinity, session_key, short_key
from .config import settings
from .database import ROUTING_CHANGES, Database
from .endpoints import (  # noqa: F401  (re-exported)
    ENDPOINT_PROTOCOLS,
    ENDPOINTS,
    UnknownEndpointError,
    normalize_endpoint,
    protocol_for_endpoint,
    resolve_endpoint,
)
from .models import Provider, ProviderStatus
from .pricing import Price
from .provider_manager import ProviderManager

if TYPE_CHECKING:
    from .decision_policy import DecisionPolicy
import contextlib

from .routing_table import (
    DEFAULT_MODE,
    MODES,
    TIER_OK,
    Candidate,
    RoutingInputs,
    RoutingTable,
    alias_groups,
    build_table,
    load_inputs,
    reaches,
)

logger = logging.getLogger(__name__)

_API_NAMES = {"ollama": "Ollama", "openai": "OpenAI"}

ROUTING_MODES = frozenset({*MODES, "chaos"})

# Wait this long after the first change before rebuilding, to batch a burst.
REBUILD_DEBOUNCE_S = 0.25
# Never rebuild more often than this.
REBUILD_MIN_INTERVAL_S = 1.0
# Reasons kept for the dashboard.
_MAX_REASONS = 20


class RouteResult:
    """Wraps a routing decision, including which model was actually resolved."""

    __slots__ = ("backend_model", "group", "model_key", "provider", "resolved_model")

    def __init__(
        self,
        provider: Provider,
        resolved_model: str,
        backend_model: str | None = None,
        model_key: str | None = None,
        group: str | None = None,
    ):
        self.provider = provider
        self.resolved_model = resolved_model
        # The model group the request named, when it named one; the member
        # chosen is model_key.
        self.group = group
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
    # None follows the router's default (LLAMA_ROUTER_SESSION_AFFINITY_ENABLED).
    session_affinity: bool | None = None
    # The API key these came from, which scopes session affinity.
    key_id: int | None = None
    # "local" or "cloud" limits routing to those providers; None allows both.
    locality: str | None = None
    # USD per UTC day; past it the key is limited to local providers.
    daily_spend_cap_usd: float | None = None
    # Set when the cap was reached and the locality narrowed because of it.
    spend_capped: bool = False
    # Decision questions asked before a generation request is routed.
    decision_policy: DecisionPolicy | None = None
    # The request came from a hub this router works for (over the worker
    # link): only providers shared with the hub may serve it.
    hub_only: bool = False


class Router:
    def __init__(
        self,
        db: Database,
        provider_manager: ProviderManager,
        *,
        debounce_s: float = REBUILD_DEBOUNCE_S,
        min_interval_s: float = REBUILD_MIN_INTERVAL_S,
        affinity: SessionAffinity | None = None,
        session_affinity: bool | None = None,
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
        self._task: asyncio.Task[None] | None = None
        self.affinity = affinity or SessionAffinity(
            settings.session_affinity_ttl_seconds,
            settings.session_affinity_max_sessions,
        )
        self.session_affinity_default = (
            settings.session_affinity_enabled
            if session_affinity is None
            else session_affinity
        )
        db.add_change_listener(self._on_db_change)
        provider_manager.stats.add_listener(self._on_stats_change)
        provider_manager.attach_router(self)

    # --- Table maintenance ---

    @property
    def table(self) -> RoutingTable | None:
        return self._table

    @property
    def inputs(self) -> RoutingInputs | None:
        """The database inputs the current table was built from."""
        return self._inputs

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
                inputs = self._inputs
                if reload or inputs is None:
                    inputs = self._inputs = await load_inputs(self._db)
                self._version += 1
                table = build_table(
                    inputs,
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
            if reload:
                gone = self.affinity.forget_providers_not_in(
                    {p.id for p in inputs.providers if p.id is not None}
                )
                if gone:
                    logger.debug("Dropped %d sessions bound to removed providers", gone)
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
            with contextlib.suppress(asyncio.CancelledError):
                await task
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

    # --- Session affinity ---

    def affinity_enabled(self, preferences: RoutingPreferences | None) -> bool:
        if preferences is not None and preferences.session_affinity is not None:
            return preferences.session_affinity
        return self.session_affinity_default

    def session_for(
        self,
        headers: Mapping[str, str],
        body: dict[str, Any],
        preferences: RoutingPreferences | None,
    ) -> str | None:
        """The affinity key for a request, or None when affinity does not apply.

        Called by the handlers before :meth:`route`. Costs one hash over the
        start of the conversation; skipped when affinity is off for the key.
        """
        if not self.affinity_enabled(preferences):
            return None
        if preferences is not None and preferences.mode == "chaos":
            return None
        key_id = preferences.key_id if preferences is not None else None
        return session_key(headers, body, key_id)

    def _not_busy(self, ranked: tuple[Candidate, ...]) -> tuple[Candidate, ...]:
        """Drop providers cooling down after a 429, unless all of them are."""
        busy = self._pm.stats.busy_for
        if not any(busy(c.provider_id) > 0 for c in ranked):
            return ranked
        ready = tuple(c for c in ranked if busy(c.provider_id) <= 0)
        return ready or ranked

    def price_for(self, provider_id: int, model: str) -> Price | None:
        """The model's price on the provider from the current table, or None."""
        table = self._table
        if table is None or table.prices is None:
            return None
        return table.prices.price(provider_id, model)

    def forget_session(self, session: str | None) -> None:
        """Let the session's next request be placed afresh (after a failure)."""
        self.affinity.forget(session)

    # --- Routing ---

    async def route(
        self,
        model_name: str,
        endpoint: str,
        preferences: RoutingPreferences | None = None,
        *,
        session: str | None = None,
    ) -> RouteResult | None:
        """Pick the best provider for *model_name* on *endpoint*.

        *endpoint* is the request path (``/api/chat``, ``/v1/embeddings``,
        bare OpenAI paths like ``/chat/completions`` are accepted). It decides
        which providers may serve the request (the protocol), which upstream
        it lands on, whether that upstream's last probe found the endpoint,
        and which live measurements score it. Walks the fallback chain when
        the requested model has no available provider.

        *session* is the request's affinity key from :meth:`session_for`. A
        session already bound to a provider stays there while that provider
        is among the eligible candidates, whatever its queue; otherwise the
        normal choice is made and the session is bound to it. Pins and chaos
        mode take precedence.

        Raises :class:`UnknownEndpointError` for a path the router does not proxy.
        """
        table = await self._current_table()
        return self.route_with(
            table, model_name, endpoint, preferences, session=session
        )

    def route_with(
        self,
        table: RoutingTable,
        model_name: str,
        endpoint: str,
        preferences: RoutingPreferences | None = None,
        *,
        session: str | None = None,
        record: bool = True,
    ) -> RouteResult | None:
        """Route against *table*. Synchronous: no I/O, no awaits.

        With *record* false the session binding is consulted but not changed,
        as :meth:`explain` needs.
        """
        path = resolve_endpoint(endpoint)
        if session is not None and not self.affinity_enabled(preferences):
            session = None
        pins = preferences.pinned_providers if preferences else None
        mode = preferences.mode if preferences else DEFAULT_MODE
        locality = preferences.locality if preferences else None
        hub_only = bool(preferences and preferences.hub_only)
        if mode == "chaos" and not (pins and model_name in pins):
            return self._route_chaos(
                table, model_name, ENDPOINTS[path][0], locality, hub_only
            )
        if mode not in MODES:
            mode = DEFAULT_MODE

        if preferences and not preferences.allow_fallback:
            chain: tuple[str, ...] = (model_name,)
        else:
            chain = table.chain(model_name)
        active = self._pm.active_requests

        for candidate_model in chain:
            if pins:
                pid = pins.get(candidate_model)
                if pid is not None:
                    pinned = self._pinned(table, candidate_model, path, pid, mode)
                    if pinned is not None and not _eligible(
                        pinned.provider, locality, hub_only
                    ):
                        # The key's locality outranks its pin: never send a
                        # local-only key's request to a cloud provider, nor a
                        # hub's request to a provider kept from it.
                        logger.warning(
                            "Pin for %s ignored: %s",
                            candidate_model,
                            _why_ineligible(
                                pinned.provider,
                                locality,
                                hub_only,
                                pinned.provider.name,
                            ),
                        )
                        pinned = None
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
                            _group_of(table, candidate_model),
                        )
            ranked = table.ranked_for(candidate_model, path, mode, locality, hub_only)
            if not ranked:
                continue
            ranked = self._not_busy(ranked)
            chosen = None
            sticky = None
            if session is not None:
                affinity = self.affinity
                sticky = affinity.get(session) if record else affinity.peek(session)
                if sticky is not None:
                    for c in ranked:
                        if sticky.matches(c.provider_id, c.model):
                            chosen = c
                            break
            if chosen is None:
                chosen = _pick(ranked, mode, active)
            if session is not None and record:
                self.affinity.bind(session, chosen.provider_id, sticky, chosen.model)
                if sticky is not None and not sticky.matches(
                    chosen.provider_id, chosen.model
                ):
                    logger.info(
                        "Session %s moved from %s on provider %d to %s on %s: "
                        "bound provider is not eligible for %s",
                        short_key(session),
                        sticky.model or "its model",
                        sticky.provider_id,
                        chosen.model,
                        chosen.provider.name,
                        candidate_model,
                    )
            if candidate_model != model_name:
                logger.info(
                    "Model %s unavailable; fell back to %s",
                    model_name,
                    candidate_model,
                )
            logger.debug(
                "Routing model %s (%s, %s) to %s on provider %s",
                candidate_model,
                path,
                mode,
                chosen.model,
                chosen.provider.name,
            )
            return RouteResult(
                chosen.provider,
                candidate_model,
                chosen.backend_model,
                chosen.model,
                _group_of(table, candidate_model),
            )
        return None

    async def explain(
        self,
        model_name: str,
        endpoint: str,
        preferences: RoutingPreferences | None = None,
        *,
        session: str | None = None,
    ) -> dict[str, Any]:
        """How a request would be routed right now, and why.

        Walks the same fallback chain, pins, tiers and scores as
        :meth:`route`, against the same table and live queue depths, and
        reports every provider considered. The decision itself is taken from
        :meth:`route_with`, so the explanation cannot drift from what a real
        request would get.
        """
        table = await self._current_table()
        decision = self.route_with(
            table, model_name, endpoint, preferences, session=session, record=False
        )
        return self.explain_with(
            table, decision, model_name, endpoint, preferences, session=session
        )

    def explain_with(
        self,
        table: RoutingTable,
        decision: RouteResult | None,
        model_name: str,
        endpoint: str,
        preferences: RoutingPreferences | None,
        *,
        session: str | None = None,
    ) -> dict[str, Any]:
        path = resolve_endpoint(endpoint)
        affinity_on = self.affinity_enabled(preferences)
        protocol, endpoint_cls = ENDPOINTS[path]
        pins = dict(preferences.pinned_providers) if preferences else {}
        mode = preferences.mode if preferences else DEFAULT_MODE
        allow_fallback = preferences.allow_fallback if preferences else True
        locality = preferences.locality if preferences else None
        hub_only = bool(preferences and preferences.hub_only)
        out: dict[str, Any] = {
            "request": {
                "model": model_name,
                "mode": mode,
                "locality": locality or "any",
                "hub_only": hub_only,
                "spend_capped": bool(preferences and preferences.spend_capped),
                "endpoint": path,
                "endpoint_class": endpoint_cls,
                "protocol": protocol or "any",
                "allow_fallback": allow_fallback,
                "pins": pins,
                "session_affinity": affinity_on,
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
        bound = self.affinity.peek(session) if session and affinity_on else None
        if session:
            out["session"] = self._explain_session(
                session, bound, decision, mode, affinity_on, model_name in pins
            )

        if mode == "chaos" and model_name not in pins:
            out["chaos_pool"] = [
                {"provider_id": p.id, "provider": p.name, "models": list(models)}
                for p, models in table.chaos.get(protocol, ())
                if _eligible(p, locality, hub_only)
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
            chosen_model = (
                decision.model_key
                if decision is not None and chosen_id is not None
                else None
            )
            step = self._explain_step(
                table,
                candidate_model,
                path,
                protocol,
                mode,
                pins,
                active,
                chosen_id,
                bound,
                locality,
                hub_only,
                chosen_model,
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
        path: str,
        protocol: str | None,
        mode: str,
        pins: dict[str, int],
        active: Callable[[int], int],
        chosen_id: int | None = None,
        bound: Binding | None = None,
        locality: str | None = None,
        hub_only: bool = False,
        chosen_model: str | None = None,
    ) -> dict[str, Any]:
        step: dict[str, Any] = {"model": model, "pinned_provider_id": None}
        if table.is_group(model):
            step["group"] = {"name": model, "members": list(table.groups[model])}
        rows: list[dict[str, Any]] = []
        all_cands = table.all_candidates.get((model, path), ())
        ranked = table.ranked_for(model, path, mode, locality, hub_only)
        ready = self._not_busy(ranked)
        busy = self._pm.stats.busy_for
        ranked_ids = {c.provider_id for c in ranked}
        best_tier = ranked[0].tier if ranked else TIER_OK
        pid = pins.get(model)
        pinned = None
        if pid is not None:
            step["pinned_provider_id"] = pid
            here = [c for c in all_cands if c.provider_id == pid]
            pinned = min(here, key=lambda c: (c.tier, c.scores[mode])) if here else None
            if pinned is not None and not _eligible(
                pinned.provider, locality, hub_only
            ):
                step["pin_ignored"] = _why_ineligible(
                    pinned.provider, locality, hub_only, pinned.provider.name
                )
                pinned = None
        skipped = 0

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
                    "warnings": [],
                    "session_bound": bound is not None
                    and bound.matches(c.provider_id, c.model),
                }
            )
            if not _eligible(c.provider, locality, hub_only):
                row["excluded"] = _why_ineligible(c.provider, locality, hub_only)
                if not _locality_ok(c.provider, locality):
                    skipped += 1
            elif pinned is not None:
                if c is not pinned:
                    row["excluded"] = "another provider is pinned for this model"
                else:
                    row["warnings"] = list(c.problems)
            elif c.provider_id not in ranked_ids:
                row["excluded"] = "; ".join(c.problems) + (
                    "; healthier providers are available"
                )
            elif c not in ready:
                row["excluded"] = (
                    f"rate limited (HTTP 429) for {busy(c.provider_id):.0f}s more; "
                    "other providers are ready"
                )
            else:
                # In the best tier available, so it is used despite problems.
                row["warnings"] = list(c.problems)
                if busy(c.provider_id) > 0:
                    row["warnings"].append(
                        f"rate limited (HTTP 429) for {busy(c.provider_id):.0f}s more"
                    )
            rows.append(row)

        if locality is not None:
            step["locality"] = (
                f"key requires {locality}; {skipped} "
                f"{'cloud' if locality == 'local' else 'local'} "
                f"provider{'' if skipped == 1 else 's'} skipped"
            )
        rows.extend(
            self._unroutable(model, protocol, {c.provider_id for c in all_cands})
        )
        eligible = [r for r in rows if r["excluded"] is None]
        # The real decision ranks first, so ties read the same way it broke them.
        eligible.sort(
            key=lambda r: (
                not (
                    r["provider_id"] == chosen_id
                    and r.get("model") in (None, chosen_model)
                ),
                r["total_score"],
            )
        )
        is_group = table.is_group(model)

        def where(provider: str, member: str | None) -> str:
            return f"{member} on {provider}" if is_group and member else provider

        if pinned is not None:
            step["outcome"] = f"pinned to {where(pinned.provider.name, pinned.model)}"
        elif (
            eligible
            and bound is not None
            and chosen_id is not None
            and bound.matches(chosen_id, chosen_model or "")
        ):
            step["outcome"] = (
                f"kept on {where(eligible[0]['provider'], eligible[0].get('model'))} "
                "for this session"
            )
        elif eligible:
            step["outcome"] = (
                f"lowest {mode} score: "
                f"{where(eligible[0]['provider'], eligible[0].get('model'))}"
            )
            if best_tier != TIER_OK:
                step["outcome"] += (
                    " (every provider has a problem; using the least bad)"
                )
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

    def _explain_session(
        self,
        session: str,
        bound: Binding | None,
        decision: RouteResult | None,
        mode: str,
        enabled: bool,
        pinned: bool,
    ) -> dict[str, Any]:
        info: dict[str, Any] = {
            "key": short_key(session),
            "bound_provider_id": bound.provider_id if bound else None,
            "bound_provider": self._provider_name(bound.provider_id if bound else None),
            "bound_model": bound.model if bound else None,
            "honored": False,
        }
        if not enabled:
            info["reason"] = "session affinity is off for this request"
        elif mode == "chaos" and not pinned:
            info["reason"] = "chaos mode ignores session affinity"
        elif pinned:
            info["reason"] = "the model is pinned, which overrides session affinity"
        elif bound is None:
            info["reason"] = (
                "no live binding; the first request binds the session "
                "to the normal choice"
            )
        elif (
            decision is not None
            and decision.provider.id is not None
            and bound.matches(decision.provider.id, decision.model_key)
        ):
            info["honored"] = True
            info["reason"] = "the session stays on the provider that served it"
        else:
            info["reason"] = (
                "the bound provider is not an eligible candidate, so the next "
                "request moves the session"
            )
        return info

    def _provider_name(self, provider_id: int | None) -> str | None:
        if provider_id is None or self._inputs is None:
            return None
        for p in self._inputs.providers:
            if p.id == provider_id:
                return p.name
        return None

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
            groups = alias_groups(inputs.aliases)
            for n in list(names):
                names |= groups.get(n, frozenset())
            if model not in names:
                continue
            if p.status == ProviderStatus.OFFLINE:
                reason = "provider is offline"
            elif not reaches(p, protocol):
                api = _API_NAMES.get(protocol or "", protocol or "")
                reason = f"provider does not serve the {api} API"
            else:
                continue
            rows.append(
                {
                    "provider_id": p.id,
                    "provider": p.name,
                    "excluded": reason,
                    "warnings": [],
                }
            )
        return rows

    @staticmethod
    def _pinned(
        table: RoutingTable,
        model: str,
        path: str,
        provider_id: int,
        mode: str = DEFAULT_MODE,
    ) -> Candidate | None:
        """The pinned provider's candidate; for a model group, where the
        provider serves several members, the best scored in *mode*."""
        here = [
            c
            for c in table.all_candidates.get((model, path), ())
            if c.provider.id == provider_id
        ]
        if here:
            mode = mode if mode in MODES else DEFAULT_MODE
            best = min(here, key=lambda c: (c.tier, c.scores[mode]))
            logger.debug(
                "Pinned routing model %s (%s) -> %s on provider %s",
                model,
                path,
                best.model,
                best.provider.name,
            )
            return best
        logger.warning(
            "Pinned route unavailable for model %s (%s) provider_id=%d",
            model,
            path,
            provider_id,
        )
        return None

    @staticmethod
    def _route_chaos(
        table: RoutingTable,
        requested_model: str,
        protocol: str | None,
        locality: str | None = None,
        hub_only: bool = False,
    ) -> RouteResult | None:
        """Chaos mode: pick a random online provider and random eligible model."""
        entries = [
            e
            for e in table.chaos.get(protocol, ())
            if _eligible(e[0], locality, hub_only)
        ]
        if not entries:
            return None
        provider, eligible = random.choice(entries)  # noqa: S311  # chaos mode
        resolved_model = random.choice(eligible)  # noqa: S311  # chaos mode
        provider_id = provider.require_id()
        logger.info(
            "Chaos routing requested=%s protocol=%s -> provider=%s model=%s",
            requested_model,
            protocol or "any",
            provider.name,
            resolved_model,
        )
        return RouteResult(
            provider, resolved_model, table.backend_name(provider_id, resolved_model)
        )


def _group_of(table: RoutingTable, name: str) -> str | None:
    return name if table.is_group(name) else None


def _locality_ok(provider: Provider, locality: str | None) -> bool:
    return locality is None or provider.locality == locality


def _eligible(provider: Provider, locality: str | None, hub_only: bool) -> bool:
    """The request's key locality and hub scope allow *provider*."""
    return _locality_ok(provider, locality) and (provider.hub_shared or not hub_only)


def _why_ineligible(
    provider: Provider, locality: str | None, hub_only: bool, subject: str = "provider"
) -> str:
    if not _locality_ok(provider, locality):
        return f"{subject} is {provider.locality}; the key requires {locality}"
    return f"{subject} is not shared with the hub"


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
    ranked: tuple[Candidate, ...], mode: str, active: Callable[[int], int]
) -> Candidate:
    """Lowest static score plus queue cost; *ranked* is sorted by static score."""
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
