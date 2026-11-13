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
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any

from . import router_identity, routing_table
from .affinity import Binding, SessionAffinity, session_key, short_key
from .alerts import alerts
from .capacity import DEFAULT_LEVEL, CapacityLimits, queue_level
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
    ModelAccess,
    RoutingInputs,
    RoutingTable,
    TableBase,
    alias_groups,
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

    __slots__ = (
        "affinity",
        "backend_model",
        "group",
        "model_key",
        "provider",
        "resolved_model",
    )

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
        # When a bound session's provider was weighed against a less busy
        # one: the costs compared and whether the session moved.
        self.affinity: AffinityWeighing | None = None


@dataclass(frozen=True, slots=True)
class AffinityWeighing:
    """Stay on a session's provider, or move to a less busy one?

    *extra_wait_ms* is how much longer the request would queue on the bound
    provider than on the alternative; *cache_value_ms* how long the
    alternative would take to read the prompt the bound provider has
    cached. The session moves when waiting costs more.
    """

    bound_provider_id: int
    alternative_provider_id: int
    alternative: str
    prompt_tokens: int
    prompt_tps: float
    extra_wait_ms: float
    cache_value_ms: float

    @property
    def moved(self) -> bool:
        return self.extra_wait_ms > self.cache_value_ms

    def to_dict(self) -> dict[str, Any]:
        return {
            "alternative_provider_id": self.alternative_provider_id,
            "alternative": self.alternative,
            "prompt_tokens": self.prompt_tokens,
            "prompt_tokens_per_second": round(self.prompt_tps, 1),
            "extra_wait_ms": round(self.extra_wait_ms, 1),
            "cache_value_ms": round(self.cache_value_ms, 1),
            "moved": self.moved,
        }


class PinUnavailableError(Exception):
    """A key with strict pins asked for a pinned model its pinned provider
    can't serve right now. The request is refused (HTTP 503), never routed
    to another provider or down the fallback chain."""

    def __init__(
        self,
        model: str,
        provider_id: int,
        reason: str,
        *,
        outside_locality: bool = False,
    ):
        self.model = model
        self.provider_id = provider_id
        self.reason = reason
        # The pinned provider serves the model but is outside the key's
        # locality: for a key past its spend cap (narrowed to local), that
        # is the cap's doing, and the caller answers 429 instead.
        self.outside_locality = outside_locality
        super().__init__(
            f"Model '{model}' is pinned to provider {provider_id}, which is "
            f"unavailable ({reason}); this API key's pins are strict"
        )


@dataclass(slots=True)
class RoutingPreferences:
    mode: str = DEFAULT_MODE  # "latency", "throughput", "balanced" or "chaos"
    allow_fallback: bool = True
    pinned_providers: dict[str, int] = field(default_factory=dict)
    # A pinned model whose pinned provider can't serve it is refused
    # (PinUnavailableError) rather than routed to another provider.
    strict_pins: bool = False
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
    # The hub asked for this one provider (the worker shares providers
    # individually): only it may serve the request, with no fallback to others.
    hub_provider: int | None = None
    # The models and groups the API key may use; None: any.
    access: ModelAccess | None = None
    # The dashboard user who issued the key (or is chatting); their quotas
    # apply across all their keys. None for ownerless keys.
    owner_user_id: int | None = None
    # The key's own quotas (None: none), tighter than the owner's.
    daily_token_limit: int | None = None
    requests_per_minute: int | None = None
    # The owner's quotas, their own or the admin's default (None: none).
    user_daily_token_limit: int | None = None
    user_requests_per_minute: int | None = None
    user_daily_spend_cap_usd: float | None = None
    # When the key stops working (Unix time; None: never), and whether it
    # is paused (403) — checked on every request, from the cached record.
    expires_at: float | None = None
    paused: bool = False


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
        cache_aware: bool | None = None,
    ):
        self._db = db
        self._pm = provider_manager
        self._debounce_s = debounce_s
        self._min_interval_s = min_interval_s
        self._table: RoutingTable | None = None
        self._inputs: RoutingInputs | None = None
        # The table's structure from _inputs, and the providers hidden from
        # the hub when it was built; rescored on live samples.
        self._base: TableBase | None = None
        self._base_hidden: frozenset[int] = frozenset()
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
            max_per_key=settings.session_affinity_max_sessions_per_key,
        )
        # (group, member model, provider id) -> requests routed, for metrics.
        self.group_routes: dict[tuple[str, str, int], int] = {}
        self.session_affinity_default = (
            settings.session_affinity_enabled
            if session_affinity is None
            else session_affinity
        )
        # Weigh a bound session's queue against its prompt cache; off, a
        # session stays on its provider whatever the queue.
        self.cache_aware = (
            settings.session_affinity_cache_aware
            if cache_aware is None
            else cache_aware
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

    def hub_loopback_ids(self) -> set[int]:
        """Providers never shared with the hub; see ProviderManager."""
        return self._pm.hub_loopback_ids()

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
        # The stats call this only for a change that can reorder candidates
        # (see ProviderStats.mark_built).
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
                hidden = frozenset(self._pm.hub_loopback_ids())
                base = self._base
                if reload or base is None or hidden != self._base_hidden:
                    # Building runs in a thread so requests keep being served:
                    # the inputs are a snapshot nothing else changes.
                    base = await asyncio.to_thread(
                        routing_table.reload, _hide_from_hub(inputs, set(hidden))
                    )
                    self._base, self._base_hidden = base, hidden
                self._version += 1
                self._pm.stats.mark_built()
                table = await asyncio.to_thread(
                    routing_table.rescore,
                    base,
                    hot_models=self._pm.hot_model_names,
                    stats=self._pm.stats,
                    version=self._version,
                    reasons=reasons,
                )
            except Exception:
                if reload:
                    self._need_reload = True
                self._need_rescore = True
                # Nothing was built from that snapshot: let the next sample
                # count as a change, so it brings another attempt.
                self._pm.stats.forget_built()
                raise
            self._table = table
            if reload and alerts.configured:
                alerts.models_served(
                    {
                        m.name
                        for p in inputs.providers
                        if p.id is not None and p.status != ProviderStatus.OFFLINE
                        for m in inputs.models.get(p.id, [])
                    }
                )
            if reload:
                self._pm.capacity.configure(
                    {
                        p.id: p.max_concurrent
                        for p in inputs.providers
                        if p.id is not None
                    }
                )
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

    def _on_hub_change(self) -> None:
        self.mark_dirty("hub router ID", reload=False)

    async def start(self) -> None:
        """Build the first table and keep it current in the background."""
        router_identity.on_hub_change(self._on_hub_change)
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
        router_identity.remove_hub_listener(self._on_hub_change)

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

    async def current_table(self) -> RoutingTable:
        """The table, built now if there is none or no task keeps it current."""
        table = self._table
        if table is None or (
            self._task is None and (self._need_reload or self._need_rescore)
        ):
            # No background task (tests, or before start): build inline.
            table = await self.rebuild()
        return table

    def _count_group(self, group: str, chosen: Candidate) -> None:
        key = (group, chosen.model, chosen.provider_id)
        self.group_routes[key] = self.group_routes.get(key, 0) + 1

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
        client: str | None = None,
    ) -> str | None:
        """The affinity key for a request, or None when affinity does not apply.

        Called by the handlers before :meth:`route`. Costs one hash over the
        start of the conversation; skipped when affinity is off for the key.
        *client* is the client's address, which scopes requests without a key.
        """
        if not self.affinity_enabled(preferences):
            return None
        if preferences is not None and preferences.mode == "chaos":
            return None
        key_id = preferences.key_id if preferences is not None else None
        return session_key(headers, body, key_id, client)

    def _not_busy(self, ranked: tuple[Candidate, ...]) -> tuple[Candidate, ...]:
        """Drop providers cooling down after a 429, unless all of them are."""
        busy = self._pm.stats.busy_for
        if not any(busy(c.provider_id) > 0 for c in ranked):
            return ranked
        ready = tuple(c for c in ranked if busy(c.provider_id) <= 0)
        return ready or ranked

    def _with_room(
        self, ranked: tuple[Candidate, ...], level: int = DEFAULT_LEVEL
    ) -> tuple[Candidate, ...]:
        """Drop providers at their ``max_concurrent`` limit (for a request
        at queue *level*) while another has room; when every one is full,
        keep them all (the request queues on the one chosen)."""
        capacity = self._pm.capacity
        if not capacity.any_limited():
            return ranked
        room = capacity.has_room
        free = tuple(c for c in ranked if room(c.provider_id, level))
        if len(free) == len(ranked) or not free:
            return ranked
        return free

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
        prompt_tokens: int = 0,
        exclude: frozenset[int] = frozenset(),
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
        is among the eligible candidates, unless (with cache-aware affinity)
        the extra wait in its queue would cost more than the next choice
        re-reading the prompt the bound provider has cached; otherwise the
        normal choice is made and the session is bound to it. Pins and chaos
        mode take precedence. *prompt_tokens* is this request's estimated
        prompt, remembered with the binding for the next request.

        Raises :class:`UnknownEndpointError` for a path the router does not
        proxy, and :class:`PinUnavailableError` when a key with strict pins
        names a pinned model its pinned provider can't serve.

        *exclude* names providers that already failed this request; they are
        not chosen again (see ``proxy_json``'s failover).
        """
        table = await self.current_table()
        return self.route_with(
            table,
            model_name,
            endpoint,
            preferences,
            session=session,
            prompt_tokens=prompt_tokens,
            exclude=exclude,
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
        prompt_tokens: int = 0,
        exclude: frozenset[int] = frozenset(),
    ) -> RouteResult | None:
        """Route against *table*. Synchronous: no I/O, no awaits.

        With *record* false the session binding is consulted but not changed,
        as :meth:`explain` needs.
        """
        strict = bool(preferences and preferences.strict_pins)
        path = resolve_endpoint(endpoint)
        if session is not None and not self.affinity_enabled(preferences):
            session = None
        pins = preferences.pinned_providers if preferences else None
        mode = preferences.mode if preferences else DEFAULT_MODE
        locality = preferences.locality if preferences else None
        hub_only = bool(preferences and preferences.hub_only)
        access = preferences.access if preferences else None
        if not table.allows(access, model_name):
            # Callers refuse such a request first (HTTP 403); never route it.
            return None
        if preferences is not None and preferences.hub_provider is not None:
            return self._route_hub_provider(
                table,
                model_name,
                path,
                preferences.hub_provider,
                preferences.allow_fallback,
                mode,
                locality,
                pins,
                access,
                strict,
            )
        if mode == "chaos" and not (pins and model_name in pins):
            return self._route_chaos(
                table, model_name, path, locality, hub_only, access, exclude
            )
        if mode not in MODES:
            mode = DEFAULT_MODE

        if preferences and not preferences.allow_fallback:
            chain: tuple[str, ...] = (model_name,)
        else:
            chain = table.chain(model_name)
        active = self._pm.active_requests

        for candidate_model in chain:
            if not table.allows(access, candidate_model):
                # A fallback never takes a key to a model it may not use.
                continue
            if pins:
                pid = pins.get(candidate_model)
                if pid is not None:
                    pinned = self._pinned(table, candidate_model, path, pid, mode)
                    why = None
                    outside = False
                    if pinned is not None and pid in exclude:
                        why = "it already failed this request"
                        pinned = None
                    elif pinned is None:
                        why = "it does not serve the model on this endpoint now"
                    elif not _eligible(pinned.provider, locality, hub_only):
                        # The key's locality outranks its pin: never send a
                        # local-only key's request to a cloud provider, nor a
                        # hub's request to a provider kept from it.
                        why = _why_ineligible(pinned.provider, locality, hub_only, "it")
                        outside = not _locality_ok(pinned.provider, locality)
                        logger.warning("Pin for %s ignored: %s", candidate_model, why)
                        pinned = None
                    if pinned is None and strict:
                        raise PinUnavailableError(
                            candidate_model, pid, why or "", outside_locality=outside
                        )
                    if pinned is not None:
                        if record and table.is_group(candidate_model):
                            self._count_group(candidate_model, pinned)
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
            ranked = table.ranked_excluding(
                candidate_model, path, mode, locality, hub_only, exclude
            )
            if not ranked:
                continue
            ranked = self._with_room(self._not_busy(ranked), queue_level(mode))
            chosen = None
            sticky = None
            weighing = None
            if session is not None:
                affinity = self.affinity
                sticky = affinity.get(session) if record else affinity.peek(session)
                if sticky is not None:
                    for c in ranked:
                        if sticky.matches(c.provider_id, c.model):
                            chosen = c
                            break
                if chosen is not None and sticky is not None and self.cache_aware:
                    # Only another provider of the bound member model: a
                    # group's session keeps its model when it moves.
                    same = tuple(c for c in ranked if c.model == chosen.model)
                    alt = _pick(same, mode, active)
                    if alt is not chosen:
                        weighing = self._weigh(sticky, chosen, alt, active)
                        if weighing.moved:
                            chosen = alt
            if chosen is None:
                chosen = _pick(ranked, mode, active)
            if session is not None and record:
                moved = weighing is not None and weighing.moved
                self.affinity.bind(
                    session,
                    chosen.provider_id,
                    sticky,
                    chosen.model,
                    prompt_tokens,
                    overflow=moved,
                )
                if moved and weighing is not None and sticky is not None:
                    logger.info(
                        "Session %s moved from provider %d to %s on %s: "
                        "%.0f ms more queueing outweighs %.0f ms to re-read "
                        "its %d-token prompt",
                        short_key(session),
                        sticky.provider_id,
                        chosen.model,
                        chosen.provider.name,
                        weighing.extra_wait_ms,
                        weighing.cache_value_ms,
                        weighing.prompt_tokens,
                    )
                elif sticky is not None and not sticky.matches(
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
            group = _group_of(table, candidate_model)
            if group is not None and record:
                self._count_group(group, chosen)
            result = RouteResult(
                chosen.provider,
                candidate_model,
                chosen.backend_model,
                chosen.model,
                group,
            )
            result.affinity = weighing
            return result
        return None

    def _weigh(
        self,
        bound: Binding,
        here: Candidate,
        alt: Candidate,
        active: Callable[[int], int],
    ) -> AffinityWeighing:
        """Compare the bound candidate *here* with the normal choice *alt*.

        The cache is worth the time *alt* would spend reading the prompt the
        bound provider holds; a cloud provider's cache is not ours to count
        on, so it is worth nothing. Waiting costs the extra requests ahead in
        the bound provider's queue, each at its balanced-mode queue weight.
        """
        tps = self._pm.stats.prompt_tps(alt.provider_id, alt.model)
        cache_ms = 0.0 if here.provider.is_cloud else bound.prompt_tokens / tps * 1000.0
        extra = (
            active(here.provider_id) - active(alt.provider_id)
        ) * here.queue_weight["balanced"]
        return AffinityWeighing(
            bound_provider_id=here.provider_id,
            alternative_provider_id=alt.provider_id,
            alternative=alt.provider.name,
            prompt_tokens=bound.prompt_tokens,
            prompt_tps=tps,
            extra_wait_ms=float(extra),
            cache_value_ms=cache_ms,
        )

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
        table = await self.current_table()
        try:
            decision = self.route_with(
                table, model_name, endpoint, preferences, session=session, record=False
            )
        except PinUnavailableError:
            # explain_with walks to the same refused pin and says so.
            decision = None
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
        strict = bool(preferences and preferences.strict_pins)
        out: dict[str, Any] = {
            "request": {
                "model": model_name,
                "mode": mode,
                "queue_level": queue_level(mode),
                "locality": locality or "any",
                "hub_only": hub_only,
                "spend_capped": bool(preferences and preferences.spend_capped),
                "endpoint": path,
                "endpoint_class": endpoint_cls,
                "protocol": protocol or "any",
                "allow_fallback": allow_fallback,
                "pins": pins,
                "strict_pins": strict,
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
        access = preferences.access if preferences else None
        if access is not None:
            out["request"]["allowed"] = {
                "models": sorted(access.models),
                "groups": sorted(access.groups),
            }
            if not table.allows(access, model_name):
                out["summary"] = (
                    f"Refused: this API key may not use '{model_name}' (HTTP 403)."
                )
                return out
        bound = self.affinity.peek(session) if session and affinity_on else None
        if session:
            out["session"] = self._explain_session(
                session, bound, decision, mode, affinity_on, model_name in pins
            )

        if mode == "chaos" and model_name not in pins:
            out["chaos_pool"] = [
                {"provider_id": p.id, "provider": p.name, "models": allowed}
                for p, models in table.chaos.get(protocol, ())
                if _eligible(p, locality, hub_only)
                and (
                    allowed := [
                        m for m in models if table.chaos_allows(access, p.id, m, path)
                    ]
                )
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
            if not table.allows(access, candidate_model):
                out["steps"].append(
                    {
                        "model": candidate_model,
                        "pinned_provider_id": None,
                        "candidates": [],
                        "outcome": "skipped: this API key may not use it",
                    }
                )
                continue
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
                strict,
            )
            out["steps"].append(step)
            if step.get("refused"):
                # route_with raised PinUnavailableError here: no other
                # provider, and no further fallback.
                out["refused"] = {
                    "status": 503,
                    "model": candidate_model,
                    "pinned_provider_id": step["pinned_provider_id"],
                }
                out["summary"] = (
                    f"Refused (HTTP 503): '{candidate_model}' is pinned to a "
                    "provider that is unavailable, and this API key's pins "
                    "are strict."
                )
                return out
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
        strict: bool = False,
    ) -> dict[str, Any]:
        step: dict[str, Any] = {"model": model, "pinned_provider_id": None}
        if table.is_group(model):
            step["group"] = {"name": model, "members": list(table.groups[model])}
        rows: list[dict[str, Any]] = []
        all_cands = table.all_candidates.get((model, path), ())
        ranked = table.ranked_for(model, path, mode, locality, hub_only)
        ready = self._not_busy(ranked)
        level = queue_level(mode)
        roomy = self._with_room(ready, level)
        capacity = self._pm.capacity
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
        # A strict pin that can't be honoured refuses the request outright.
        refused = pid is not None and pinned is None and strict
        skipped = 0

        for c in all_cands:
            q = active(c.provider_id)
            row = c.to_dict()
            limit = capacity.limit(c.provider_id)
            if limit is not None:
                gate = capacity.gate(c.provider_id)
                row["max_concurrent"] = limit
                row["queued_requests"] = gate.queued if gate is not None else 0
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
            elif refused:
                row["excluded"] = (
                    "another provider is pinned for this model; "
                    "this key's pins are strict"
                )
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
            elif c not in roomy:
                row["excluded"] = (
                    f"at capacity ({_capacity_text(capacity, c.provider_id)}); "
                    "another provider has room"
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

        if refused:
            step["refused"] = True
            step["outcome"] = "pinned provider unavailable; this key's pins are strict"
        elif pinned is not None:
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
            first_id = eligible[0]["provider_id"]
            if not capacity.has_room(first_id, level):
                step["outcome"] += (
                    f"; all at capacity; would queue on {eligible[0]['provider']} "
                    f"({_capacity_text(capacity, first_id)})"
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
            if decision.affinity is not None:
                w = decision.affinity
                info["weighing"] = w.to_dict()
                info["reason"] += (
                    f": {w.extra_wait_ms:.0f} ms more queueing than on "
                    f"{w.alternative} is less than the {w.cache_value_ms:.0f} ms "
                    f"it would take to re-read the {w.prompt_tokens}-token prompt"
                )
        elif decision is not None and decision.affinity is not None:
            w = decision.affinity
            info["weighing"] = w.to_dict()
            info["reason"] = (
                f"the bound provider's queue costs {w.extra_wait_ms:.0f} ms more "
                f"than {w.alternative}'s, more than the {w.cache_value_ms:.0f} ms "
                f"it takes to re-read the {w.prompt_tokens}-token prompt there, "
                "so the next request moves the session"
            )
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

    def _route_hub_provider(
        self,
        table: RoutingTable,
        model_name: str,
        path: str,
        pid: int,
        allow_fallback: bool,
        mode: str,
        locality: str | None,
        pins: dict[str, int] | None = None,
        access: ModelAccess | None = None,
        strict: bool = False,
    ) -> RouteResult | None:
        """Route to the one provider the hub asked for, or nowhere.

        The worker's API key still has its say: a model it pins to another
        provider, or may not use, is not served by this one. With strict
        pins, such a model is refused rather than tried further down the
        fallback chain.
        """
        chain = table.chain(model_name) if allow_fallback else (model_name,)
        for candidate_model in chain:
            if not table.allows(access, candidate_model):
                continue
            pinned = pins.get(candidate_model) if pins else None
            if pinned is not None and pinned != pid:
                logger.warning(
                    "Hub asked for provider %d, but the key pins %s to provider %d",
                    pid,
                    candidate_model,
                    pinned,
                )
                if strict:
                    raise PinUnavailableError(
                        candidate_model,
                        pinned,
                        f"the hub asked for provider {pid}",
                    )
                continue
            c = self._pinned(table, candidate_model, path, pid, mode)
            if c is None or not _eligible(c.provider, locality, hub_only=True):
                if strict and pinned is not None:
                    raise PinUnavailableError(
                        candidate_model,
                        pinned,
                        "it does not serve the model on this endpoint now"
                        if c is None
                        else _why_ineligible(c.provider, locality, True, "it"),
                        outside_locality=c is not None
                        and not _locality_ok(c.provider, locality),
                    )
                continue
            return RouteResult(
                c.provider,
                candidate_model,
                c.backend_model,
                c.model,
                _group_of(table, candidate_model),
            )
        return None

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
        path: str,
        locality: str | None = None,
        hub_only: bool = False,
        access: ModelAccess | None = None,
        exclude: frozenset[int] = frozenset(),
    ) -> RouteResult | None:
        """Chaos mode: pick a random online provider and random eligible model
        (one the key may use), other than the providers in *exclude*."""
        protocol = ENDPOINTS[path][0]
        entries = []
        for provider, models in table.chaos.get(protocol, ()):
            if not _eligible(provider, locality, hub_only) or provider.id in exclude:
                continue
            if access is not None:
                models = tuple(
                    m
                    for m in models
                    if table.chaos_allows(access, provider.id, m, path)
                )
                if not models:
                    continue
            entries.append((provider, models))
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


def _hide_from_hub(inputs: RoutingInputs, hidden: set[int]) -> RoutingInputs:
    """*inputs* with the providers in *hidden* not shared with the hub."""
    if not hidden:
        return inputs
    return replace(
        inputs,
        providers=[
            p.model_copy(update={"hub_shared": False})
            if p.id in hidden and p.hub_shared
            else p
            for p in inputs.providers
        ],
    )


def _capacity_text(capacity: CapacityLimits, provider_id: int) -> str:
    gate = capacity.gate(provider_id)
    if gate is None:
        return "no limit"
    text = f"{gate.in_use}/{gate.limit}"
    if gate.queued:
        text += f", {gate.queued} queued"
    return text


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
