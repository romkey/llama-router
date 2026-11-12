"""Precomputed routing table.

Routing inputs (providers, models, addresses, benchmarks, fallbacks) are read
from the database in one pass, combined with in-memory signals (hot models,
health-check round trips, provider health, request outcomes), and turned into
an immutable table. The table holds, for every model and every proxied
endpoint, the candidates ranked per routing mode. The request path only reads
the table and adds the live queue depth, so a routing decision never awaits
anything.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field, replace
from typing import Any

from .config import settings
from .database import Database
from .endpoints import (
    DECISION_PATHS,
    ENDPOINTS,
    GENERATE,
    RERANK_PATH,
    normalize_endpoint,  # noqa: F401  (re-exported for callers)
)
from .models import (
    ENDPOINT_MISSING,
    ENDPOINT_SUPPORTED,
    ENDPOINT_UNKNOWN,
    BenchmarkResult,
    ModelGroup,
    Provider,
    ProviderAddress,
    ProviderModel,
    ProviderStatus,
    measured_cold_start,
)
from .pricing import Price, PriceBook
from .routing_stats import (
    HEALTHY,
    MODEL_CIRCUIT_MIN_SAMPLES,
    MODEL_CIRCUIT_RATE,
    ProviderHealth,
    ProviderStats,
    model_circuit_open,
)
from .upstream_url import LOCALITIES

# Decisions that are not passed through are asked as chat completions.
DECISION_UPSTREAM = "/v1/chat/completions"

MODES = ("latency", "throughput", "balanced", "cheapest")
DEFAULT_MODE = "balanced"
# Request protocols: None takes any provider, otherwise only providers that
# speak that protocol.
PROTOCOLS: tuple[str | None, ...] = (None, "ollama", "openai")

# Startup cost assumed for a cold model nobody has benchmarked. Latency mode
# keeps the old router's large value so a measured provider always wins;
# balanced mode uses a typical model load instead.
UNKNOWN_STARTUP_MS = 1_000_000.0
BALANCED_UNKNOWN_STARTUP_MS = 10_000.0
# Time to first token assumed for a loaded model nobody has measured on its
# provider, when no other provider of the model has been measured either.
# Unmeasured is not instant: an untried provider must not outrank one whose
# real time to first token is known.
UNKNOWN_HOT_TTFT_MS = 500.0
# Balanced mode estimates the time to finish a typical response.
NOMINAL_TOKENS = 256
UNKNOWN_TPS = 10.0
# Embedding, image and audio calls are scored on measured service time.
UNKNOWN_SERVICE_MS = 10_000.0
BALANCED_UNKNOWN_SERVICE_MS = 2_000.0
# Cost of each request already in flight on a provider.
QUEUE_WEIGHT = {"latency": 1000.0, "throughput": 100.0}
# Added per unit of recent failure rate (0..1).
FAILURE_PENALTY = {
    "latency": 10_000.0,
    "throughput": 100.0,
    "balanced": 10_000.0,
    # In micro-dollars: a provider failing half the time costs $0.05 more.
    "cheapest": 100_000.0,
}
# "cheapest" scores are micro-dollars for a nominal request. A cloud model
# with no known price ranks behind every priced one.
UNKNOWN_PRICE_MICRO_USD = 1_000_000.0
# Within one price, the balanced score breaks ties.
CHEAPEST_TIEBREAK = 0.001
# Recent failure rate for one model at which a candidate is skipped while
# others exist. Provider-wide failures have their own breaker in the stats.
CIRCUIT_OPEN_RATE = MODEL_CIRCUIT_RATE
CIRCUIT_MIN_SAMPLES = MODEL_CIRCUIT_MIN_SAMPLES
# Model larger than the provider's GPU memory: likely CPU offload.
OVERSIZE_PENALTY_MS = 5_000.0

# Candidate tiers. A request goes to the best tier that has anyone in it.
TIER_OK = 0
TIER_FAILING = 1  # circuit open: failing recently, may still work
TIER_MISSING = 2  # the last probe found the endpoint missing: a near-certain 404

SUPPORT_SUPPORTED = ENDPOINT_SUPPORTED
SUPPORT_MISSING = ENDPOINT_MISSING
SUPPORT_UNKNOWN = ENDPOINT_UNKNOWN


@dataclass(slots=True)
class Candidate:
    provider: Provider
    model: str
    backend_model: str
    endpoint: str
    endpoint_class: str = GENERATE
    # The protocol the provider is reached by ("ollama" or "openai").
    upstream: str = "ollama"
    is_hot: bool = False
    startup_ms: float | None = None
    tokens_per_second: float | None = None
    # Benchmarked speed, used until live samples arrive.
    bench_tps: float | None = None
    rtt_ms: float | None = None
    ttft_ms: float | None = None
    # Stand-in for an unmeasured hot candidate's ttft_ms: the median measured
    # on the other providers it is ranked with, aliases included (see
    # rescore). None: UNKNOWN_HOT_TTFT_MS.
    ttft_estimate_ms: float | None = None
    service_ms: float | None = None
    model_failure_rate: float = 0.0
    health: ProviderHealth = HEALTHY
    failure_rate: float = 0.0
    circuit_open: bool = False
    oversize: bool = False
    # Price of the model here, if known (see pricing).
    price: Price | None = None
    endpoint_support: str = SUPPORT_UNKNOWN
    tier: int = TIER_OK
    problems: tuple[str, ...] = ()
    scores: dict[str, float] = field(default_factory=dict)
    queue_weight: dict[str, float] = field(default_factory=dict)

    @property
    def provider_id(self) -> int:
        return self.provider.require_id()

    def to_dict(self) -> dict[str, Any]:
        return {
            "provider_id": self.provider.id,
            "provider": self.provider.name,
            "model": self.model,
            "backend_model": self.backend_model,
            "upstream": self.upstream,
            "hot": self.is_hot,
            "startup_ms": self.startup_ms,
            "tokens_per_second": self.tokens_per_second,
            "rtt_ms": None if self.rtt_ms is None else round(self.rtt_ms, 2),
            "ttft_ms": None if self.ttft_ms is None else round(self.ttft_ms, 2),
            "ttft_estimate_ms": (
                None
                if self.ttft_estimate_ms is None
                else round(self.ttft_estimate_ms, 2)
            ),
            "service_ms": (
                None if self.service_ms is None else round(self.service_ms, 2)
            ),
            "model_failure_rate": round(self.model_failure_rate, 4),
            "failure_rate": round(self.failure_rate, 4),
            "health_score": self.health.score,
            "circuit_open": self.circuit_open,
            "oversize": self.oversize,
            "endpoint_support": self.endpoint_support,
            "price": (
                None
                if self.price is None
                else {
                    "input_per_mtok": self.price.input_per_mtok,
                    "output_per_mtok": self.price.output_per_mtok,
                    "source": self.price.source,
                }
            ),
            "tier": self.tier,
            "problems": list(self.problems),
            "scores": {m: round(s, 2) for m, s in self.scores.items()},
        }


@dataclass(slots=True)
class RoutingInputs:
    """Everything the table needs from the database, read in one pass."""

    providers: list[Provider]
    models: dict[int, list[ProviderModel]]
    addresses: dict[int, list[ProviderAddress]]
    benchmarks: dict[tuple[int, str, str | None], BenchmarkResult]
    fallbacks: dict[str, str]
    # alias -> canonical model name
    aliases: dict[str, str] = field(default_factory=dict)
    # Rows of model_prices.
    prices: list[dict[str, Any]] = field(default_factory=list)
    groups: list[ModelGroup] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class ModelAccess:
    """The models and groups an API key may use, by name.

    A model allows it under any of its aliases. A group allows the group's
    name only, not its members by their own names.
    """

    models: frozenset[str] = frozenset()
    groups: frozenset[str] = frozenset()


def alias_groups(aliases: dict[str, str]) -> dict[str, frozenset[str]]:
    """Every name that belongs to an alias group -> all names in the group.

    A group is a canonical name and the aliases pointing at it. Chains are
    followed (a -> b -> c puts all three together); a loop stops at the repeat.
    """

    def canonical(name: str) -> str:
        seen = {name}
        while name in aliases and aliases[name] not in seen:
            name = aliases[name]
            seen.add(name)
        return name

    members: dict[str, set[str]] = {}
    for alias in aliases:
        root = canonical(alias)
        group = members.setdefault(root, {root})
        group.add(alias)
    out: dict[str, frozenset[str]] = {}
    for group in members.values():
        frozen = frozenset(group)
        for name in group:
            out[name] = frozen
    return out


async def load_inputs(db: Database) -> RoutingInputs:
    providers = await db.list_providers()
    models: dict[int, list[ProviderModel]] = {}
    for m in await db.get_all_provider_models():
        models.setdefault(m.provider_id, []).append(m)
    addresses: dict[int, list[ProviderAddress]] = {}
    for a in await db.get_all_addresses():
        addresses.setdefault(a.provider_id, []).append(a)
    benchmarks = latest_benchmarks(await db.get_all_benchmark_results())
    fallbacks = await db.get_all_model_fallbacks()
    aliases = await db.get_model_aliases()
    prices = await db.list_model_prices()
    groups = await db.list_model_groups()
    return RoutingInputs(
        providers, models, addresses, benchmarks, fallbacks, aliases, prices, groups
    )


def latest_benchmarks(
    rows: list[BenchmarkResult],
) -> dict[tuple[int, str, str | None], BenchmarkResult]:
    """Latest measured startup and speed per (provider, model, protocol).

    Also keyed with protocol None for protocol-agnostic routing. *rows* are
    newest first. Startup and speed are taken separately, each from the
    newest run that measured it: a run on an already-loaded model records no
    startup and must not erase the last real cold-start measurement.

    Only real cold starts count as startup measurements. Older rows can hold
    near-zero values from warm runs, and llama.cpp benchmarks used to store
    0 for every run.
    """
    merged: dict[tuple[int, str, str | None], BenchmarkResult] = {}
    for b in rows:
        startup = measured_cold_start(b.startup_time_ms)
        for key in (
            (b.provider_id, b.model_name, b.protocol),
            (b.provider_id, b.model_name, None),
        ):
            cur = merged.get(key)
            if cur is None:
                merged[key] = b.model_copy(update={"startup_time_ms": startup})
                continue
            if cur.startup_time_ms is None and startup is not None:
                cur.startup_time_ms = startup
            if cur.tokens_per_second is None and b.tokens_per_second is not None:
                cur.tokens_per_second = b.tokens_per_second
    return merged


def fallback_chains(
    fallbacks: dict[str, str], max_depth: int = 10
) -> dict[str, tuple[str, ...]]:
    """Each model's fallback chain, itself first, stopping at a repeat."""
    chains: dict[str, tuple[str, ...]] = {}
    for start in fallbacks:
        chain = [start]
        seen = {start}
        cur = start
        for _ in range(max_depth):
            nxt = fallbacks.get(cur)
            if not nxt or nxt in seen:
                break
            chain.append(nxt)
            seen.add(nxt)
            cur = nxt
        chains[start] = tuple(chain)
    return chains


def best_address(addresses: list[ProviderAddress]) -> ProviderAddress | None:
    """The address the provider's clients use: live+preferred > live > preferred."""
    if not addresses:
        return None
    picks: tuple[Callable[[ProviderAddress], bool], ...] = (
        lambda a: a.is_live and a.is_preferred,
        lambda a: a.is_live,
        lambda a: a.is_preferred,
    )
    for pick in picks:
        for a in addresses:
            if pick(a):
                return a
    return addresses[0]


def upstream_for(provider: Provider) -> str:
    """The protocol a request to *provider* is sent with."""
    return provider.protocol.value


def endpoint_support(
    provider: Provider, path: str, stats: ProviderStats | None = None
) -> str:
    """What the last probe said about *path* on the provider's upstream.

    A decision route is supported natively when the upstream serves it, and
    otherwise wherever chat completions are, unless the provider has lately
    answered a decision question without logprobs.
    """
    if path == RERANK_PATH:
        return rerank_support(provider)
    native = provider.endpoint_state(path)
    if path not in DECISION_PATHS or native == ENDPOINT_SUPPORTED:
        return native
    if native_decision_path(provider, path) is not None:
        return ENDPOINT_SUPPORTED
    if (
        stats is not None
        and provider.id is not None
        and stats.lacks_logprobs(provider.id)
    ):
        return ENDPOINT_MISSING
    return provider.endpoint_state(DECISION_UPSTREAM)


# Upstream rerank routes, most usual first. Each also counts under its bare
# form (``/rerank``); all take the same body.
RERANK_UPSTREAMS = ("/v1/rerank", "/v1/reranking", "/v2/rerank")


def rerank_support(provider: Provider) -> str:
    """Supported when the upstream serves any rerank route, else what the
    probe said of ``/v1/rerank``."""
    if rerank_upstream_path(provider) is not None:
        return ENDPOINT_SUPPORTED
    return provider.endpoint_state(RERANK_PATH)


def rerank_upstream_path(provider: Provider) -> str | None:
    """The rerank route the provider's upstream serves; None when its probe
    found none."""
    for p in RERANK_UPSTREAMS:
        if provider.endpoint_state(p) == ENDPOINT_SUPPORTED:
            return p
    return None


def native_decision_path(provider: Provider, path: str) -> str | None:
    """The upstream route a decision request on *path* is passed through to.

    *path* itself when the upstream serves it, else another decision route it
    serves: ``/v1/decisions`` and ``/v1/systemone`` take the same body, so a
    Jev upstream that has only the latter answers both. None when the
    upstream serves neither and the request is emulated.
    """
    for p in (path, *DECISION_PATHS):
        if provider.endpoint_state(p) == ENDPOINT_SUPPORTED:
            return p
    return None


def decides_natively(provider: Provider, path: str) -> bool:
    """The upstream serves decision route *path*, or one with the same body."""
    return native_decision_path(provider, path) is not None


def offers(provider: Provider, path: str, support: str) -> bool:
    """Whether routing may offer *provider* for *path* at all.

    A provider whose preset declares its routes is left out of every other
    route rather than ranked last: a decision-only service asked for a chat
    completion can only fail.
    """
    return support != SUPPORT_MISSING or provider.declared_endpoints is None


# Ollama routes an OpenAI-style provider serves through translation, and the
# route the translated request uses there.
TRANSLATED_ROUTES = {
    "/api/chat": "/v1/chat/completions",
    "/api/generate": "/v1/chat/completions",
    "/api/embed": "/v1/embeddings",
    "/api/embeddings": "/v1/embeddings",
}


def serves(provider: Provider, protocol: str | None) -> bool:
    """Whether *provider* speaks *protocol* natively (None: any)."""
    return protocol is None or provider.protocol.value == protocol


def translates(provider: Provider, protocol: str | None) -> bool:
    """Whether *provider* takes *protocol* requests through translation."""
    return (
        protocol == "ollama" and provider.is_openai and settings.ollama_api_translation
    )


def reaches(provider: Provider, protocol: str | None) -> bool:
    return serves(provider, protocol) or translates(provider, protocol)


def _chaos_eligible(
    provider: Provider, models: list[ProviderModel], protocol: str | None
) -> list[str]:
    if not reaches(provider, protocol) or provider.declared_endpoints is not None:
        # Chaos picks a provider before looking at the route, so a provider
        # that serves only a few routes would fail most requests.
        return []
    return [m.name for m in models]


def combined_failure_rate(model_rate: float, health: ProviderHealth) -> float:
    """Chance a request to this candidate fails, for the penalty.

    Every request outcome feeds both the model's rate and the provider's
    request rate, so they overlap: take the larger instead of adding them.
    Whole-check failures are a separate signal and combine independently.
    """
    requests = max(model_rate, health.request_failure_rate)
    return 1.0 - (1.0 - requests) * (1.0 - health.probe_failure_rate)


def _cold_ms(c: Candidate) -> tuple[float, float]:
    """Startup cost for (latency, balanced) scoring."""
    if c.is_hot:
        hot = _hot_ttft_ms(c)
        return hot, hot
    if c.startup_ms is not None:
        return c.startup_ms, c.startup_ms
    if c.ttft_ms is not None:
        return c.ttft_ms, c.ttft_ms
    return UNKNOWN_STARTUP_MS, BALANCED_UNKNOWN_STARTUP_MS


def _hot_ttft_ms(c: Candidate) -> float:
    """Time to first token of a loaded model: measured, else estimated."""
    if c.ttft_ms is not None:
        return c.ttft_ms
    if c.ttft_estimate_ms is not None:
        return c.ttft_estimate_ms
    return UNKNOWN_HOT_TTFT_MS


def _add_cheapest(c: Candidate) -> None:
    """Score "cheapest": price of a nominal request, then the balanced score."""
    cost = (
        c.price.nominal() * 1_000_000
        if c.price is not None
        else UNKNOWN_PRICE_MICRO_USD
    )
    c.scores["cheapest"] = (
        cost
        + c.failure_rate * FAILURE_PENALTY["cheapest"]
        + c.scores["balanced"] * CHEAPEST_TIEBREAK
    )
    c.queue_weight["cheapest"] = c.queue_weight["balanced"] * CHEAPEST_TIEBREAK


def score(c: Candidate) -> None:
    """Fill *c.scores* and *c.queue_weight* for every mode. Lower is better."""
    _score_speed(c)
    _add_cheapest(c)


def _score_speed(c: Candidate) -> None:
    cold, cold_bal = _cold_ms(c)
    rtt = c.rtt_ms or 0.0
    oversize = OVERSIZE_PENALTY_MS if c.oversize else 0.0
    f = c.failure_rate
    if c.endpoint_class == GENERATE:
        tps = c.tokens_per_second
        gen_ms = NOMINAL_TOKENS / (tps if tps else UNKNOWN_TPS) * 1000.0
        c.scores = {
            "latency": cold + rtt + oversize + f * FAILURE_PENALTY["latency"],
            "throughput": -(tps or 0.0) + f * FAILURE_PENALTY["throughput"],
            "balanced": cold_bal
            + rtt
            + gen_ms
            + oversize
            + f * FAILURE_PENALTY["balanced"],
        }
        c.queue_weight = {
            "latency": QUEUE_WEIGHT["latency"],
            "throughput": QUEUE_WEIGHT["throughput"],
            "balanced": max(1000.0, gen_ms),
        }
        return
    # Embeddings, images, audio: one response, no token stream. What matters
    # is how long the call takes, measured end to end on real requests.
    service = c.service_ms
    c.scores = {
        # Unmeasured is not instant: a provider with a real measurement must
        # not lose to one nobody has timed yet.
        "latency": cold
        + rtt
        + (service if service is not None else UNKNOWN_SERVICE_MS)
        + oversize
        + f * FAILURE_PENALTY["latency"],
        "throughput": (service if service is not None else UNKNOWN_SERVICE_MS)
        + oversize
        + f * FAILURE_PENALTY["balanced"],
        "balanced": cold_bal
        + rtt
        + (service if service is not None else BALANCED_UNKNOWN_SERVICE_MS)
        + oversize
        + f * FAILURE_PENALTY["balanced"],
    }
    per_request = max(1000.0, service or 1000.0)
    c.queue_weight = {
        "latency": QUEUE_WEIGHT["latency"],
        "throughput": per_request,
        "balanced": per_request,
    }


def _classify(c: Candidate) -> None:
    problems: list[str] = []
    tier = TIER_OK
    if c.circuit_open:
        tier = TIER_FAILING
        if c.health.circuit_open:
            problems.append(
                f"provider failing {round(c.health.request_failure_rate * 100)}% "
                "of recent requests (circuit open)"
            )
        else:
            problems.append(
                f"model failing {round(c.model_failure_rate * 100)}% of recent "
                f"{c.endpoint_class} requests here (circuit open)"
            )
    if c.endpoint_support == SUPPORT_MISSING:
        tier = TIER_MISSING
        problems.append(
            f"last probe found {c.endpoint} missing on its {c.upstream} API"
        )
    c.tier = tier
    c.problems = tuple(problems)


@dataclass(slots=True)
class RoutingTable:
    version: int = 0
    built_at: float = 0.0
    build_ms: float = 0.0
    reasons: tuple[str, ...] = ()
    # (model, endpoint) -> every online candidate, in database order.
    all_candidates: dict[tuple[str, str], tuple[Candidate, ...]] = field(
        default_factory=dict
    )
    # (model, endpoint, mode) -> the best tier's candidates, best score first.
    ranked: dict[tuple[str, str, str], tuple[Candidate, ...]] = field(
        default_factory=dict
    )
    # The same, for keys limited to one locality: (model, endpoint, mode,
    # locality). The best tier is chosen among that locality's providers only.
    # Filled on first lookup (see ranked_for): most keys are never asked for.
    ranked_by_locality: dict[
        tuple[str, str, str, str | None], tuple[Candidate, ...]
    ] = field(default_factory=dict)
    # The same for requests from a hub this router works for, which may only
    # use providers shared with it: (model, endpoint, mode, locality or None).
    # Also filled on first lookup.
    ranked_for_hub: dict[tuple[str, str, str, str | None], tuple[Candidate, ...]] = (
        field(default_factory=dict)
    )
    chains: dict[str, tuple[str, ...]] = field(default_factory=dict)
    chaos: dict[str | None, tuple[tuple[Provider, tuple[str, ...]], ...]] = field(
        default_factory=dict
    )
    backend_names: dict[tuple[int, str], str] = field(default_factory=dict)
    health: dict[int, ProviderHealth] = field(default_factory=dict)
    prices: PriceBook | None = None
    # Model group name -> its members as written ("model" or "model@provider").
    groups: dict[str, tuple[str, ...]] = field(default_factory=dict)
    # Group name -> (served by any online provider, served by one shared
    # with the hub), for model listings.
    group_reach: dict[str, tuple[bool, bool]] = field(default_factory=dict)
    # Problems found while building, such as a group shadowing a model.
    warnings: tuple[str, ...] = ()
    # Name -> every name of its alias group (see alias_groups).
    alias_names: dict[str, frozenset[str]] = field(default_factory=dict)

    def is_group(self, name: str) -> bool:
        return name in self.groups

    def allows(self, access: ModelAccess | None, name: str) -> bool:
        """Whether an API key limited to *access* may use *name* (None:
        unlimited). A group name is checked against the key's groups; any
        other name against its models, under any alias."""
        if access is None:
            return True
        if name in self.groups:
            return name in access.groups
        names = self.alias_names.get(name)
        if names is None:
            return name in access.models
        return not names.isdisjoint(access.models)

    def chaos_allows(
        self, access: ModelAccess | None, provider_id: int | None, model: str, path: str
    ) -> bool:
        """Whether chaos mode may give a limited key *model* on a provider.

        Chaos mode picks a model rather than routing a name, so besides the
        key's own models it may pick whatever routing one of the key's
        groups on *path* could: the group's own candidates, which already
        account for raw names, aliases and members tied to one provider.
        Asking for an allowed group in chaos mode would otherwise find
        nothing."""
        if access is None or self.allows(access, model):
            return True
        return any(
            c.provider_id == provider_id and c.model == model
            for group in access.groups
            for c in self.all_candidates.get((group, path), ())
        )

    def group_listing(self, hub_only: bool = False) -> list[dict[str, Any]]:
        """Model-list rows for the groups a client can use, shaped like the
        rows of ``Database.list_all_models``."""
        return [
            {
                "name": name,
                "size": 0,
                "digest": "",
                "modified_at": None,
                "details": {
                    "family": "group",
                    "group": True,
                    "members": list(self.groups.get(name, ())),
                },
            }
            for name in self.listed_groups(hub_only)
        ]

    def listed_groups(self, hub_only: bool = False) -> list[str]:
        """Groups a client should see: at least one member can be served."""
        return sorted(
            g
            for g, (any_p, shared) in self.group_reach.items()
            if (shared if hub_only else any_p)
        )

    def chain(self, model: str) -> tuple[str, ...]:
        return self.chains.get(model) or (model,)

    def ranked_for(
        self,
        model: str,
        endpoint: str,
        mode: str,
        locality: str | None = None,
        hub_only: bool = False,
    ) -> tuple[Candidate, ...]:
        """Ranked candidates, limited to *locality* when given and to
        providers shared with the hub when *hub_only*."""
        if not hub_only and locality is None:
            return self.ranked.get((model, endpoint, mode), ())
        memo = self.ranked_for_hub if hub_only else self.ranked_by_locality
        key = (model, endpoint, mode, locality)
        hit = memo.get(key)
        if hit is not None:
            return hit
        cands = self.all_candidates.get((model, endpoint))
        if (
            not cands
            or mode not in MODES
            or (locality is not None and locality not in LOCALITIES)
        ):
            # Not memoized, so unknown names cannot grow the table.
            return ()
        ranked = _rank(
            [
                c
                for c in cands
                if (locality is None or c.provider.locality == locality)
                and (c.provider.hub_shared or not hub_only)
            ],
            mode,
        )
        memo[key] = ranked
        return ranked

    def backend_name(self, provider_id: int, model: str) -> str:
        return self.backend_names.get((provider_id, model), model)

    def first_choices(self, endpoint: str) -> dict[str, dict[str, str]]:
        """model -> mode -> first-choice provider name, for one endpoint."""
        rows: dict[str, dict[str, str]] = {}
        for (model, ep, mode), ranked in self.ranked.items():
            if ep == endpoint and ranked:
                rows.setdefault(model, {})[mode] = ranked[0].provider.name
        return rows

    def to_dict(self) -> dict[str, Any]:
        models: dict[str, Any] = {}
        for (model, endpoint), cands in sorted(self.all_candidates.items()):
            entry = models.setdefault(model, {})
            entry[endpoint] = {
                "candidates": [c.to_dict() for c in cands],
                "ranked": {
                    mode: [
                        c.provider.name
                        for c in self.ranked.get((model, endpoint, mode), ())
                    ]
                    for mode in MODES
                },
            }
        return {
            "version": self.version,
            "built_at": self.built_at,
            "build_ms": round(self.build_ms, 3),
            "reasons": list(self.reasons),
            "fallback_chains": {k: list(v) for k, v in self.chains.items()},
            "groups": {k: list(v) for k, v in self.groups.items()},
            "warnings": list(self.warnings),
            "provider_health": {str(k): v.to_dict() for k, v in self.health.items()},
            "models": models,
        }


@dataclass(slots=True)
class TableBase:
    """A table's structure, read from the database inputs and not yet scored.

    Kept between rebuilds so a change in live samples only rescores (see
    :func:`rescore`) instead of reading the database and building every
    candidate again.
    """

    # (model, endpoint) -> every candidate on an online provider, live
    # fields unset. Shared with groups, so one object can sit in several.
    candidates: dict[tuple[str, str], tuple[Candidate, ...]] = field(
        default_factory=dict
    )
    providers: list[Provider] = field(default_factory=list)
    # Provider id -> its live address ids, whose round trips count.
    live_addresses: dict[int, list[int]] = field(default_factory=dict)
    chains: dict[str, tuple[str, ...]] = field(default_factory=dict)
    chaos: dict[str | None, tuple[tuple[Provider, tuple[str, ...]], ...]] = field(
        default_factory=dict
    )
    backend_names: dict[tuple[int, str], str] = field(default_factory=dict)
    prices: PriceBook | None = None
    groups: dict[str, tuple[str, ...]] = field(default_factory=dict)
    warnings: tuple[str, ...] = ()
    alias_names: dict[str, frozenset[str]] = field(default_factory=dict)


def build_table(
    inputs: RoutingInputs,
    *,
    hot_models: Callable[[int, str], set[str]],
    stats: ProviderStats | None,
    version: int = 0,
    reasons: tuple[str, ...] = (),
) -> RoutingTable:
    """Build a table from scratch: :func:`reload`, then :func:`rescore`."""
    started = time.perf_counter()
    table = rescore(
        reload(inputs),
        hot_models=hot_models,
        stats=stats,
        version=version,
        reasons=reasons,
    )
    table.build_ms = (time.perf_counter() - started) * 1000
    return table


def reload(inputs: RoutingInputs) -> TableBase:
    """Build every candidate from the database inputs, without live samples."""
    online = [
        p
        for p in inputs.providers
        if p.id is not None and p.status != ProviderStatus.OFFLINE
    ]

    base = TableBase(providers=online)
    grouped: dict[tuple[str, str], list[Candidate]] = {}
    backend_names: dict[tuple[int, str], str] = {}
    groups = alias_groups(inputs.aliases)
    base.alias_names = groups
    book = PriceBook(inputs.providers, inputs.models, inputs.prices)
    base.prices = book

    for p in online:
        pid = p.require_id()
        models = inputs.models.get(pid, [])
        up = upstream_for(p)
        addresses = inputs.addresses.get(pid, [])
        # Only live addresses count: an address that stopped answering keeps
        # its old samples, which must not make the provider look fast.
        base.live_addresses[pid] = [
            a.id for a in addresses if a.is_live and a.id is not None
        ]
        gpu_bytes = p.gpu_ram_bytes
        # Endpoint support depends only on the provider, not the model.
        support: dict[str, str] = {}
        for path, (protocol, _cls) in ENDPOINTS.items():
            if serves(p, protocol):
                sup = endpoint_support(p, path)
            elif translates(p, protocol) and path in TRANSLATED_ROUTES:
                # The probe knows the route the translated request lands on.
                sup = endpoint_support(p, TRANSLATED_ROUTES[path])
            else:
                continue
            # A decision route may still be ruled out by live samples
            # (rescore), so keep it for now.
            if offers(p, path, sup) or path in DECISION_PATHS:
                support[path] = sup

        for m in models:
            backend = m.backend_name
            backend_names[(pid, m.name)] = backend
            if m.raw_name:
                backend_names[(pid, m.raw_name)] = backend
            oversize = bool(gpu_bytes and m.size and m.size > gpu_bytes)
            keys = {m.name}
            if m.raw_name:
                keys.add(m.raw_name)
            # Any alias of the model reaches it too, under its own name here.
            for name in list(keys):
                keys |= groups.get(name, frozenset())
            for key in keys:
                backend_names.setdefault((pid, key), backend)
            for path, sup in support.items():
                protocol, cls = ENDPOINTS[path]
                bench = inputs.benchmarks.get((pid, m.name, up))
                if bench is None and protocol is None:
                    bench = inputs.benchmarks.get((pid, m.name, None))
                c = Candidate(
                    provider=p,
                    model=m.name,
                    backend_model=backend,
                    endpoint=path,
                    endpoint_class=cls,
                    upstream=up,
                    startup_ms=bench.startup_time_ms if bench else None,
                    bench_tps=(
                        bench.tokens_per_second if bench and cls == GENERATE else None
                    ),
                    oversize=oversize,
                    price=book.price(pid, m.name),
                    endpoint_support=sup,
                )
                for key in keys:
                    bucket = grouped.setdefault((key, path), [])
                    if all(x.provider.id != pid for x in bucket):
                        bucket.append(c)

    base.warnings = tuple(_add_groups(base, grouped, inputs))
    base.candidates = {k: tuple(v) for k, v in grouped.items()}

    for protocol in PROTOCOLS:
        entries = []
        for p in online:
            eligible = _chaos_eligible(p, inputs.models.get(p.id or 0, []), protocol)
            if eligible:
                entries.append((p, tuple(eligible)))
        base.chaos[protocol] = tuple(entries)

    base.chains = fallback_chains(inputs.fallbacks)
    base.backend_names = backend_names
    return base


def rescore(
    base: TableBase,
    *,
    hot_models: Callable[[int, str], set[str]],
    stats: ProviderStats | None,
    version: int = 0,
    reasons: tuple[str, ...] = (),
) -> RoutingTable:
    """A new table from *base* scored with the live samples in *stats*.

    *base* is left as it is: every candidate is copied, so a table being
    served is never changed under a request.
    """
    started = time.perf_counter()
    now = stats.now() if stats else 0.0
    table = RoutingTable(
        version=version,
        reasons=reasons,
        chains=base.chains,
        chaos=base.chaos,
        backend_names=base.backend_names,
        prices=base.prices,
        groups=base.groups,
        warnings=base.warnings,
        alias_names=base.alias_names,
    )
    hot: dict[int, set[str]] = {}
    rtt: dict[int, float | None] = {}
    for p in base.providers:
        pid = p.require_id()
        hot[pid] = hot_models(pid, upstream_for(p))
        live_ids = base.live_addresses.get(pid)
        rtt[pid] = stats.rtt_ms(pid, live_ids) if stats and live_ids else None
        table.health[pid] = stats.provider_health(pid, now) if stats else HEALTHY

    # Measured time to first token per (model, endpoint), for estimating the
    # candidates of that model nobody has measured yet.
    measured = _measured_ttft(base, stats)

    # A candidate can sit in several buckets (aliases, groups): score it once.
    scored: dict[int, Candidate | None] = {}

    def live(c: Candidate) -> Candidate | None:
        if id(c) in scored:
            return scored[id(c)]
        pid = c.provider_id
        new = replace(c, scores={}, queue_weight={})
        sup = c.endpoint_support
        if c.endpoint in DECISION_PATHS:
            sup = endpoint_support(c.provider, c.endpoint, stats)
        if not offers(c.provider, c.endpoint, sup):
            scored[id(c)] = None
            return None
        ms = stats.model(pid, c.model, c.endpoint_class) if stats else None
        model_rate = ms.failures.current(now) if ms else 0.0
        health = table.health[pid]
        observed_tps = ms.tokens_per_second.value if ms else None
        # Loaded now: Ollama's /api/ps, or the model llama.cpp reports in
        # /props. A cloud service has every model loaded.
        new.is_hot = (
            c.provider.is_cloud
            or c.model in hot[pid]
            or bool(stats and stats.is_warm(pid, c.model, now))
        )
        new.tokens_per_second = (
            (observed_tps or c.bench_tps) if c.endpoint_class == GENERATE else None
        )
        new.rtt_ms = rtt[pid]
        new.ttft_ms = ms.ttft_ms.value if ms else None
        new.ttft_estimate_ms = (
            _median(measured.get(id(c), {}).values()) if new.ttft_ms is None else None
        )
        new.service_ms = ms.duration_ms.value if ms else None
        new.model_failure_rate = model_rate
        new.health = health
        new.failure_rate = combined_failure_rate(model_rate, health)
        new.circuit_open = (
            bool(ms and model_circuit_open(ms, now)) or health.circuit_open
        )
        new.endpoint_support = sup
        score(new)
        _classify(new)
        scored[id(c)] = new
        return new

    for (model, path), cands in base.candidates.items():
        bucket = tuple(x for x in (live(c) for c in cands) if x is not None)
        if not bucket:
            continue
        table.all_candidates[(model, path)] = bucket
        best_tier = min(c.tier for c in bucket)
        usable = [c for c in bucket if c.tier == best_tier]
        for mode in MODES:
            table.ranked[(model, path, mode)] = _sorted(usable, mode)

    # Group name -> (served by any provider, served by one shared with the
    # hub), from the candidates rescoring kept.
    for g in base.groups:
        table.group_reach[g] = (False, False)
    for (model, _path), bucket in table.all_candidates.items():
        if model in table.group_reach:
            shared = table.group_reach[model][1] or any(
                c.provider.hub_shared for c in bucket
            )
            table.group_reach[model] = (True, shared)

    table.built_at = time.time()
    table.build_ms = (time.perf_counter() - started) * 1000
    return table


def _measured_ttft(
    base: TableBase, stats: ProviderStats | None
) -> dict[int, dict[tuple[int, str], float]]:
    """Candidate (by ``id``) -> live time to first token of every measured
    candidate it is ranked against, keyed by (provider id, model).

    Peers are the candidates of each (model, endpoint) bucket the candidate
    is in, so an alias of a model counts the providers listed under other
    names in the same bucket. A model group's bucket is left out: its other
    members are different models."""
    out: dict[int, dict[tuple[int, str], float]] = {}
    if stats is None:
        return out
    for (name, _path), cands in base.candidates.items():
        if name in base.groups:
            continue
        peers: dict[tuple[int, str], float] = {}
        for c in cands:
            ms = stats.model(c.provider_id, c.model, c.endpoint_class)
            if ms is not None and ms.ttft_ms.value is not None:
                peers[(c.provider_id, c.model)] = ms.ttft_ms.value
        if peers:
            for c in cands:
                out.setdefault(id(c), {}).update(peers)
    return out


def _median(values: Iterable[float]) -> float | None:
    ordered = sorted(values)
    if not ordered:
        return None
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2


def _sorted(cands: list[Candidate], mode: str) -> tuple[Candidate, ...]:
    return tuple(sorted(cands, key=lambda c: c.scores[mode]))


def _rank(cands: list[Candidate], mode: str) -> tuple[Candidate, ...]:
    """The best tier's candidates, best score in *mode* first."""
    if not cands:
        return ()
    best_tier = min(c.tier for c in cands)
    return _sorted([c for c in cands if c.tier == best_tier], mode)


def member_label(model: str, provider: str | None) -> str:
    return model if provider is None else f"{model}@{provider}"


def _add_groups(
    table: TableBase,
    grouped: dict[tuple[str, str], list[Candidate]],
    inputs: RoutingInputs,
) -> list[str]:
    """Add a candidate bucket per (group, endpoint) to *grouped*.

    A group's candidates are its members' candidates, already scored, so
    ranking a group is the same as ranking a model and a request for a
    group name costs one lookup. A provider can serve several members, so
    candidates are told apart by provider and model.
    """
    warnings: list[str] = []
    if not inputs.groups:
        return warnings
    names = {p.id: p.name for p in inputs.providers}
    group_names = {g.name for g in inputs.groups}
    model_buckets: dict[str, list[str]] = {}
    for model, path in grouped:
        model_buckets.setdefault(model, []).append(path)
    for g in inputs.groups:
        table.groups[g.name] = tuple(
            member_label(
                m.model_name,
                None if m.provider_id is None else names.get(m.provider_id, "?"),
            )
            for m in g.members
        )
        if g.name in model_buckets:
            warnings.append(
                f"model group {g.name} hides a model of the same name; "
                "requests for that name go to the group"
            )
            for path in model_buckets[g.name]:
                del grouped[(g.name, path)]
        merged: dict[str, list[Candidate]] = {}
        for m in g.members:
            if m.model_name in group_names:
                continue  # groups do not nest
            for path in model_buckets.get(m.model_name, ()):
                bucket = merged.setdefault(path, [])
                for c in grouped.get((m.model_name, path), ()):
                    if m.provider_id is not None and c.provider_id != m.provider_id:
                        continue
                    if any(
                        x.provider_id == c.provider_id and x.model == c.model
                        for x in bucket
                    ):
                        continue
                    bucket.append(c)
        for path, bucket in merged.items():
            if bucket:
                grouped[(g.name, path)] = bucket
    return warnings
