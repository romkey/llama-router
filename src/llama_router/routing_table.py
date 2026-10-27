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
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from .database import Database
from .endpoints import (
    ENDPOINTS,
    GENERATE,
    normalize_endpoint,  # noqa: F401  (re-exported for callers)
)
from .models import (
    BenchmarkResult,
    Provider,
    ProviderAddress,
    ProviderModel,
    ProviderStatus,
)
from .probe import MISSING, SUPPORTED
from .routing_stats import HEALTHY, ProviderHealth, ProviderStats

MODES = ("latency", "throughput", "balanced")
DEFAULT_MODE = "balanced"
PROTOCOLS: tuple[str | None, ...] = (None, "ollama", "llamacpp")

# Startup cost assumed for a cold model nobody has benchmarked. Latency mode
# keeps the old router's large value so a measured provider always wins;
# balanced mode uses a typical model load instead.
UNKNOWN_STARTUP_MS = 1_000_000.0
BALANCED_UNKNOWN_STARTUP_MS = 10_000.0
# Balanced mode estimates the time to finish a typical response.
NOMINAL_TOKENS = 256
UNKNOWN_TPS = 10.0
# Embedding, image and audio calls are scored on measured service time.
UNKNOWN_SERVICE_MS = 10_000.0
BALANCED_UNKNOWN_SERVICE_MS = 2_000.0
# Cost of each request already in flight on a provider.
QUEUE_WEIGHT = {"latency": 1000.0, "throughput": 100.0}
# Added per unit of recent failure rate (0..1).
FAILURE_PENALTY = {"latency": 10_000.0, "throughput": 100.0, "balanced": 10_000.0}
# Recent failure rate for one model at which a candidate is skipped while
# others exist. Provider-wide failures have their own breaker in the stats.
CIRCUIT_OPEN_RATE = 0.5
CIRCUIT_MIN_SAMPLES = 3
# Model larger than the provider's GPU memory: likely CPU offload.
OVERSIZE_PENALTY_MS = 5_000.0

# Candidate tiers. A request goes to the best tier that has anyone in it.
TIER_OK = 0
TIER_FAILING = 1  # circuit open: failing recently, may still work
TIER_MISSING = 2  # the last probe found the endpoint missing: a near-certain 404

SUPPORT_SUPPORTED = "supported"
SUPPORT_MISSING = "missing"
SUPPORT_UNKNOWN = "unknown"


@dataclass(slots=True)
class Candidate:
    provider: Provider
    model: str
    backend_model: str
    endpoint: str
    endpoint_class: str = GENERATE
    # Which of the provider's upstreams the request would reach.
    upstream: str = "ollama"
    is_hot: bool = False
    startup_ms: float | None = None
    tokens_per_second: float | None = None
    rtt_ms: float | None = None
    ttft_ms: float | None = None
    service_ms: float | None = None
    model_failure_rate: float = 0.0
    health: ProviderHealth = HEALTHY
    failure_rate: float = 0.0
    circuit_open: bool = False
    oversize: bool = False
    endpoint_support: str = SUPPORT_UNKNOWN
    tier: int = TIER_OK
    problems: tuple[str, ...] = ()
    scores: dict[str, float] = field(default_factory=dict)
    queue_weight: dict[str, float] = field(default_factory=dict)

    @property
    def provider_id(self) -> int:
        assert self.provider.id is not None
        return self.provider.id

    def to_dict(self) -> dict[str, Any]:
        return {
            "provider_id": self.provider.id,
            "provider": self.provider.name,
            "backend_model": self.backend_model,
            "upstream": self.upstream,
            "hot": self.is_hot,
            "startup_ms": self.startup_ms,
            "tokens_per_second": self.tokens_per_second,
            "rtt_ms": None if self.rtt_ms is None else round(self.rtt_ms, 2),
            "ttft_ms": None if self.ttft_ms is None else round(self.ttft_ms, 2),
            "service_ms": (
                None if self.service_ms is None else round(self.service_ms, 2)
            ),
            "model_failure_rate": round(self.model_failure_rate, 4),
            "failure_rate": round(self.failure_rate, 4),
            "health_score": self.health.score,
            "circuit_open": self.circuit_open,
            "oversize": self.oversize,
            "endpoint_support": self.endpoint_support,
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
    return RoutingInputs(providers, models, addresses, benchmarks, fallbacks)


def latest_benchmarks(
    rows: list[BenchmarkResult],
) -> dict[tuple[int, str, str | None], BenchmarkResult]:
    """Latest measured startup and speed per (provider, model, protocol).

    Also keyed with protocol None for protocol-agnostic routing. *rows* are
    newest first. Startup and speed are taken separately, each from the
    newest run that measured it: a run on an already-loaded model records no
    startup and must not erase the last real cold-start measurement.
    """
    merged: dict[tuple[int, str, str | None], BenchmarkResult] = {}
    for b in rows:
        for key in (
            (b.provider_id, b.model_name, b.protocol),
            (b.provider_id, b.model_name, None),
        ):
            cur = merged.get(key)
            if cur is None:
                merged[key] = b.model_copy()
                continue
            if cur.startup_time_ms is None and b.startup_time_ms is not None:
                cur.startup_time_ms = b.startup_time_ms
            if not cur.tokens_per_second and b.tokens_per_second:
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
    for pick in (
        lambda a: a.is_live and a.is_preferred,
        lambda a: a.is_live,
        lambda a: a.is_preferred,
    ):
        for a in addresses:
            if pick(a):
                return a
    return addresses[0]


def upstream_for(provider: Provider, protocol: str | None) -> str:
    """Which upstream a request lands on (mirrors V1Client's preference)."""
    if protocol == "ollama":
        return "ollama"
    if protocol == "llamacpp":
        return "llamacpp"
    return "llamacpp" if provider.supports_llamacpp else "ollama"


def _base_url_matches(
    base_url: str, provider: Provider, upstream: str, addr: ProviderAddress | None
) -> bool:
    if provider.is_worker:
        return base_url.endswith(f"({upstream})")
    if addr is None:
        return False
    url = addr.url if upstream == "ollama" else (addr.llamacpp_url or addr.url)
    return base_url.rstrip("/") == url.rstrip("/")


def endpoint_support(
    provider: Provider,
    path: str,
    upstream: str,
    addr: ProviderAddress | None,
) -> str:
    """What the last probe said about *path* on the upstream that serves it.

    Uses the probe of that upstream's URL when it can be told apart (a
    provider serving both APIs can have the endpoint on one and not the
    other); otherwise all probed URLs.
    """
    caps = provider.capabilities or {}
    entries = [e for e in caps.get("endpoints") or [] if e.get("path") == path]
    if not entries:
        return SUPPORT_UNKNOWN
    own = [
        e
        for e in entries
        if _base_url_matches(str(e.get("base_url") or ""), provider, upstream, addr)
    ]
    for group in (own, entries):
        if not group:
            continue
        if any(e.get("state") in SUPPORTED for e in group):
            return SUPPORT_SUPPORTED
        if any(e.get("state") == MISSING for e in group):
            return SUPPORT_MISSING
        return SUPPORT_UNKNOWN
    return SUPPORT_UNKNOWN


def _serves(provider: Provider, protocol: str | None) -> bool:
    if protocol == "ollama":
        return provider.supports_ollama
    if protocol == "llamacpp":
        return provider.supports_llamacpp
    return True


def _chaos_eligible(
    provider: Provider, models: list[ProviderModel], protocol: str | None
) -> list[str]:
    out: list[str] = []
    for m in models:
        details = m.details or {}
        if protocol == "ollama":
            if "_in_ollama" in details:
                if details.get("_in_ollama"):
                    out.append(m.name)
            elif provider.supports_ollama:
                out.append(m.name)
        elif protocol == "llamacpp":
            if "_in_llamacpp" in details:
                if details.get("_in_llamacpp"):
                    out.append(m.name)
            elif provider.supports_llamacpp:
                out.append(m.name)
        else:
            out.append(m.name)
    return out


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
        hot = c.ttft_ms if c.ttft_ms is not None else 0.0
        return hot, hot
    if c.startup_ms is not None:
        return c.startup_ms, c.startup_ms
    if c.ttft_ms is not None:
        return c.ttft_ms, c.ttft_ms
    return UNKNOWN_STARTUP_MS, BALANCED_UNKNOWN_STARTUP_MS


def score(c: Candidate) -> None:
    """Fill *c.scores* and *c.queue_weight* for every mode. Lower is better."""
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
    chains: dict[str, tuple[str, ...]] = field(default_factory=dict)
    chaos: dict[str | None, tuple[tuple[Provider, tuple[str, ...]], ...]] = field(
        default_factory=dict
    )
    backend_names: dict[tuple[int, str], str] = field(default_factory=dict)
    health: dict[int, ProviderHealth] = field(default_factory=dict)

    def chain(self, model: str) -> tuple[str, ...]:
        return self.chains.get(model) or (model,)

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
            "provider_health": {str(k): v.to_dict() for k, v in self.health.items()},
            "models": models,
        }


def build_table(
    inputs: RoutingInputs,
    *,
    hot_models: Callable[[int], set[str]],
    stats: ProviderStats | None,
    version: int = 0,
    reasons: tuple[str, ...] = (),
) -> RoutingTable:
    started = time.perf_counter()
    now = stats.now() if stats else 0.0
    online = [
        p
        for p in inputs.providers
        if p.id is not None and p.status != ProviderStatus.OFFLINE
    ]

    table = RoutingTable(version=version, reasons=reasons)
    grouped: dict[tuple[str, str], list[Candidate]] = {}
    backend_names: dict[tuple[int, str], str] = {}

    for p in online:
        pid = p.id
        assert pid is not None
        models = inputs.models.get(pid, [])
        hot = hot_models(pid)
        addresses = inputs.addresses.get(pid, [])
        live_ids = [a.id for a in addresses if a.is_live and a.id is not None]
        # Only live addresses count: an address that stopped answering keeps
        # its old samples, which must not make the provider look fast.
        rtt = stats.rtt_ms(pid, live_ids) if stats and live_ids else None
        health = stats.provider_health(pid, now) if stats else HEALTHY
        table.health[pid] = health
        addr = best_address(addresses)
        gpu_bytes = p.gpu_ram_bytes
        # Endpoint support depends only on the provider, not the model.
        support: dict[str, tuple[str, str]] = {}
        for path, (protocol, _cls) in ENDPOINTS.items():
            if _serves(p, protocol):
                up = upstream_for(p, protocol)
                support[path] = (up, endpoint_support(p, path, up, addr))

        for m in models:
            backend = m.backend_name
            backend_names[(pid, m.name)] = backend
            if m.raw_name:
                backend_names[(pid, m.raw_name)] = backend
            is_hot = m.name in hot or bool(stats and stats.is_warm(pid, m.name, now))
            oversize = bool(gpu_bytes and m.size and m.size > gpu_bytes)
            keys = {m.name}
            if m.raw_name:
                keys.add(m.raw_name)
            for path, (up, sup) in support.items():
                protocol, cls = ENDPOINTS[path]
                ms = stats.model(pid, m.name, cls) if stats else None
                model_rate = ms.failures.current(now) if ms else 0.0
                model_circuit = bool(
                    ms
                    and ms.failures.samples >= CIRCUIT_MIN_SAMPLES
                    and model_rate >= CIRCUIT_OPEN_RATE
                )
                bench = inputs.benchmarks.get((pid, m.name, up))
                if bench is None and protocol is None:
                    bench = inputs.benchmarks.get((pid, m.name, None))
                observed_tps = ms.tokens_per_second.value if ms else None
                c = Candidate(
                    provider=p,
                    model=m.name,
                    backend_model=backend,
                    endpoint=path,
                    endpoint_class=cls,
                    upstream=up,
                    is_hot=is_hot,
                    startup_ms=bench.startup_time_ms if bench else None,
                    tokens_per_second=(
                        observed_tps or (bench.tokens_per_second if bench else None)
                        if cls == GENERATE
                        else None
                    ),
                    rtt_ms=rtt,
                    ttft_ms=ms.ttft_ms.value if ms else None,
                    service_ms=ms.duration_ms.value if ms else None,
                    model_failure_rate=model_rate,
                    health=health,
                    failure_rate=combined_failure_rate(model_rate, health),
                    circuit_open=model_circuit or health.circuit_open,
                    oversize=oversize,
                    endpoint_support=sup,
                )
                score(c)
                _classify(c)
                for key in keys:
                    bucket = grouped.setdefault((key, path), [])
                    if all(x.provider.id != pid for x in bucket):
                        bucket.append(c)

    for (model, path), cands in grouped.items():
        table.all_candidates[(model, path)] = tuple(cands)
        best_tier = min(c.tier for c in cands)
        usable = [c for c in cands if c.tier == best_tier]
        for mode in MODES:
            table.ranked[(model, path, mode)] = tuple(
                sorted(usable, key=lambda c, m=mode: c.scores[m])
            )

    for protocol in PROTOCOLS:
        entries = []
        for p in online:
            if not _serves(p, protocol):
                continue
            eligible = _chaos_eligible(p, inputs.models.get(p.id or 0, []), protocol)
            if eligible:
                entries.append((p, tuple(eligible)))
        table.chaos[protocol] = tuple(entries)

    table.chains = fallback_chains(inputs.fallbacks)
    table.backend_names = backend_names
    table.built_at = time.time()
    table.build_ms = (time.perf_counter() - started) * 1000
    return table
