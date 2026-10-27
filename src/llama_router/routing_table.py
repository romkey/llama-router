"""Precomputed routing table.

Routing inputs (providers, models, addresses, benchmarks, fallbacks) are read
from the database in one pass, combined with in-memory signals (hot models,
round trips, request outcomes), and turned into an immutable table of
candidates sorted per routing mode. The request path only reads the table
and adds the live queue depth, so a routing decision never awaits anything.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from .database import Database
from .models import (
    BenchmarkResult,
    Provider,
    ProviderAddress,
    ProviderModel,
    ProviderStatus,
)
from .probe import MISSING, SUPPORTED
from .routing_stats import ProviderStats

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
# Cost of each request already in flight on a provider.
QUEUE_WEIGHT = {"latency": 1000.0, "throughput": 100.0}
# Added per unit of recent failure rate (0..1).
FAILURE_PENALTY = {"latency": 10_000.0, "throughput": 100.0, "balanced": 10_000.0}
# Recent failure rate at which a candidate is skipped while others exist.
CIRCUIT_OPEN_RATE = 0.5
CIRCUIT_MIN_SAMPLES = 3
# Model larger than the provider's GPU memory: likely CPU offload.
OVERSIZE_PENALTY_MS = 5_000.0


def normalize_endpoint(path: str | None) -> str | None:
    """Upstream path a request is served on: bare OpenAI routes gain /v1."""
    if not path:
        return None
    if path.startswith("/v1/") or path.startswith("/api/"):
        return path
    return "/v1" + path


@dataclass(slots=True)
class Candidate:
    provider: Provider
    model: str
    backend_model: str
    is_hot: bool = False
    startup_ms: float | None = None
    tokens_per_second: float | None = None
    rtt_ms: float | None = None
    ttft_ms: float | None = None
    failure_rate: float = 0.0
    circuit_open: bool = False
    oversize: bool = False
    missing_endpoints: frozenset[str] = frozenset()
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
            "hot": self.is_hot,
            "startup_ms": self.startup_ms,
            "tokens_per_second": self.tokens_per_second,
            "rtt_ms": self.rtt_ms,
            "ttft_ms": self.ttft_ms,
            "failure_rate": round(self.failure_rate, 4),
            "circuit_open": self.circuit_open,
            "oversize": self.oversize,
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
    # Newest first: keep the first row seen per (provider, model, protocol)
    # and per (provider, model) for protocol-agnostic routing.
    benchmarks: dict[tuple[int, str, str | None], BenchmarkResult] = {}
    for b in await db.get_all_benchmark_results():
        benchmarks.setdefault((b.provider_id, b.model_name, b.protocol), b)
        benchmarks.setdefault((b.provider_id, b.model_name, None), b)
    fallbacks = await db.get_all_model_fallbacks()
    return RoutingInputs(providers, models, addresses, benchmarks, fallbacks)


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


def _missing_endpoints(provider: Provider) -> frozenset[str]:
    """Paths the last probe found missing on every probed URL."""
    caps = provider.capabilities or {}
    endpoints = caps.get("endpoints") or []
    missing = {e.get("path") for e in endpoints if e.get("state") == MISSING}
    supported = {e.get("path") for e in endpoints if e.get("state") in SUPPORTED}
    return frozenset(p for p in missing - supported if p)


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


def score(c: Candidate) -> None:
    """Fill *c.scores* and *c.queue_weight* for every mode. Lower is better."""
    if c.is_hot:
        cold = c.ttft_ms if c.ttft_ms is not None else 0.0
        cold_bal = cold
    elif c.startup_ms is not None:
        cold = cold_bal = c.startup_ms
    elif c.ttft_ms is not None:
        cold = cold_bal = c.ttft_ms
    else:
        cold = UNKNOWN_STARTUP_MS
        cold_bal = BALANCED_UNKNOWN_STARTUP_MS
    rtt = c.rtt_ms or 0.0
    oversize = OVERSIZE_PENALTY_MS if c.oversize else 0.0
    tps = c.tokens_per_second
    gen_ms = NOMINAL_TOKENS / (tps if tps else UNKNOWN_TPS) * 1000.0
    f = c.failure_rate
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


@dataclass(slots=True)
class RoutingTable:
    version: int = 0
    built_at: float = 0.0
    build_ms: float = 0.0
    reasons: tuple[str, ...] = ()
    # (model, protocol) -> every online candidate, in database order.
    all_candidates: dict[tuple[str, str | None], tuple[Candidate, ...]] = field(
        default_factory=dict
    )
    # (model, protocol, mode) -> usable candidates, best static score first.
    ranked: dict[tuple[str, str | None, str], tuple[Candidate, ...]] = field(
        default_factory=dict
    )
    chains: dict[str, tuple[str, ...]] = field(default_factory=dict)
    chaos: dict[str | None, tuple[tuple[Provider, tuple[str, ...]], ...]] = field(
        default_factory=dict
    )
    backend_names: dict[tuple[int, str], str] = field(default_factory=dict)

    def chain(self, model: str) -> tuple[str, ...]:
        return self.chains.get(model) or (model,)

    def backend_name(self, provider_id: int, model: str) -> str:
        return self.backend_names.get((provider_id, model), model)

    def to_dict(self) -> dict[str, Any]:
        models: dict[str, Any] = {}
        for (model, protocol), cands in sorted(
            self.all_candidates.items(), key=lambda kv: (kv[0][0], kv[0][1] or "")
        ):
            entry = models.setdefault(model, {})
            entry[protocol or "any"] = {
                "candidates": [c.to_dict() for c in cands],
                "ranked": {
                    mode: [
                        c.provider.name
                        for c in self.ranked.get((model, protocol, mode), ())
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
    grouped: dict[tuple[str, str | None], list[Candidate]] = {}
    backend_names: dict[tuple[int, str], str] = {}

    for p in online:
        pid = p.id
        assert pid is not None
        models = inputs.models.get(pid, [])
        hot = hot_models(pid)
        live_ids = [
            a.id
            for a in inputs.addresses.get(pid, [])
            if a.is_live and a.id is not None
        ]
        rtt = stats.rtt_ms(pid, live_ids or None) if stats else None
        missing = _missing_endpoints(p)
        gpu_bytes = p.gpu_ram_bytes
        for m in models:
            backend = m.backend_name
            backend_names[(pid, m.name)] = backend
            if m.raw_name:
                backend_names[(pid, m.raw_name)] = backend
            ms = stats.model(pid, m.name) if stats else None
            failure = ms.failures.current(now) if ms else 0.0
            circuit = bool(
                ms
                and ms.failures.samples >= CIRCUIT_MIN_SAMPLES
                and failure >= CIRCUIT_OPEN_RATE
            )
            is_hot = m.name in hot or bool(stats and stats.is_warm(pid, m.name, now))
            observed_tps = ms.tokens_per_second.value if ms else None
            ttft = ms.ttft_ms.value if ms else None
            oversize = bool(gpu_bytes and m.size and m.size > gpu_bytes)
            keys = {m.name}
            if m.raw_name:
                keys.add(m.raw_name)
            for protocol in PROTOCOLS:
                if not _serves(p, protocol):
                    continue
                bench = inputs.benchmarks.get((pid, m.name, protocol))
                c = Candidate(
                    provider=p,
                    model=m.name,
                    backend_model=backend,
                    is_hot=is_hot,
                    startup_ms=bench.startup_time_ms if bench else None,
                    tokens_per_second=observed_tps
                    or (bench.tokens_per_second if bench else None),
                    rtt_ms=rtt,
                    ttft_ms=ttft,
                    failure_rate=failure,
                    circuit_open=circuit,
                    oversize=oversize,
                    missing_endpoints=missing,
                )
                score(c)
                for key in keys:
                    bucket = grouped.setdefault((key, protocol), [])
                    if all(x.provider.id != pid for x in bucket):
                        bucket.append(c)

    for key, cands in grouped.items():
        table.all_candidates[key] = tuple(cands)
        usable = [c for c in cands if not c.circuit_open] or cands
        for mode in MODES:
            table.ranked[(key[0], key[1], mode)] = tuple(
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
