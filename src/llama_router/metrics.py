"""Prometheus metrics for ``GET /metrics``.

Renders the text exposition format by hand: the format is a few lines of
plain text, which is less than the ``prometheus_client`` dependency and its
registry would add. Every value comes from state the router already keeps
(provider rows, the in-memory routing stats, the worker hub and the
auto-benchmarker), so a scrape costs what the dashboard's status call costs.

Format reference: https://prometheus.io/docs/instrumenting/exposition_formats/
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from typing import Any

from . import __version__
from .models import ProviderInfo, ProviderStatus

CONTENT_TYPE = "text/plain; version=0.0.4; charset=utf-8"
PREFIX = "llama_router_"


def _escape(value: object) -> str:
    text = str(value)
    return text.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


def _fmt(value: float | int | bool | None) -> str:
    if value is None:
        return "NaN"
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, int):
        return str(value)
    if math.isnan(value):
        return "NaN"
    if math.isinf(value):
        return "+Inf" if value > 0 else "-Inf"
    return repr(float(value))


class Family:
    """One metric family: its header lines and samples, in exposition order."""

    def __init__(self, name: str, kind: str, help_text: str) -> None:
        self.name = PREFIX + name
        self.kind = kind
        self.help = help_text
        self.samples: list[tuple[dict[str, object], float | int | bool | None]] = []

    def add(self, value: float | int | bool | None, **labels: object) -> None:
        self.samples.append((labels, value))

    def lines(self) -> Iterable[str]:
        yield f"# HELP {self.name} {_escape(self.help)}"
        yield f"# TYPE {self.name} {self.kind}"
        for labels, value in self.samples:
            if labels:
                body = ",".join(f'{k}="{_escape(v)}"' for k, v in labels.items())
                yield f"{self.name}{{{body}}} {_fmt(value)}"
            else:
                yield f"{self.name} {_fmt(value)}"


def render(families: Iterable[Family]) -> str:
    out: list[str] = []
    for fam in families:
        out.extend(fam.lines())
    return "\n".join(out) + "\n"


def _provider_labels(info: ProviderInfo) -> dict[str, object]:
    p = info.provider
    return {
        "provider": p.name,
        "provider_id": p.id,
        "protocol": p.protocol.value,
        "locality": p.locality,
    }


def _latest_benchmarks(info: ProviderInfo) -> dict[tuple[str, str], Any]:
    """Newest benchmark per (model, protocol); rows arrive newest first."""
    latest: dict[tuple[str, str], Any] = {}
    for b in info.benchmarks:
        key = (b.model_name, (b.protocol or info.provider.protocol.value).lower())
        latest.setdefault(key, b)
    return latest


def collect(
    infos: list[ProviderInfo],
    stats: Any,
    *,
    router: Any = None,
    hub: Any = None,
    worker_link: dict[str, Any] | None = None,
    auto_benchmark: Any = None,
) -> list[Family]:
    """Build every metric family from the router's current state.

    *stats* is the :class:`~llama_router.routing_stats.ProviderStats`;
    *router*, *hub* and *auto_benchmark* are duck-typed so the collector
    needs nothing started that the dashboard does not already have.
    """
    info_fam = Family("info", "gauge", "llama-router version.")
    info_fam.add(1, version=__version__)

    providers = Family("providers", "gauge", "Providers by status.")
    counts = dict.fromkeys(ProviderStatus, 0)
    for info in infos:
        counts[info.provider.status] += 1
    for status, n in counts.items():
        providers.add(n, status=status.value)

    models = Family("models", "gauge", "Distinct model names across online providers.")
    names = {
        m.name
        for info in infos
        if info.provider.status is not ProviderStatus.OFFLINE
        for m in info.models
    }
    models.add(len(names))

    up = Family("provider_up", "gauge", "1 when the provider is not offline.")
    active = Family(
        "provider_active_requests", "gauge", "Requests in flight on the provider."
    )
    model_count = Family(
        "provider_models", "gauge", "Models the provider currently offers."
    )
    health = Family(
        "provider_health_score",
        "gauge",
        "0 (failing everything) to 100 (healthy), from recent requests and probes.",
    )
    failure = Family(
        "provider_request_failure_rate",
        "gauge",
        "Share of recent requests that failed for provider-wide reasons (0..1).",
    )
    probe_failure = Family(
        "provider_probe_failure_rate",
        "gauge",
        "Share of recent health checks the provider failed (0..1).",
    )
    circuit = Family(
        "provider_circuit_open",
        "gauge",
        "1 while the provider is skipped for failing too many recent requests.",
    )
    busy = Family(
        "provider_rate_limited_seconds",
        "gauge",
        "Seconds left of the cooldown after the provider answered HTTP 429.",
    )
    addr_up = Family(
        "address_up", "gauge", "1 when the address answered its last health check."
    )
    addr_rtt = Family(
        "address_rtt_milliseconds",
        "gauge",
        "Moving average health-check round trip to the address.",
    )
    addr_checks = Family(
        "address_health_checks_total", "counter", "Health checks of the address."
    )
    addr_failures = Family(
        "address_health_check_failures_total",
        "counter",
        "Health checks of the address that failed.",
    )
    bench_tps = Family(
        "benchmark_tokens_per_second",
        "gauge",
        "Latest benchmark: generation speed of the model on the provider.",
    )
    bench_startup = Family(
        "benchmark_startup_milliseconds",
        "gauge",
        "Latest benchmark: cold start of the model on the provider.",
    )
    worker_conn = Family(
        "worker_connected", "gauge", "1 while the worker's socket is open."
    )
    worker_inflight = Family(
        "worker_in_flight", "gauge", "Requests in flight over the worker's socket."
    )
    worker_jobs = Family(
        "worker_jobs_total",
        "counter",
        "Requests sent to the worker since start-up, across reconnects.",
    )

    by_id: dict[int, ProviderInfo] = {}
    for info in infos:
        p = info.provider
        if p.id is None:
            continue
        by_id[p.id] = info
        labels = _provider_labels(info)
        up.add(p.status is not ProviderStatus.OFFLINE, **labels)
        active.add(info.active_requests, **labels)
        model_count.add(len(info.models), **labels)
        h = stats.provider_health(p.id)
        health.add(h.score, **labels)
        failure.add(h.request_failure_rate, **labels)
        probe_failure.add(h.probe_failure_rate, **labels)
        circuit.add(h.circuit_open, **labels)
        busy.add(stats.busy_for(p.id), **labels)
        for a in info.addresses:
            alabels = {**labels, "url": a.url}
            addr_up.add(a.is_live, **alabels)
            tally = stats.tally(p.id, a.id) if a.id is not None else None
            if tally is not None:
                addr_rtt.add(tally.ewma.value, **alabels)
                addr_checks.add(tally.checks, **alabels)
                addr_failures.add(tally.failures, **alabels)
        for (model, protocol), b in _latest_benchmarks(info).items():
            blabels = {**labels, "model": model, "benchmark_protocol": protocol}
            if b.tokens_per_second is not None:
                bench_tps.add(b.tokens_per_second, **blabels)
            if b.startup_time_ms is not None:
                bench_startup.add(b.startup_time_ms, **blabels)
        if p.is_worker and not p.is_worker_child:
            # One series per socket; the providers it shares count toward it.
            live = info.worker or {}
            wlabels = {"worker": p.worker_name, "provider_id": p.id}
            worker_conn.add(bool(live.get("connected")), **wlabels)
            worker_inflight.add(int(live.get("in_flight") or 0), **wlabels)
            if hub is not None:
                jobs = int(hub.jobs_total(p.id))
            else:
                jobs = int(live.get("jobs_total") or 0)
            worker_jobs.add(jobs, **wlabels)

    requests = Family(
        "requests_total",
        "counter",
        "Proxied requests since start-up, by provider, model and outcome. "
        "Requests the client got wrong are not counted against the provider.",
    )
    live_tps = Family(
        "model_tokens_per_second",
        "gauge",
        "Moving average generation speed from real requests.",
    )
    live_ttft = Family(
        "model_time_to_first_token_milliseconds",
        "gauge",
        "Moving average time to first token from real streamed requests.",
    )
    live_duration = Family(
        "model_request_duration_milliseconds",
        "gauge",
        "Moving average dispatch-to-last-byte time of successful requests.",
    )
    for pid, model, cls, s in sorted(stats.model_entries(), key=lambda e: e[:3]):
        owner = by_id.get(pid)
        if owner is None:
            continue
        labels = {**_provider_labels(owner), "model": model, "endpoint_class": cls}
        requests.add(s.requests - s.errors, **labels, outcome="ok")
        requests.add(s.errors, **labels, outcome="error")
        if s.tokens_per_second.value is not None:
            live_tps.add(s.tokens_per_second.value, **labels)
        if s.ttft_ms.value is not None:
            live_ttft.add(s.ttft_ms.value, **labels)
        if s.duration_ms.value is not None:
            live_duration.add(s.duration_ms.value, **labels)

    families = [
        info_fam,
        providers,
        models,
        up,
        active,
        model_count,
        health,
        failure,
        probe_failure,
        circuit,
        busy,
        addr_up,
        addr_rtt,
        addr_checks,
        addr_failures,
        bench_tps,
        bench_startup,
        requests,
        live_tps,
        live_ttft,
        live_duration,
        worker_conn,
        worker_inflight,
        worker_jobs,
    ]

    table = getattr(router, "table", None) if router is not None else None
    if table is not None:
        version = Family(
            "routing_table_version", "gauge", "Rebuilds of the routing table."
        )
        version.add(table.version)
        built = Family(
            "routing_table_built_timestamp_seconds",
            "gauge",
            "When the routing table was last rebuilt (Unix time).",
        )
        built.add(table.built_at)
        build_ms = Family(
            "routing_table_build_milliseconds",
            "gauge",
            "How long the last routing table rebuild took.",
        )
        build_ms.add(table.build_ms)
        families += [version, built, build_ms]
        if table.groups:
            group_up = Family(
                "model_group_available",
                "gauge",
                "1 while at least one member of the model group can be served.",
            )
            for name in sorted(table.groups):
                group_up.add(table.group_reach.get(name, (False, False))[0], group=name)
            families.append(group_up)
        group_routes = getattr(router, "group_routes", None)
        if group_routes:
            routed = Family(
                "model_group_routes_total",
                "counter",
                "Requests for a model group since start-up, by the member model "
                "and provider chosen.",
            )
            for (group, model, pid), n in sorted(group_routes.items()):
                owner = by_id.get(pid)
                if owner is not None:
                    routed.add(n, **_provider_labels(owner), group=group, model=model)
            families.append(routed)
        affinity = getattr(router, "affinity", None)
        if affinity is not None:
            sessions = Family(
                "session_affinity_sessions",
                "gauge",
                "Conversations currently bound to a provider.",
            )
            sessions.add(int(affinity.stats().get("sessions") or 0))
            families.append(sessions)

    if worker_link is not None:
        link_up = Family(
            "worker_link_connected",
            "gauge",
            "1 while this router is connected to its hub as a worker.",
        )
        link_up.add(
            worker_link.get("state") == "connected",
            hub=worker_link.get("hub_url") or "",
        )
        link_jobs = Family(
            "worker_link_jobs_total",
            "counter",
            "Jobs this router has served for its hub since the link started.",
        )
        link_jobs.add(int(worker_link.get("jobs_total") or 0))
        families += [link_up, link_jobs]

    if auto_benchmark is not None:
        runs = Family(
            "auto_benchmark_runs_total",
            "counter",
            "Automatic benchmarks since start-up, by outcome.",
        )
        for outcome in ("completed", "failed", "interrupted"):
            runs.add(int(getattr(auto_benchmark, outcome, 0)), outcome=outcome)
        families.append(runs)

    return families
