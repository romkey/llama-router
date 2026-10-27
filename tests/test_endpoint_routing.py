from __future__ import annotations

import pytest

from llama_router.database import Database
from llama_router.endpoints import ENDPOINTS, UnknownEndpoint, endpoint_class
from llama_router.models import (
    BenchmarkResult,
    ProviderModel,
    ProviderStatus,
    ProviderType,
)
from llama_router.provider_manager import ProviderManager
from llama_router.router import Router, RoutingPreferences
from llama_router.routing_stats import ProviderStats
from llama_router.routing_table import TIER_FAILING, TIER_MISSING, TIER_OK

CHAT = "/v1/chat/completions"


class FakeClock:
    def __init__(self) -> None:
        self.t = 1000.0

    def __call__(self) -> float:
        return self.t


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock()


@pytest.fixture
def pm(db: Database, clock: FakeClock) -> ProviderManager:
    manager = ProviderManager(db)
    manager.stats = ProviderStats(clock=clock)
    return manager


async def _provider(db: Database, name: str, models: list[str], **kw):
    p = await db.add_provider(name, kw.pop("ptype", ProviderType.OLLAMA))
    await db.update_provider_status(p.id, ProviderStatus.IDLE)
    await db.set_provider_models(
        p.id, [ProviderModel(provider_id=p.id, name=m) for m in models]
    )
    return p


async def _bench(db, pid, model, startup, tps, protocol="ollama"):
    await db.save_benchmark(
        BenchmarkResult(
            provider_id=pid,
            model_name=model,
            protocol=protocol,
            startup_time_ms=startup,
            tokens_per_second=tps,
        )
    )


def _mode(mode: str) -> RoutingPreferences:
    return RoutingPreferences(mode=mode)


# --- endpoints are part of every route ---


async def test_unknown_endpoint_is_rejected(db: Database, pm: ProviderManager):
    await _provider(db, "a", ["m"])
    rt = Router(db, pm)
    with pytest.raises(UnknownEndpoint):
        await rt.route("m", "/api/tags")


async def test_table_is_built_for_every_endpoint(db: Database, pm: ProviderManager):
    await _provider(db, "ollama-box", ["m"])
    await _provider(db, "lcpp-box", ["m"], ptype=ProviderType.LLAMACPP)
    rt = Router(db, pm)
    table = await rt.rebuild()
    for path, (protocol, _cls) in ENDPOINTS.items():
        names = {c.provider.name for c in table.all_candidates[("m", path)]}
        if protocol == "ollama":
            assert names == {"ollama-box"}, path
        else:
            assert names == {"ollama-box", "lcpp-box"}, path
    # Ollama-native routes never reach a llama.cpp-only provider.
    r = await rt.route("m", "/api/embed")
    assert r.provider.name == "ollama-box"


async def test_probe_of_the_serving_upstream_decides(db: Database, pm: ProviderManager):
    both = await _provider(db, "both", ["m"], ptype=ProviderType.BOTH)
    await db.add_address(
        both.id, "http://box:11434", "http://box:8080", is_preferred=True
    )
    other = await _provider(db, "other", ["m"], ptype=ProviderType.LLAMACPP)
    await _bench(db, both.id, "m", startup=10, tps=500, protocol="llamacpp")
    await _bench(db, other.id, "m", startup=900, tps=10, protocol="llamacpp")
    # The Ollama URL has speech, the llama.cpp URL (which v1 calls use) does not.
    await db.set_provider_capabilities(
        both.id,
        {
            "endpoints": [
                {
                    "path": "/v1/audio/speech",
                    "state": "available",
                    "base_url": "http://box:11434",
                },
                {
                    "path": "/v1/audio/speech",
                    "state": "missing",
                    "base_url": "http://box:8080",
                },
            ]
        },
    )
    rt = Router(db, pm)
    table = await rt.rebuild()
    cand = next(
        c
        for c in table.all_candidates[("m", "/v1/audio/speech")]
        if c.provider.name == "both"
    )
    assert cand.upstream == "llamacpp"
    assert cand.endpoint_support == "missing" and cand.tier == TIER_MISSING
    assert (await rt.route("m", "/v1/audio/speech")).provider.id == other.id
    # Chat is unaffected.
    assert (await rt.route("m", CHAT, _mode("latency"))).provider.id == both.id


async def test_failing_beats_certainly_missing(db: Database, pm: ProviderManager):
    missing = await _provider(db, "missing", ["m"], ptype=ProviderType.LLAMACPP)
    failing = await _provider(db, "failing", ["m"], ptype=ProviderType.LLAMACPP)
    await db.set_provider_capabilities(
        missing.id,
        {"endpoints": [{"path": "/v1/embeddings", "state": "missing"}]},
    )
    for _ in range(5):
        pm.stats.record_request(failing.id, "m", ok=False, endpoint_class="embed")
    rt = Router(db, pm)
    table = await rt.rebuild()
    tiers = {
        c.provider.name: c.tier for c in table.all_candidates[("m", "/v1/embeddings")]
    }
    assert tiers == {"missing": TIER_MISSING, "failing": TIER_FAILING}
    assert (await rt.route("m", "/v1/embeddings")).provider.id == failing.id


async def test_stats_are_kept_per_endpoint_class(db: Database, pm: ProviderManager):
    a = await _provider(db, "a", ["m"])
    b = await _provider(db, "b", ["m"])
    for p in (a, b):
        await _bench(db, p.id, "m", startup=100, tps=50)
    # a is slow at embeddings, b is slow at chat.
    for _ in range(3):
        pm.stats.record_request(
            a.id, "m", ok=True, endpoint_class="embed", duration_ms=900
        )
        pm.stats.record_request(
            b.id, "m", ok=True, endpoint_class="embed", duration_ms=40
        )
        pm.stats.record_request(a.id, "m", ok=True, tokens_per_second=90)
        pm.stats.record_request(b.id, "m", ok=True, tokens_per_second=15)
    rt = Router(db, pm)
    for mode in ("latency", "throughput", "balanced"):
        assert (await rt.route("m", "/v1/embeddings", _mode(mode))).provider.id == b.id
        assert (await rt.route("m", "/api/embed", _mode(mode))).provider.id == b.id
    assert (await rt.route("m", CHAT, _mode("throughput"))).provider.id == a.id
    assert (await rt.route("m", CHAT, _mode("balanced"))).provider.id == a.id


async def test_model_failures_on_one_endpoint_class_stay_there(
    db: Database, pm: ProviderManager
):
    a = await _provider(db, "a", ["m"])
    await _provider(db, "b", ["m"])
    await _bench(db, a.id, "m", startup=10, tps=500)
    for _ in range(3):
        pm.stats.record_request(a.id, "m", ok=False, endpoint_class="image")
    rt = Router(db, pm)
    table = await rt.rebuild()
    by_ep = {
        path: next(
            c for c in table.all_candidates[("m", path)] if c.provider.id == a.id
        )
        for path in (CHAT, "/v1/images/generations")
    }
    assert by_ep["/v1/images/generations"].tier == TIER_FAILING
    assert by_ep[CHAT].tier == TIER_OK
    assert (await rt.route("m", CHAT, _mode("latency"))).provider.id == a.id


def test_endpoint_classes():
    assert endpoint_class("/chat/completions") == "generate"
    assert endpoint_class("/api/embeddings") == "embed"
    assert endpoint_class("/v1/images/edits") == "image"
    assert endpoint_class("/v1/audio/transcriptions") == "audio"


# --- provider health score ---


async def test_provider_wide_failures_open_its_circuit(
    db: Database, pm: ProviderManager, clock: FakeClock
):
    flaky = await _provider(db, "flaky", ["m1", "m2", "m3"])
    steady = await _provider(db, "steady", ["m3"])
    await _bench(db, flaky.id, "m3", startup=10, tps=500)
    await _bench(db, steady.id, "m3", startup=900, tps=10)
    # Two failures each on two other models: no single model trips its own
    # breaker, but the provider as a whole is failing.
    for model in ("m1", "m2"):
        for _ in range(2):
            pm.stats.record_request(flaky.id, model, ok=False)
    pm.stats.record_request(flaky.id, "m1", ok=False)
    health = pm.stats.provider_health(flaky.id)
    assert health.circuit_open and health.score < 50
    rt = Router(db, pm)
    for mode in ("latency", "throughput", "balanced"):
        assert (await rt.route("m3", CHAT, _mode(mode))).provider.id == steady.id
    out = await rt.explain("m3", CHAT, _mode("latency"))
    row = next(c for c in out["steps"][0]["candidates"] if c["provider"] == "flaky")
    assert "provider failing" in row["excluded"]
    # Failures age out and the provider is used again.
    clock.t += 3600
    await rt.rebuild()
    assert pm.stats.provider_health(flaky.id).score == 100
    assert (await rt.route("m3", CHAT, _mode("latency"))).provider.id == flaky.id


async def test_failing_provider_scores_worse_before_circuit(
    db: Database, pm: ProviderManager
):
    a = await _provider(db, "a", ["m"])
    b = await _provider(db, "b", ["m"])
    for p in (a, b):
        await _bench(db, p.id, "m", startup=100, tps=50)
    # One failure: penalised, not cut off.
    pm.stats.record_request(a.id, "other", ok=False)
    rt = Router(db, pm)
    table = await rt.rebuild()
    cands = {c.provider.name: c for c in table.all_candidates[("m", CHAT)]}
    assert cands["a"].tier == TIER_OK
    assert cands["a"].scores["latency"] > cands["b"].scores["latency"]
    assert cands["a"].health.score < 100 == cands["b"].health.score
    assert (await rt.route("m", CHAT, _mode("latency"))).provider.id == b.id


def test_health_checks_count_only_when_nothing_answers(clock: FakeClock):
    stats = ProviderStats(clock=clock)
    # One dead address of two: the provider still answered.
    stats.record_probe(1, 10, ok=True, ms=5.0)
    stats.record_probe(1, 11, ok=False, ms=5000.0)
    stats.record_check(1, ok=True)
    assert stats.provider_health(1).score == 100
    stats.record_check(1, ok=False)
    assert stats.provider_health(1).probe_failure_rate > 0
    assert stats.provider_health(1).score < 100
    # Request failures alone never open the breaker on health checks.
    assert not stats.provider_health(1).circuit_open


# --- health-check latency tally ---


def test_latency_tally(clock: FakeClock):
    stats = ProviderStats(clock=clock)
    for ms in (10.0, 20.0, 30.0, 40.0, 100.0):
        stats.record_probe(1, 7, ok=True, ms=ms)
    stats.record_probe(1, 7, ok=False, ms=5000.0)
    t = stats.tally(1, 7).to_dict()
    assert t["checks"] == 6 and t["failures"] == 1
    assert t["consecutive_failures"] == 1
    assert t["min_ms"] == 10.0 and t["max_ms"] == 100.0
    assert t["p50_ms"] == 30.0 and t["p95_ms"] == 100.0
    assert t["mean_ms"] == 40.0 and t["last_ms"] == 100.0
    # Routing uses the moving average of successful checks only.
    assert stats.rtt_ms(1, [7]) == pytest.approx(stats.tally(1, 7).ewma.value)


async def test_provider_checks_feed_the_tally(db: Database, pm: ProviderManager):
    p = await db.add_provider("box", ProviderType.OLLAMA)
    good = await db.add_address(p.id, "http://good", is_preferred=True)
    bad = await db.add_address(p.id, "http://bad")
    provider = await db.get_provider(p.id)

    async def probe_once(prov, addr):
        if addr.url == "http://bad":
            raise RuntimeError("connection refused")
        return True

    pm._probe_address_once = probe_once
    assert await pm._probe_address(provider, good) is True
    with pytest.raises(RuntimeError):
        await pm._probe_address(provider, bad)
    assert pm.stats.tally(p.id, good.id).checks == 1
    assert pm.stats.tally(p.id, good.id).failures == 0
    assert pm.stats.tally(p.id, bad.id).failures == 1


async def test_dashboard_shows_provider_health(db: Database, pm: ProviderManager):
    from fastapi.testclient import TestClient

    from llama_router.dashboard import deps as dash_deps
    from llama_router.dashboard.app import app as dash_app

    p = await _provider(db, "box", ["m"])
    addr = await db.add_address(p.id, "http://box", is_preferred=True)
    pm.stats.record_probe(p.id, addr.id, ok=True, ms=12.0)
    rt = Router(db, pm)
    await rt.rebuild()
    dash_deps.init(db, pm)
    with TestClient(dash_app) as client:
        page = client.get("/").text
        table = client.get("/api/routing-table").json()
    assert "Provider health" in page and "p50 12 ms" in page
    assert "Balanced (default)" in page
    assert table["provider_health"][str(p.id)]["score"] == 100


async def test_unmeasured_service_time_is_not_instant(
    db: Database, pm: ProviderManager
):
    measured = await _provider(db, "measured", ["m"])
    await _provider(db, "unmeasured", ["m"])
    for _ in range(3):
        pm.stats.record_request(
            measured.id, "m", ok=True, endpoint_class="embed", duration_ms=300
        )
    rt = Router(db, pm)
    for mode in ("latency", "throughput", "balanced"):
        r = await rt.route("m", "/v1/embeddings", _mode(mode))
        assert r.provider.id == measured.id, mode


def test_model_and_provider_failures_are_not_double_counted(clock: FakeClock):
    from llama_router.routing_table import combined_failure_rate

    stats = ProviderStats(clock=clock)
    # A single-model provider: every outcome lands in both rates.
    for ok in (False, True, True, False, True):
        stats.record_request(1, "m", ok=ok)
    model_rate = stats.model(1, "m").failures.current(clock())
    health = stats.provider_health(1)
    assert model_rate == pytest.approx(health.request_failure_rate)
    assert combined_failure_rate(model_rate, health) == pytest.approx(model_rate)
    # Whole-check failures still add on top.
    stats.record_check(1, ok=False)
    assert combined_failure_rate(model_rate, stats.provider_health(1)) > model_rate
