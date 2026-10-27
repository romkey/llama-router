from __future__ import annotations

import asyncio
import hashlib

import pytest
from starlette.requests import Request

from llama_router import request_logger
from llama_router.auth import key_hash, routing_preferences_from_request
from llama_router.background import drain
from llama_router.database import Database
from llama_router.models import (
    BenchmarkResult,
    ProviderModel,
    ProviderStatus,
    ProviderProtocol,
)
from llama_router.provider_manager import ProviderManager
from llama_router.router import Router, RoutingPreferences
from llama_router.routing_stats import WARM_WINDOW_S, ProviderStats
from llama_router.routing_table import fallback_chains


class FakeClock:
    def __init__(self) -> None:
        self.t = 1000.0

    def __call__(self) -> float:
        return self.t


async def _provider(db: Database, name: str, models: list[str], **kw):
    p = await db.add_provider(name, kw.pop("ptype", ProviderProtocol.OLLAMA), **kw)
    await db.update_provider_status(p.id, ProviderStatus.IDLE)
    await db.set_provider_models(
        p.id, [ProviderModel(provider_id=p.id, name=m) for m in models]
    )
    return p


async def _bench(db: Database, pid: int, model: str, startup: float, tps: float):
    await db.save_benchmark(
        BenchmarkResult(
            provider_id=pid,
            model_name=model,
            startup_time_ms=startup,
            tokens_per_second=tps,
        )
    )


def _prefs(mode: str, **kw) -> RoutingPreferences:
    return RoutingPreferences(mode=mode, **kw)


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock()


@pytest.fixture
def pm(db: Database, clock: FakeClock) -> ProviderManager:
    manager = ProviderManager(db)
    manager.stats = ProviderStats(clock=clock)
    return manager


async def test_modes_pick_different_providers(db: Database, pm: ProviderManager):
    quick = await _provider(db, "quick-start", ["m"])
    fast = await _provider(db, "fast-gen", ["m"])
    await _bench(db, quick.id, "m", startup=100, tps=10)
    await _bench(db, fast.id, "m", startup=900, tps=200)
    rt = Router(db, pm)

    lat = await rt.route("m", "/v1/chat/completions", preferences=_prefs("latency"))
    thr = await rt.route("m", "/v1/chat/completions", preferences=_prefs("throughput"))
    bal = await rt.route("m", "/v1/chat/completions", preferences=_prefs("balanced"))
    assert lat.provider.id == quick.id
    assert thr.provider.id == fast.id
    # 900 + 256/200 s beats 100 + 256/10 s for a typical response.
    assert bal.provider.id == fast.id


async def test_route_does_no_database_work_once_built(
    db: Database, pm: ProviderManager, monkeypatch
):
    p = await _provider(db, "srv", ["m"])
    await db.set_model_fallback("big", "m")
    rt = Router(db, pm)
    await rt.rebuild()

    calls = 0

    def spy(name):
        orig = getattr(db, name)

        async def wrapped(*a, **kw):
            nonlocal calls
            calls += 1
            return await orig(*a, **kw)

        return wrapped

    for name in ("_select_all", "_select_one", "_execute", "_execute_rowcount"):
        monkeypatch.setattr(db, name, spy(name))

    for _ in range(50):
        r = await rt.route("big", "/api/chat", preferences=_prefs("latency"))
        assert r.provider.id == p.id and r.resolved_model == "m"
    assert calls == 0


async def test_backend_name_comes_from_table(db: Database, pm: ProviderManager):
    p = await db.add_provider("srv")
    await db.update_provider_status(p.id, ProviderStatus.IDLE)
    await db.set_provider_models(
        p.id,
        [ProviderModel(provider_id=p.id, name="clean", raw_name="reg.io/clean:1")],
    )
    rt = Router(db, pm)
    by_clean = await rt.route("clean", "/v1/chat/completions")
    by_raw = await rt.route("reg.io/clean:1", "/v1/chat/completions")
    assert by_clean.backend_model == "reg.io/clean:1"
    assert by_raw.backend_model == "reg.io/clean:1"


async def test_database_change_rebuilds_before_next_route(
    db: Database, pm: ProviderManager
):
    rt = Router(db, pm)
    assert await rt.route("new-model", "/v1/chat/completions") is None
    p = await _provider(db, "srv", ["new-model"])
    r = await rt.route("new-model", "/v1/chat/completions")
    assert r is not None and r.provider.id == p.id
    await db.update_provider_status(p.id, ProviderStatus.OFFLINE)
    assert await rt.route("new-model", "/v1/chat/completions") is None


async def test_unchanged_status_does_not_dirty_table(db: Database, pm: ProviderManager):
    p = await _provider(db, "srv", ["m"])
    rt = Router(db, pm)
    table = await rt.rebuild()
    await db.update_provider_status(p.id, ProviderStatus.IDLE)
    await rt.route("m", "/v1/chat/completions")
    assert rt.table is table


async def test_background_rebuild_coalesces_bursts(db: Database, pm: ProviderManager):
    rt = Router(db, pm, debounce_s=0.05, min_interval_s=0.0)
    await rt.start()
    try:
        start_version = rt.table.version
        p = await db.add_provider("srv")
        await db.update_provider_status(p.id, ProviderStatus.IDLE)
        for i in range(10):
            await db.set_model_fallback(f"a{i}", "m")
        await db.set_provider_models(p.id, [ProviderModel(provider_id=p.id, name="m")])
        for _ in range(100):
            await asyncio.sleep(0.02)
            if rt.table.version > start_version and not rt._wake.is_set():
                r = rt.route_with(rt.table, "a3", "/v1/chat/completions")
                if r is not None:
                    break
        r = rt.route_with(rt.table, "a3", "/v1/chat/completions")
        assert r is not None and r.provider.id == p.id
        assert rt.table.version - start_version <= 3
    finally:
        await rt.stop()


async def test_hot_model_beats_faster_cold_start(db: Database, pm: ProviderManager):
    cold = await _provider(db, "cold", ["m"])
    hot = await _provider(db, "hot", ["m"])
    await _bench(db, cold.id, "m", startup=50, tps=50)
    await _bench(db, hot.id, "m", startup=5000, tps=50)
    rt = Router(db, pm)
    assert (
        await rt.route("m", "/v1/chat/completions", preferences=_prefs("latency"))
    ).provider.id == cold.id
    pm._set_hot_models(hot.id, [{"name": "m"}])
    assert (
        await rt.route("m", "/v1/chat/completions", preferences=_prefs("latency"))
    ).provider.id == hot.id


async def test_round_trip_breaks_ties(db: Database, pm: ProviderManager):
    near = await _provider(db, "near", ["m"])
    far = await _provider(db, "far", ["m"])
    for p in (near, far):
        await _bench(db, p.id, "m", startup=200, tps=50)
    near_addr = await db.add_address(near.id, "http://near")
    far_addr = await db.add_address(far.id, "http://far")
    await db.set_address_live(near_addr.id, True)
    await db.set_address_live(far_addr.id, True)
    pm.stats.record_rtt(far.id, far_addr.id, 80.0)
    pm.stats.record_rtt(near.id, near_addr.id, 2.0)
    rt = Router(db, pm)
    assert (
        await rt.route("m", "/v1/chat/completions", preferences=_prefs("latency"))
    ).provider.id == near.id


async def test_queue_depth_still_counts(db: Database, pm: ProviderManager):
    a = await _provider(db, "a", ["m"])
    b = await _provider(db, "b", ["m"])
    await _bench(db, a.id, "m", startup=100, tps=50)
    await _bench(db, b.id, "m", startup=300, tps=50)
    rt = Router(db, pm)
    assert (
        await rt.route("m", "/v1/chat/completions", preferences=_prefs("latency"))
    ).provider.id == a.id
    pm._active_requests[a.id] = 2
    assert (
        await rt.route("m", "/v1/chat/completions", preferences=_prefs("latency"))
    ).provider.id == b.id


async def test_failing_provider_is_skipped_then_retried(
    db: Database, pm: ProviderManager, clock: FakeClock
):
    good = await _provider(db, "good", ["m"])
    bad = await _provider(db, "bad", ["m"])
    await _bench(db, bad.id, "m", startup=60, tps=500)
    await _bench(db, good.id, "m", startup=900, tps=20)
    rt = Router(db, pm)
    assert (
        await rt.route("m", "/v1/chat/completions", preferences=_prefs("latency"))
    ).provider.id == bad.id
    for _ in range(5):
        pm.stats.record_request(bad.id, "m", ok=False)
    for mode in ("latency", "throughput", "balanced"):
        assert (
            await rt.route("m", "/v1/chat/completions", preferences=_prefs(mode))
        ).provider.id == good.id
    # Pins still win over an open circuit.
    pinned = _prefs("latency", pinned_providers={"m": bad.id})
    assert (
        await rt.route("m", "/v1/chat/completions", preferences=pinned)
    ).provider.id == bad.id
    clock.t += 3600
    await rt.rebuild()
    assert (
        await rt.route("m", "/v1/chat/completions", preferences=_prefs("latency"))
    ).provider.id == bad.id


async def test_only_failing_providers_still_route(db: Database, pm: ProviderManager):
    bad = await _provider(db, "bad", ["m"])
    for _ in range(5):
        pm.stats.record_request(bad.id, "m", ok=False)
    rt = Router(db, pm)
    assert (await rt.route("m", "/v1/chat/completions")).provider.id == bad.id


async def test_observed_speed_overrides_benchmark(db: Database, pm: ProviderManager):
    a = await _provider(db, "a", ["m"])
    b = await _provider(db, "b", ["m"])
    await _bench(db, a.id, "m", startup=100, tps=100)
    await _bench(db, b.id, "m", startup=100, tps=50)
    rt = Router(db, pm)
    assert (
        await rt.route("m", "/v1/chat/completions", preferences=_prefs("throughput"))
    ).provider.id == a.id
    pm.stats.record_request(a.id, "m", ok=True, tokens_per_second=10.0)
    assert (
        await rt.route("m", "/v1/chat/completions", preferences=_prefs("throughput"))
    ).provider.id == b.id


async def test_recent_success_counts_as_warm(
    db: Database, pm: ProviderManager, clock: FakeClock
):
    cold = await _provider(db, "cold", ["m"])
    warm = await _provider(db, "warm", ["m"])
    await _bench(db, cold.id, "m", startup=100, tps=50)
    await _bench(db, warm.id, "m", startup=4000, tps=50)
    pm.stats.record_request(warm.id, "m", ok=True)
    rt = Router(db, pm)
    assert (
        await rt.route("m", "/v1/chat/completions", preferences=_prefs("latency"))
    ).provider.id == warm.id
    clock.t += WARM_WINDOW_S + 1
    await rt.rebuild()
    assert (
        await rt.route("m", "/v1/chat/completions", preferences=_prefs("latency"))
    ).provider.id == cold.id


async def test_endpoint_missing_on_probe_is_avoided(db: Database, pm: ProviderManager):
    chat_only = await _provider(db, "chat-only", ["m"], ptype=ProviderProtocol.OPENAI)
    full = await _provider(db, "full", ["m"], ptype=ProviderProtocol.OPENAI)
    await _bench(db, chat_only.id, "m", startup=60, tps=500)
    await _bench(db, full.id, "m", startup=900, tps=10)
    await db.set_provider_capabilities(
        chat_only.id,
        {
            "endpoints": [
                {"path": "/v1/audio/speech", "state": "missing"},
                {"path": "/v1/chat/completions", "state": "available"},
            ]
        },
    )
    rt = Router(db, pm)
    speech = await rt.route("m", "/v1/audio/speech")
    chat = await rt.route("m", "/chat/completions")
    assert speech.provider.id == full.id
    assert chat.provider.id == chat_only.id


def test_fallback_chains_stop_at_cycles_and_depth():
    chains = fallback_chains({"a": "b", "b": "c", "c": "a", "x": "y"})
    assert chains["a"] == ("a", "b", "c")
    assert chains["x"] == ("x", "y")
    long = {f"m{i}": f"m{i + 1}" for i in range(30)}
    assert len(fallback_chains(long)["m0"]) == 11


async def test_chaos_uses_table(db: Database, pm: ProviderManager):
    p = await _provider(db, "srv", ["m1", "m2"])
    rt = Router(db, pm)
    r = await rt.route("anything", "/v1/chat/completions", preferences=_prefs("chaos"))
    assert r.provider.id == p.id and r.resolved_model in {"m1", "m2"}


async def test_dashboard_routing_table_endpoint(db: Database, pm: ProviderManager):
    from fastapi.testclient import TestClient

    from llama_router.dashboard import deps as dash_deps
    from llama_router.dashboard.app import app as dash_app

    await _provider(db, "srv", ["m"])
    rt = Router(db, pm)
    pm.attach_router(rt)
    await rt.rebuild()
    dash_deps.init(db, pm)
    with TestClient(dash_app) as client:
        data = client.get("/api/routing-table").json()
    assert data["models"]["m"]["/v1/chat/completions"]["ranked"]["balanced"] == ["srv"]


# --- request sampling ---


async def _consume(it):
    out = []
    async for chunk in it:
        out.append(chunk)
    return out


def _request() -> Request:
    return Request(
        {"type": "http", "method": "POST", "path": "/api/chat", "headers": []}
    )


async def test_stream_logger_records_ttft_and_speed(db: Database, pm: ProviderManager):
    p = await _provider(db, "srv", ["m"])
    request_logger.attach_stats(pm.stats)
    try:

        async def upstream():
            yield b'{"message":{"content":"hi"},"done":false}\n'
            yield b'{"done":true,"eval_count":40,"eval_duration":2000000000}\n'

        import time

        logged = request_logger.StreamLogger(
            upstream(),
            db=db,
            provider=p,
            protocol="ollama",
            endpoint="/api/chat",
            request=_request(),
            model="alias",
            routed_model="m",
            request_body={"model": "m"},
            start_time=time.monotonic(),
        )
        await _consume(logged)
        await drain()
    finally:
        request_logger.attach_stats(None)
    s = pm.stats.model(p.id, "m")
    assert s is not None
    assert s.tokens_per_second.value == pytest.approx(20.0)
    assert s.ttft_ms.value is not None and s.ttft_ms.value >= 0
    logs = await db.get_request_logs(10, 0)
    assert logs and logs[0].streamed and logs[0].status == "ok"


async def test_client_errors_do_not_count_against_provider(
    db: Database, pm: ProviderManager
):
    p = await _provider(db, "srv", ["m"])
    request_logger.attach_stats(pm.stats)
    try:
        for detail in ("HTTP 400: bad", "HTTP 503: down"):
            request_logger.log_request_background(
                db,
                provider=p,
                protocol="ollama",
                endpoint="/api/chat",
                model="m",
                duration_ms=1.0,
                status="error",
                error_detail=detail,
                source_ip="test",
            )
        await drain()
    finally:
        request_logger.attach_stats(None)
    assert pm.stats.model(p.id, "m").failures.samples == 1


# --- API key cache ---


def _key_request(key: str) -> Request:
    return Request(
        {
            "type": "http",
            "method": "POST",
            "path": "/api/chat",
            "headers": [(b"api-key", key.encode())],
        }
    )


async def test_api_key_is_cached_and_invalidated(db: Database, monkeypatch):
    key = "lrk_test-key-value"
    key_id = await db.create_api_key("lrk_test", key_hash(key), "throughput", True)
    await db.set_allow_unauthenticated(False)

    prefs = await routing_preferences_from_request(db, _key_request(key))
    assert prefs.mode == "throughput"

    lookups = 0
    orig = db.lookup_api_key

    async def counting(h):
        nonlocal lookups
        lookups += 1
        return await orig(h)

    monkeypatch.setattr(db, "lookup_api_key", counting)
    for _ in range(5):
        again = await routing_preferences_from_request(db, _key_request(key))
        assert again.mode == "throughput"
    assert lookups == 0

    p = await db.add_provider("srv")
    await db.set_api_key_model_pin(key_id, "m", p.id)
    pinned = await routing_preferences_from_request(db, _key_request(key))
    assert pinned.pinned_providers == {"m": p.id}
    assert lookups > 0
    await drain()


async def test_unknown_key_is_rejected(db: Database):
    from fastapi import HTTPException

    await db.set_allow_unauthenticated(False)
    with pytest.raises(HTTPException) as err:
        await routing_preferences_from_request(db, _key_request("lrk_nope"))
    assert err.value.status_code == 401
    # Turning on open access applies at once.
    await db.set_allow_unauthenticated(True)
    assert await routing_preferences_from_request(db, _key_request("lrk_nope")) is None


async def test_legacy_sha256_keys_still_work(db: Database):
    key = "lrk_legacy"
    await db.create_api_key(
        "lrk_legacy", hashlib.sha256(key.encode()).hexdigest(), "latency", False
    )
    await db.set_allow_unauthenticated(False)
    prefs = await routing_preferences_from_request(db, _key_request(key))
    assert prefs.mode == "latency" and prefs.allow_fallback is False


async def test_dead_address_round_trips_are_ignored(db: Database, pm: ProviderManager):
    p = await _provider(db, "srv", ["m"])
    addr = await db.add_address(p.id, "http://srv")
    pm.stats.record_rtt(p.id, addr.id, 1.0)
    rt = Router(db, pm)
    table = await rt.rebuild()
    assert table.all_candidates[("m", "/v1/chat/completions")][0].rtt_ms is None
    await db.set_address_live(addr.id, True)
    table = await rt.rebuild()
    assert table.all_candidates[("m", "/v1/chat/completions")][
        0
    ].rtt_ms == pytest.approx(1.0)


async def test_raw_name_requests_use_discovered_name_for_stats(
    db: Database, pm: ProviderManager
):
    p = await db.add_provider("srv")
    await db.update_provider_status(p.id, ProviderStatus.IDLE)
    await db.set_provider_models(
        p.id,
        [ProviderModel(provider_id=p.id, name="clean", raw_name="reg.io/clean:1")],
    )
    rt = Router(db, pm)
    r = await rt.route("reg.io/clean:1", "/v1/chat/completions")
    assert r.resolved_model == "reg.io/clean:1"
    assert r.model_key == "clean"


async def test_key_deleted_during_lookup_is_not_cached(db: Database, monkeypatch):
    key = "lrk_racing"
    key_id = await db.create_api_key("lrk_racing", key_hash(key), "latency", True)
    await db.set_allow_unauthenticated(False)
    orig = db.lookup_api_key
    deleted = False

    async def lookup_then_delete(h):
        nonlocal deleted
        row = await orig(h)
        if row and not deleted:
            deleted = True
            await db.delete_api_key(key_id)
        return row

    monkeypatch.setattr(db, "lookup_api_key", lookup_then_delete)
    # The in-flight request still gets its answer...
    assert await routing_preferences_from_request(db, _key_request(key)) is not None
    # ...but the deleted key must not keep working from the cache.
    from fastapi import HTTPException

    with pytest.raises(HTTPException) as err:
        await routing_preferences_from_request(db, _key_request(key))
    assert err.value.status_code == 401
