from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone

import pytest

from llama_router.auto_benchmark import AutoBenchmarker
from llama_router.database import Database
from llama_router.llamacpp_client import _chat_metrics
from llama_router.models import (
    BenchmarkResult,
    ProviderModel,
    ProviderStatus,
    ProviderType,
)
from llama_router.provider_manager import ProviderManager
from llama_router.routing_table import latest_benchmarks


async def _provider(db: Database, name: str, models: list[str], **kw):
    status = kw.pop("status", ProviderStatus.IDLE)
    p = await db.add_provider(name, kw.pop("ptype", ProviderType.OLLAMA))
    await db.update_provider_status(p.id, status)
    await db.set_provider_models(
        p.id, [ProviderModel(provider_id=p.id, name=m) for m in models]
    )
    return p


class FakeBench:
    """Stands in for ProviderManager.benchmark_provider."""

    def __init__(self, gate: asyncio.Event | None = None, fail: set[str] = ()):
        self.calls: list[tuple[int, str, str, str]] = []
        self.gate = gate
        self.fail = set(fail)
        self.started = asyncio.Event()

    async def __call__(self, pid, model, benchmark_api=None, *, source="manual"):
        self.calls.append((pid, model, benchmark_api, source))
        self.started.set()
        if self.gate is not None:
            await self.gate.wait()
        if model in self.fail:
            raise RuntimeError(f"{model} is broken")
        return BenchmarkResult(
            provider_id=pid,
            model_name=model,
            protocol=benchmark_api,
            tokens_per_second=42.0,
        )


@pytest.fixture
def pm(db: Database) -> ProviderManager:
    manager = ProviderManager(db)
    # Everyone has been idle for an hour.
    manager._started_at -= 3600
    return manager


async def _drain(ab: AutoBenchmarker) -> None:
    while ab._runs:
        await asyncio.gather(*list(ab._runs.values()), return_exceptions=True)


async def test_plan_orders_missing_then_oldest(db: Database, pm: ProviderManager):
    a = await _provider(db, "a", ["fresh", "old", "older", "new"])
    await _provider(db, "down", ["x"], status=ProviderStatus.OFFLINE)
    for model in ("fresh", "old", "older"):
        await db.save_benchmark(
            BenchmarkResult(provider_id=a.id, model_name=model, protocol="ollama")
        )
    now = datetime.now(timezone.utc)
    ages = {"fresh": 1, "old": 10, "older": 30}

    async def aged():
        rows = await Database.get_all_benchmark_results(db)
        for r in rows:
            r.created_at = now - timedelta(days=ages[r.model_name])
        return rows

    db.get_all_benchmark_results = aged
    ab = AutoBenchmarker(db, pm)
    plan = await ab.plan()
    assert [(j.model, j.reason) for j in plan] == [
        ("new", "missing"),
        ("older", "stale"),
        ("old", "stale"),
    ]


async def test_plan_follows_each_providers_apis(db: Database, pm: ProviderManager):
    await _provider(db, "lcpp", ["m"], ptype=ProviderType.LLAMACPP)
    await _provider(db, "both", ["m"], ptype=ProviderType.BOTH)
    ab = AutoBenchmarker(db, pm)
    jobs = {(j.provider_name, j.api) for j in await ab.plan()}
    assert jobs == {("lcpp", "llamacpp"), ("both", "ollama"), ("both", "llamacpp")}


async def test_runs_only_when_idle(db: Database, pm: ProviderManager):
    p = await _provider(db, "a", ["m1", "m2"])
    fake = FakeBench()
    pm.benchmark_provider = fake
    ab = AutoBenchmarker(db, pm)

    pm._last_activity[p.id] = pm._started_at + 3600  # just served a request
    await ab.run_once()
    await _drain(ab)
    assert fake.calls == []

    pm._last_activity[p.id] -= 3600
    await ab.run_once()
    await _drain(ab)
    assert fake.calls == [
        (p.id, "m1", "ollama", "auto"),
        (p.id, "m2", "ollama", "auto"),
    ]
    assert ab.completed == 2

    pm.acquire(p.id)  # busy right now
    pm._last_activity[p.id] -= 3600
    await ab.run_once()
    await _drain(ab)
    assert len(fake.calls) == 2
    pm.release(p.id)


async def test_real_request_cancels_benchmark(db: Database, pm: ProviderManager):
    p = await _provider(db, "a", ["m1", "m2"])
    fake = FakeBench(gate=asyncio.Event())
    pm.benchmark_provider = fake
    ab = AutoBenchmarker(db, pm)
    await ab.run_once()
    await asyncio.wait_for(fake.started.wait(), 1)
    assert [j["model"] for j in (await ab.status())["running"]] == ["m1"]

    pm.acquire(p.id)  # a real request arrives
    await _drain(ab)
    pm.release(p.id)
    assert ab.interrupted == 1 and ab.completed == 0
    # The provider's run stopped; m2 was not started.
    assert [c[1] for c in fake.calls] == ["m1"]
    assert (await ab.status())["recent"][0]["outcome"] == "interrupted"
    # Not backed off: it is due again next time the provider is idle.
    assert {j.model for j in await ab.plan()} == {"m1", "m2"}


async def test_failures_back_off(db: Database, pm: ProviderManager):
    await _provider(db, "a", ["bad", "good"])
    fake = FakeBench(fail={"bad"})
    pm.benchmark_provider = fake
    ab = AutoBenchmarker(db, pm)
    await ab.run_once()
    await _drain(ab)
    assert ab.failed == 1 and ab.completed == 1
    status = await ab.status()
    assert status["backed_off"] == 1
    assert "bad is broken" in status["recent"][-1]["error"]
    assert "bad" not in {j.model for j in await ab.plan()}


async def test_settings_persist_and_disable(db: Database, pm: ProviderManager):
    await _provider(db, "a", ["m"])
    fake = FakeBench()
    pm.benchmark_provider = fake
    ab = AutoBenchmarker(db, pm)
    cfg = await ab.set_config(enabled=False, idle_seconds=600, max_age_days=3)
    assert (cfg.enabled, cfg.idle_seconds, cfg.max_age_days) == (False, 600, 3.0)
    await ab.run_once()
    await _drain(ab)
    assert fake.calls == []
    # A fresh instance reads the saved settings.
    again = await AutoBenchmarker(db, pm).config()
    assert again == cfg
    with pytest.raises(ValueError):
        await ab.set_config(idle_seconds=5)
    with pytest.raises(ValueError):
        await ab.set_config(max_age_days=0)


async def test_stop_cancels_running_benchmark(db: Database, pm: ProviderManager):
    await _provider(db, "a", ["m"])
    gate = asyncio.Event()
    fake = FakeBench(gate=gate)
    pm.benchmark_provider = fake
    ab = AutoBenchmarker(db, pm, check_seconds=3600)
    await ab.start()
    await asyncio.wait_for(fake.started.wait(), 1)
    await ab.stop()
    assert ab._runs == {} and ab._current == {}


class _FakeOllama:
    def __init__(self, startup_ms: float) -> None:
        self.startup_ms = startup_ms

    async def benchmark_chat(self, model, prompt):
        return {"startup_time_ms": self.startup_ms, "tokens_per_second": 33.0}

    async def close(self):
        pass


async def test_warm_runs_do_not_record_startup(db: Database, pm: ProviderManager):
    p = await _provider(db, "a", ["m"])
    pm._ollama_clients[p.id] = _FakeOllama(startup_ms=2.0)
    pm._set_hot_models(p.id, [{"name": "m"}])
    warm = await pm.benchmark_provider(p.id, "m", source="auto")
    assert warm.startup_time_ms is None and warm.source == "auto"

    pm._set_hot_models(p.id, [])
    pm._ollama_clients[p.id] = _FakeOllama(startup_ms=4200.0)
    cold = await pm.benchmark_provider(p.id, "m")
    assert cold.startup_time_ms == 4200.0 and cold.source == "manual"

    rows = await db.get_all_benchmark_results()
    assert {r.source for r in rows} == {"auto", "manual"}
    logs = [e.endpoint for e in await db.get_request_logs(10, 0)]
    assert "auto-benchmark" in logs and "benchmark" in logs


def test_latest_benchmarks_keep_last_measured_startup():
    rows = [  # newest first
        BenchmarkResult(
            provider_id=1,
            model_name="m",
            protocol="ollama",
            startup_time_ms=None,
            tokens_per_second=50.0,
        ),
        BenchmarkResult(
            provider_id=1,
            model_name="m",
            protocol="ollama",
            startup_time_ms=3000.0,
            tokens_per_second=40.0,
        ),
    ]
    merged = latest_benchmarks(rows)
    assert merged[(1, "m", "ollama")].startup_time_ms == 3000.0
    assert merged[(1, "m", "ollama")].tokens_per_second == 50.0
    assert merged[(1, "m", None)].startup_time_ms == 3000.0


def test_llamacpp_metrics_use_server_timings():
    data = {
        "timings": {
            "predicted_per_second": 61.5,
            "prompt_ms": 40.0,
            "predicted_ms": 900.0,
        },
        "usage": {"completion_tokens": 55},
    }
    m = _chat_metrics(data, elapsed_s=3.44)
    assert m["tokens_per_second"] == 61.5
    assert m["startup_time_ms"] == pytest.approx(2500.0)
    # No timings: no startup claim, speed from tokens over wall time.
    m = _chat_metrics({"usage": {"completion_tokens": 50}}, elapsed_s=2.0)
    assert m == {"startup_time_ms": None, "tokens_per_second": 25.0}


async def test_auto_benchmark_api(db: Database, pm: ProviderManager):
    from fastapi.testclient import TestClient

    from llama_router.dashboard import deps as dash_deps
    from llama_router.dashboard.app import app as dash_app

    await _provider(db, "a", ["m"])
    dash_deps.init(db, pm)
    dash_deps.init_auto_benchmark(AutoBenchmarker(db, pm))
    with TestClient(dash_app) as client:
        status = client.get("/api/auto-benchmark").json()
        assert status["config"]["enabled"] is True
        assert status["pending"] == 1 and status["next"][0]["model"] == "m"
        saved = client.post(
            "/api/auto-benchmark",
            json={"enabled": False, "idle_minutes": 10, "max_age_days": 2},
        ).json()
        assert saved == {"enabled": False, "idle_seconds": 600, "max_age_days": 2.0}
        bad = client.post("/api/auto-benchmark", json={"idle_minutes": 0.1})
        assert bad.status_code == 400
        page = client.get("/").text
        assert "Auto-benchmark" in page and 'id="ab-status"' in page
    dash_deps.init_auto_benchmark(None)
