"""Latency scoring for loaded models nobody has measured yet."""

from __future__ import annotations

import pytest

from llama_router.database import Database
from llama_router.models import (
    BenchmarkResult,
    ProviderModel,
    ProviderProtocol,
    ProviderStatus,
)
from llama_router.provider_manager import ProviderManager
from llama_router.router import Router
from llama_router.routing_stats import ProviderStats
from llama_router.routing_table import UNKNOWN_HOT_TTFT_MS, Candidate
from tests.conftest import FakeClock

CHAT = "/v1/chat/completions"


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock()


@pytest.fixture
def pm(db: Database, clock: FakeClock) -> ProviderManager:
    manager = ProviderManager(db)
    manager.stats = ProviderStats(clock=clock)
    return manager


async def _hot(
    db: Database, pm: ProviderManager, name: str, *, tps: float = 50.0
) -> int:
    """A provider with model "m" loaded."""
    p = await db.add_provider(name, ProviderProtocol.OLLAMA)
    pid = p.require_id()
    await db.update_provider_status(pid, ProviderStatus.IDLE)
    await db.set_provider_models(pid, [ProviderModel(provider_id=pid, name="m")])
    await db.save_benchmark(
        BenchmarkResult(
            provider_id=pid, model_name="m", startup_time_ms=3000, tokens_per_second=tps
        )
    )
    pm._set_hot_models(pid, [{"name": "m"}])
    return pid


async def _ranked(rt: Router, mode: str) -> tuple[Candidate, ...]:
    await rt.rebuild()
    table = await rt.current_table()
    return table.ranked_for("m", CHAT, mode)


async def test_unmeasured_hot_candidate_no_longer_beats_a_measured_faster_one(
    db: Database, pm: ProviderManager
) -> None:
    fast = await _hot(db, pm, "fast")
    slow = await _hot(db, pm, "slow")
    untried = await _hot(db, pm, "untried")
    pm.stats.record_request(fast, "m", ok=True, ttft_ms=100.0)
    pm.stats.record_request(slow, "m", ok=True, ttft_ms=300.0)
    rt = Router(db, pm)

    ranked = await _ranked(rt, "latency")
    assert [c.provider_id for c in ranked] == [fast, untried, slow]
    estimate = next(c for c in ranked if c.provider_id == untried)
    # The median of the providers measured: (100 + 300) / 2.
    assert estimate.ttft_ms is None
    assert estimate.ttft_estimate_ms == pytest.approx(200.0)
    assert estimate.to_dict()["ttft_estimate_ms"] == pytest.approx(200.0)
    result = await rt.route("m", CHAT)
    assert result is not None


async def test_nothing_measured_keeps_the_order(
    db: Database, pm: ProviderManager
) -> None:
    quick = await _hot(db, pm, "quick", tps=80.0)
    steady = await _hot(db, pm, "steady", tps=20.0)
    rt = Router(db, pm)
    for mode in ("balanced", "throughput"):
        ranked = await _ranked(rt, mode)
        assert [c.provider_id for c in ranked] == [quick, steady]
    ranked = await _ranked(rt, "latency")
    # Both unmeasured: the same assumed first token, so a tie, not a winner.
    scores = {c.provider_id: c.scores["latency"] for c in ranked}
    assert scores[quick] == scores[steady] == pytest.approx(UNKNOWN_HOT_TTFT_MS)
    assert all(c.ttft_estimate_ms is None for c in ranked)


async def test_a_real_sample_replaces_the_estimate_on_rescore(
    db: Database, pm: ProviderManager
) -> None:
    measured = await _hot(db, pm, "measured")
    untried = await _hot(db, pm, "untried")
    pm.stats.record_request(measured, "m", ok=True, ttft_ms=400.0)
    rt = Router(db, pm)
    ranked = await _ranked(rt, "latency")
    by_id = {c.provider_id: c for c in ranked}
    assert by_id[untried].ttft_estimate_ms == pytest.approx(400.0)

    pm.stats.record_request(untried, "m", ok=True, ttft_ms=50.0)
    ranked = await _ranked(rt, "latency")
    assert ranked[0].provider_id == untried
    assert ranked[0].ttft_ms == pytest.approx(50.0)
    assert ranked[0].ttft_estimate_ms is None
    assert ranked[0].scores["latency"] == pytest.approx(50.0)


async def test_cloud_models_count_as_loaded_but_not_instant(
    db: Database, pm: ProviderManager
) -> None:
    local = await _hot(db, pm, "local")
    cloud = await db.add_provider("cloud", ProviderProtocol.OPENAI)
    cid = cloud.require_id()
    await db.update_provider_settings(cid, locality="cloud")
    await db.update_provider_status(cid, ProviderStatus.IDLE)
    await db.set_provider_models(cid, [ProviderModel(provider_id=cid, name="m")])
    pm.stats.record_request(local, "m", ok=True, ttft_ms=150.0)
    rt = Router(db, pm)
    ranked = await _ranked(rt, "latency")
    by_id = {c.provider_id: c for c in ranked}
    assert by_id[cid].is_hot
    # Scored as the measured provider, not as an instant first token.
    assert by_id[cid].ttft_estimate_ms == pytest.approx(150.0)
    assert by_id[cid].scores["latency"] >= by_id[local].scores["latency"]
