"""Keeping the routing table current without rebuilding it on every sample."""

from __future__ import annotations

import asyncio
import inspect
from typing import Any

import pytest

from llama_router import router as router_mod
from llama_router import routing_table
from llama_router.database import CHANGE_STATUS, Database
from llama_router.models import (
    Provider,
    ProviderModel,
    ProviderProtocol,
    ProviderStatus,
)
from llama_router.provider_manager import ProviderManager
from llama_router.router import Router
from llama_router.routing_stats import WARM_WINDOW_S, ProviderStats

ENDPOINT = "/v1/chat/completions"


class FakeClock:
    def __init__(self) -> None:
        self.t = 1000.0

    def __call__(self) -> float:
        return self.t


async def _provider(db: Database, name: str, models: list[str]) -> Provider:
    p = await db.add_provider(name, ProviderProtocol.OLLAMA)
    pid = p.require_id()
    await db.update_provider_status(pid, ProviderStatus.IDLE)
    await db.set_provider_models(
        pid, [ProviderModel(provider_id=pid, name=m) for m in models]
    )
    return p


def _count(monkeypatch: pytest.MonkeyPatch, module: object, name: str) -> list[int]:
    """Count calls to *module.name*, still calling it."""
    calls = [0]
    real = getattr(module, name)

    def wrapper(*args: Any, **kwargs: Any) -> Any:
        calls[0] += 1
        return real(*args, **kwargs)

    async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
        calls[0] += 1
        return await real(*args, **kwargs)

    monkeypatch.setattr(
        module, name, async_wrapper if inspect.iscoroutinefunction(real) else wrapper
    )
    return calls


@pytest.mark.asyncio
async def test_stats_change_rescores_without_reading_database(
    db: Database, monkeypatch: pytest.MonkeyPatch
) -> None:
    p1 = await _provider(db, "a", ["m"])
    p2 = await _provider(db, "b", ["m"])
    pm = ProviderManager(db)
    rt = Router(db, pm)
    first = await rt.rebuild()
    loads = _count(monkeypatch, router_mod, "load_inputs")
    reloads = _count(monkeypatch, routing_table, "reload")

    # b turns out much faster: only the scores change.
    pm.stats.record_request(p2.require_id(), "m", ok=True, tokens_per_second=500.0)
    pm.stats.record_request(p1.require_id(), "m", ok=True, tokens_per_second=5.0)
    assert rt._need_rescore
    assert not rt._need_reload
    table = await rt.current_table()

    assert loads[0] == 0
    assert reloads[0] == 0
    assert table.version == first.version + 1
    assert (
        table.ranked_for("m", ENDPOINT, "throughput")[0].provider.id == p2.require_id()
    )
    # The served table was copied, not changed under its readers.
    assert first.ranked_for("m", ENDPOINT, "throughput")[0].tokens_per_second is None


@pytest.mark.asyncio
async def test_burst_of_similar_outcomes_rebuilds_at_most_once(
    db: Database, monkeypatch: pytest.MonkeyPatch
) -> None:
    p = await _provider(db, "a", ["m"])
    pm = ProviderManager(db)
    pm.stats.record_request(p.require_id(), "m", ok=True, tokens_per_second=50.0)
    # No debounce or spacing: only the stats decide what is worth a rebuild.
    rt = Router(db, pm, debounce_s=0.0, min_interval_s=0.0)
    await rt.start()
    try:
        rescores = _count(monkeypatch, routing_table, "rescore")
        for i in range(10):
            pm.stats.record_request(
                p.require_id(), "m", ok=True, tokens_per_second=50.0 + i % 2
            )
            await asyncio.sleep(0.01)
        await asyncio.sleep(0.1)
        assert rescores[0] <= 1
    finally:
        await rt.stop()


@pytest.mark.asyncio
async def test_busy_idle_flip_does_not_reload(db: Database) -> None:
    p = await _provider(db, "a", ["m"])
    pm = ProviderManager(db)
    rt = Router(db, pm)
    await rt.rebuild()
    seen: list[str] = []
    db.add_change_listener(seen.append)

    await db.update_provider_status(p.require_id(), ProviderStatus.BUSY)
    await db.update_provider_status(p.require_id(), ProviderStatus.IDLE)
    assert CHANGE_STATUS not in seen
    assert not rt._need_reload
    # The status is still stored for display.
    await db.update_provider_status(p.require_id(), ProviderStatus.BUSY)
    stored = await db.get_provider(p.require_id())
    assert stored is not None
    assert stored.status == ProviderStatus.BUSY

    # Going offline still matters to routing.
    await db.update_provider_status(p.require_id(), ProviderStatus.OFFLINE)
    assert CHANGE_STATUS in seen
    assert rt._need_reload


def test_stats_notify_only_on_moves_that_can_reorder() -> None:
    clock = FakeClock()
    stats = ProviderStats(clock=clock)
    fired: list[str] = []
    stats.add_listener(fired.append)

    # Nothing known yet at the last build: the first sample matters.
    stats.record_request(1, "m", ok=True, tokens_per_second=100.0, ttft_ms=200.0)
    assert fired == ["request"]
    stats.mark_built()
    fired.clear()

    # Small moves in speed and first-token time do not.
    stats.record_request(1, "m", ok=True, tokens_per_second=105.0, ttft_ms=210.0)
    assert fired == []
    # A large one does.
    stats.record_request(1, "m", ok=True, tokens_per_second=200.0)
    assert fired == ["request"]
    stats.mark_built()
    fired.clear()

    # One failure moves the failure rate by more than RATE_MOVE.
    stats.record_request(1, "m", ok=False)
    assert fired == ["request"]
    stats.mark_built()
    fired.clear()

    # A model going cold and warming up again flips its warm flag.
    clock.t += 3600.0
    stats.mark_built()
    stats.record_request(1, "m", ok=True, tokens_per_second=200.0)
    assert fired == ["request"]


def test_health_checks_notify_only_on_moves_that_can_reorder() -> None:
    stats = ProviderStats(clock=FakeClock())
    fired: list[str] = []
    stats.add_listener(fired.append)

    stats.record_probe(1, 10, ok=True, ms=20.0)
    stats.record_check(1, ok=True)
    assert fired == ["health check"]
    stats.mark_built()
    fired.clear()

    # Jitter, a passing check and a failed address are not worth a rebuild.
    stats.record_probe(1, 10, ok=True, ms=21.0)
    stats.record_check(1, ok=True)
    stats.record_probe(1, 11, ok=False)
    assert fired == []
    # A failed check is.
    stats.record_check(1, ok=False)
    assert fired == ["health check"]


@pytest.mark.asyncio
async def test_locality_and_hub_rankings_are_built_on_first_lookup(
    db: Database,
) -> None:
    local = await _provider(db, "local", ["m"])
    cloud = await _provider(db, "cloud", ["m"])
    await db.update_provider_settings(cloud.require_id(), locality="cloud")
    await db.set_hub_shared_providers([local.require_id()])
    pm = ProviderManager(db)
    rt = Router(db, pm)
    table = await rt.rebuild()
    assert table.ranked_by_locality == {}
    assert table.ranked_for_hub == {}

    ranked = table.ranked_for("m", ENDPOINT, "balanced", locality="cloud")
    assert [c.provider.name for c in ranked] == ["cloud"]
    assert list(table.ranked_by_locality) == [("m", ENDPOINT, "balanced", "cloud")]
    assert table.ranked_for("m", ENDPOINT, "balanced", locality="cloud") is ranked

    hub = table.ranked_for("m", ENDPOINT, "balanced", hub_only=True)
    assert [c.provider.name for c in hub] == ["local"]
    assert table.ranked_for("m", ENDPOINT, "balanced", "cloud", hub_only=True) == ()

    # Names nobody serves are not remembered.
    assert table.ranked_for("nope", ENDPOINT, "balanced", locality="cloud") == ()
    assert table.ranked_for("m", ENDPOINT, "nope", locality="cloud") == ()
    assert ("nope", ENDPOINT, "balanced", "cloud") not in table.ranked_by_locality


def test_health_checks_notice_what_time_alone_changed() -> None:
    clock = FakeClock()
    stats = ProviderStats(clock=clock)
    fired: list[str] = []
    stats.add_listener(fired.append)

    for _ in range(5):
        stats.record_request(1, "m", ok=False)
    assert stats.provider_health(1).circuit_open
    stats.mark_built()
    fired.clear()

    stats.record_check(1, ok=True)
    assert fired == []
    # Failures decay with no traffic: the circuit closes by the next check.
    clock.t += 3600.0
    stats.record_check(1, ok=True)
    assert fired == ["health check"]


def test_health_checks_notice_a_model_going_cold() -> None:
    clock = FakeClock()
    stats = ProviderStats(clock=clock)
    fired: list[str] = []
    stats.add_listener(fired.append)
    stats.record_request(1, "m", ok=True, tokens_per_second=50.0)
    stats.mark_built()
    fired.clear()

    clock.t += WARM_WINDOW_S + 1
    stats.record_check(1, ok=True)
    assert fired == ["health check"]
