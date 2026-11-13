"""Queue order follows the routing mode: latency, then balanced, then bulk."""

from __future__ import annotations

import asyncio

import pytest

from llama_router import metrics
from llama_router.capacity import (
    BULK_LEVEL,
    DEFAULT_LEVEL,
    LATENCY_LEVEL,
    SlotGate,
    queue_level,
)
from llama_router.config import settings
from llama_router.database import Database
from llama_router.limits import limiter
from llama_router.models import (
    BenchmarkResult,
    ProviderModel,
    ProviderProtocol,
    ProviderStatus,
)
from llama_router.provider_manager import ProviderManager
from llama_router.router import Router, RoutingPreferences

CHAT = "/v1/chat/completions"


@pytest.fixture(autouse=True)
def _long_wait(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "upstream_queue_timeout_seconds", 5.0)


def test_mode_sets_the_level() -> None:
    assert queue_level("latency") == LATENCY_LEVEL
    assert queue_level("balanced") == DEFAULT_LEVEL
    assert queue_level("chaos") == DEFAULT_LEVEL
    assert queue_level(None) == DEFAULT_LEVEL
    assert queue_level("throughput") == BULK_LEVEL
    assert queue_level("cheapest") == BULK_LEVEL


async def _waiter(gate: SlotGate, level: int, tag: str, order: list[str]) -> None:
    await gate.acquire(None, level)
    order.append(tag)


async def test_levels_are_served_in_order_and_fifo_within_one() -> None:
    gate = SlotGate(1)
    await gate.acquire(None)
    order: list[str] = []
    plan = [
        (BULK_LEVEL, "bulk-1"),
        (DEFAULT_LEVEL, "balanced-1"),
        (LATENCY_LEVEL, "latency-1"),
        (BULK_LEVEL, "bulk-2"),
        (DEFAULT_LEVEL, "balanced-2"),
        (LATENCY_LEVEL, "latency-2"),
    ]
    tasks = []
    for level, tag in plan:
        tasks.append(asyncio.create_task(_waiter(gate, level, tag, order)))
        await asyncio.sleep(0)
    assert gate.queued == 6
    for _ in plan:
        gate.release()
        await asyncio.sleep(0)
    await asyncio.gather(*tasks)
    assert order == [
        "latency-1",
        "latency-2",
        "balanced-1",
        "balanced-2",
        "bulk-1",
        "bulk-2",
    ]


async def test_bulk_never_takes_the_last_slot_of_a_wider_gate() -> None:
    gate = SlotGate(2)
    await gate.acquire(None, LATENCY_LEVEL)
    assert not gate.has_room(BULK_LEVEL)
    assert gate.has_room(LATENCY_LEVEL)
    assert gate.has_room(DEFAULT_LEVEL)
    bulk = asyncio.create_task(gate.acquire(None, BULK_LEVEL))
    await asyncio.sleep(0)
    assert not bulk.done()
    # The last slot stays for a latency or balanced request...
    await gate.acquire(None, DEFAULT_LEVEL)
    assert gate.in_use == 2
    # ...and the bulk waiter goes once two are free.
    gate.release()
    await asyncio.sleep(0)
    assert not bulk.done()
    gate.release()
    await asyncio.wait_for(bulk, 1)
    assert gate.in_use == 1


async def test_bulk_may_take_the_only_slot_when_no_one_waits() -> None:
    gate = SlotGate(1)
    coro = gate.acquire(None, BULK_LEVEL)
    with pytest.raises(StopIteration):
        coro.send(None)  # took it without suspending
    assert gate.in_use == 1


async def test_bulk_waits_behind_others_on_a_one_slot_gate() -> None:
    gate = SlotGate(1)
    await gate.acquire(None)
    order: list[str] = []
    bulk = asyncio.create_task(_waiter(gate, BULK_LEVEL, "bulk", order))
    await asyncio.sleep(0)
    fast = asyncio.create_task(_waiter(gate, LATENCY_LEVEL, "latency", order))
    await asyncio.sleep(0)
    gate.release()
    await asyncio.sleep(0)
    gate.release()
    await asyncio.gather(bulk, fast)
    assert order == ["latency", "bulk"]


async def test_a_lower_level_arrival_takes_a_slot_bulk_may_not() -> None:
    gate = SlotGate(2)
    await gate.acquire(None)
    bulk = asyncio.create_task(gate.acquire(None, BULK_LEVEL))
    await asyncio.sleep(0)
    # A bulk waiter is queued, but a latency request still finds the slot.
    coro = gate.acquire(None, LATENCY_LEVEL)
    with pytest.raises(StopIteration):
        coro.send(None)
    bulk.cancel()
    await asyncio.gather(bulk, return_exceptions=True)
    assert gate.queued == 0


async def test_nothing_is_preempted() -> None:
    gate = SlotGate(1)
    await gate.acquire(None, BULK_LEVEL)
    fast = asyncio.create_task(gate.acquire(None, LATENCY_LEVEL))
    await asyncio.sleep(0.01)
    assert not fast.done()
    assert gate.in_use == 1
    gate.release()
    await asyncio.wait_for(fast, 1)


async def test_upstream_limiter_serves_levels_in_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "upstream_max_concurrency", 1)
    monkeypatch.setattr(settings, "upstream_max_concurrency_per_key", 0)
    limiter.reset()
    held = await limiter.acquire(None, stream=False, deadline=None)
    order: list[int] = []

    async def wait(level: int) -> None:
        got = await limiter.acquire(None, stream=False, deadline=None, level=level)
        order.append(level)
        limiter.release(None, stream=False, held=got)

    tasks = []
    for level in (BULK_LEVEL, DEFAULT_LEVEL, LATENCY_LEVEL):
        tasks.append(asyncio.create_task(wait(level)))
        await asyncio.sleep(0)
    text = metrics.render(metrics.collect([], _Stats()))
    assert 'llama_router_upstream_queued_requests{level="3"} 1' in text
    limiter.release(None, stream=False, held=held)
    await asyncio.gather(*tasks)
    assert order == [LATENCY_LEVEL, DEFAULT_LEVEL, BULK_LEVEL]
    limiter.reset()


class _Stats:
    def model_entries(self) -> list[object]:
        return []


async def _provider(db: Database, name: str, *, startup: float, limit: int) -> int:
    p = await db.add_provider(name, ProviderProtocol.OPENAI)
    pid = p.require_id()
    await db.update_provider_status(pid, ProviderStatus.IDLE)
    await db.set_provider_models(pid, [ProviderModel(provider_id=pid, name="m")])
    await db.update_provider_settings(pid, max_concurrent=limit)
    await db.save_benchmark(
        BenchmarkResult(
            provider_id=pid,
            model_name="m",
            protocol="openai",
            startup_time_ms=startup,
            tokens_per_second=50.0,
        )
    )
    return pid


async def test_routing_sets_bulk_aside_from_a_last_slot(db: Database) -> None:
    pm = ProviderManager(db)
    rt = Router(db, pm)
    best = await _provider(db, "best", startup=100.0, limit=2)
    other = await _provider(db, "other", startup=900.0, limit=2)
    await rt.route("m", CHAT)
    gate = await pm.take_slot(best)
    # One slot left on best: a latency key takes it, a throughput key
    # goes where two are free; the mode applies from the next request.
    fast = await rt.route("m", CHAT, RoutingPreferences(mode="latency"))
    assert fast is not None
    assert fast.provider.id == best
    bulk = await rt.route("m", CHAT, RoutingPreferences(mode="throughput"))
    assert bulk is not None
    assert bulk.provider.id == other
    explained = await rt.explain("m", CHAT, RoutingPreferences(mode="throughput"))
    assert explained["request"]["queue_level"] == BULK_LEVEL
    assert gate is not None
    gate.release()


async def test_upstream_slot_takes_the_level_of_the_keys_mode() -> None:
    from starlette.requests import Request

    from llama_router.limits import upstream_slot

    request = Request({"type": "http", "headers": [], "state": {}})
    for mode, level in (("latency", 1), ("balanced", 2), ("cheapest", 3)):
        slot = await upstream_slot(request, RoutingPreferences(mode=mode))
        assert slot.level == level
        slot.release()
    slot = await upstream_slot(request, None)
    assert slot.level == DEFAULT_LEVEL
    slot.release()
