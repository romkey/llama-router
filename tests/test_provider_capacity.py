"""Provider capacity limits (``providers.max_concurrent``)."""

from __future__ import annotations

import asyncio
import contextlib
import sqlite3
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import httpx
import pytest
from alembic import command

from llama_router import capacity, metrics, proxy
from llama_router.auth import generate_api_key, key_prefix
from llama_router.capacity import ProviderAtCapacity, QueueTimeout, SlotGate
from llama_router.config import settings
from llama_router.database import Database
from llama_router.llamacpp_api import deps as v1_deps
from llama_router.llamacpp_api.app import app as v1_app
from llama_router.models import (
    BenchmarkResult,
    ProviderModel,
    ProviderProtocol,
    ProviderStatus,
)
from llama_router.provider_manager import ProviderManager
from llama_router.provider_secrets import key_hash
from llama_router.router import Router, RoutingPreferences
from tests.conftest import asgi_client
from tests.test_alembic_roundtrip import _cfg

CHAT = "/v1/chat/completions"
BODY = {"model": "m", "messages": [{"role": "user", "content": "hi"}]}


@pytest.fixture
def pm(db: Database) -> ProviderManager:
    return ProviderManager(db)


@pytest.fixture
async def rt(db: Database, pm: ProviderManager) -> Router:
    return Router(db, pm)


@pytest.fixture
async def v1(
    db: Database, pm: ProviderManager, rt: Router
) -> AsyncIterator[httpx.AsyncClient]:
    v1_deps.init(db, pm, rt)
    async with asgi_client(v1_app) as client:
        yield client
    v1_deps._db, v1_deps._pm, v1_deps._router = None, None, None


async def _provider(
    db: Database, name: str, *, startup: float, limit: int | None = None
) -> int:
    p = await db.add_provider(name, ProviderProtocol.OPENAI)
    pid = p.require_id()
    await db.update_provider_status(pid, ProviderStatus.IDLE)
    await db.set_provider_models(pid, [ProviderModel(provider_id=pid, name="m")])
    if limit is not None:
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


async def _key(db: Database, pins: dict[str, int] | None = None, **kw: Any) -> str:
    plaintext = generate_api_key()
    key_id = await db.create_api_key(
        key_prefix(plaintext), key_hash(plaintext), "balanced", True, **kw
    )
    for model, pid in (pins or {}).items():
        await db.set_api_key_model_pin(key_id, model, pid)
    return plaintext


class Upstream:
    """A fake provider that answers once *gate* is set."""

    def __init__(self) -> None:
        self.called: list[int] = []
        self.release = asyncio.Event()
        self.release.set()
        self.started = asyncio.Event()

    def client(self, pm: ProviderManager, provider_id: int) -> Any:
        up = self

        class Client:
            async def chat_completions(self, body: dict[str, Any]) -> dict[str, Any]:
                up.called.append(provider_id)
                up.started.set()
                await up.release.wait()
                return {"choices": [{"message": {"content": f"from {provider_id}"}}]}

            async def chat_completions_stream(
                self, body: dict[str, Any]
            ) -> AsyncIterator[bytes]:
                up.called.append(provider_id)
                yield b"data: one\n\n"
                up.started.set()
                await up.release.wait()
                yield b"data: two\n\n"

        return Client()


@pytest.fixture
def up(monkeypatch: pytest.MonkeyPatch) -> Upstream:
    u = Upstream()
    monkeypatch.setattr(proxy, "get_v1_client", u.client)
    monkeypatch.setattr(proxy, "log_request_background", lambda *a, **k: None)
    return u


# --- the gate ---


async def test_free_slot_never_suspends() -> None:
    gate = SlotGate(2)
    coro = gate.acquire(None)
    with pytest.raises(StopIteration):
        coro.send(None)
    assert gate.in_use == 1


async def test_waiters_are_served_first_come_first_served(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "upstream_queue_timeout_seconds", 5.0)
    gate = SlotGate(1)
    await gate.acquire(None)
    order: list[int] = []

    async def wait(n: int) -> None:
        await gate.acquire(None)
        order.append(n)

    tasks = [asyncio.create_task(wait(n)) for n in range(3)]
    await asyncio.sleep(0)
    assert gate.queued == 3
    # A newcomer cannot take a slot the queue is waiting for.
    assert not gate.try_acquire()
    for _ in range(3):
        gate.release()
        await asyncio.sleep(0)
    await asyncio.gather(*tasks)
    assert order == [0, 1, 2]
    assert gate.waits.count == 4


async def test_wait_times_out_with_429(
    db: Database, pm: ProviderManager, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(settings, "upstream_queue_timeout_seconds", 0.05)
    pm.capacity.configure({7: 1})
    gate = pm.capacity.gate(7)
    assert gate is not None
    await gate.acquire(None)
    with pytest.raises(ProviderAtCapacity) as err:
        await pm.take_slot(7)
    assert err.value.status_code == 429
    assert err.value.headers == {"Retry-After": "1"}
    assert gate.queued == 0
    assert gate.refused == 1


async def test_cancelled_waiter_passes_its_slot_on(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "upstream_queue_timeout_seconds", 5.0)
    gate = SlotGate(1)
    await gate.acquire(None)
    first = asyncio.create_task(gate.acquire(None))
    second = asyncio.create_task(gate.acquire(None))
    await asyncio.sleep(0)
    gate.release()  # handed to first...
    first.cancel()  # ...which is cancelled before it runs
    with contextlib.suppress(asyncio.CancelledError):
        await first
    await asyncio.wait_for(second, 1)
    assert gate.in_use == 1


async def test_deadline_bounds_the_wait(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "upstream_queue_timeout_seconds", 30.0)
    gate = SlotGate(1)
    await gate.acquire(None)
    loop = asyncio.get_running_loop()
    with pytest.raises(QueueTimeout):
        await gate.acquire(loop.time() - 1)


async def test_lowered_limit_sheds_load_before_serving_waiters(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "upstream_queue_timeout_seconds", 5.0)
    gate = SlotGate(3)
    for _ in range(3):
        await gate.acquire(None)
    waiter = asyncio.create_task(gate.acquire(None))
    await asyncio.sleep(0)
    gate.resize(1)
    gate.release()
    gate.release()
    await asyncio.sleep(0)
    # Two of three released: one still runs, which is the new limit.
    assert not waiter.done()
    assert gate.in_use == 1
    gate.release()
    await asyncio.wait_for(waiter, 1)
    assert gate.in_use == 1


def test_configure_adds_resizes_and_drops_gates() -> None:
    limits = capacity.CapacityLimits()
    limits.configure({1: 2, 2: None})
    assert limits.limit(1) == 2
    assert limits.gate(2) is None
    assert limits.has_room(2)
    limits.configure({1: 3})
    assert limits.limit(1) == 3
    limits.configure({1: None})
    assert limits.gate(1) is None
    assert not limits.any_limited()


# --- routing ---


async def test_full_provider_set_aside_while_another_has_room(
    db: Database, pm: ProviderManager, rt: Router
) -> None:
    best = await _provider(db, "best", startup=100.0, limit=1)
    other = await _provider(db, "other", startup=900.0)
    first = await rt.route("m", CHAT)
    assert first is not None
    assert first.provider.id == best
    gate = await pm.take_slot(best)
    assert gate is not None
    second = await rt.route("m", CHAT)
    assert second is not None
    assert second.provider.id == other
    explained = await rt.explain("m", CHAT)
    rows = {r["provider_id"]: r for r in explained["steps"][0]["candidates"]}
    assert rows[best]["excluded"] == "at capacity (1/1); another provider has room"
    assert rows[best]["max_concurrent"] == 1
    gate.release()
    again = await rt.route("m", CHAT)
    assert again is not None
    assert again.provider.id == best


async def test_all_full_routes_to_best_and_explains_the_queue(
    db: Database, pm: ProviderManager, rt: Router
) -> None:
    best = await _provider(db, "best", startup=100.0, limit=1)
    other = await _provider(db, "other", startup=900.0, limit=1)
    await rt.route("m", CHAT)
    held = [await pm.take_slot(best), await pm.take_slot(other)]
    chosen = await rt.route("m", CHAT)
    assert chosen is not None
    assert chosen.provider.id == best
    explained = await rt.explain("m", CHAT)
    assert (
        "all at capacity; would queue on best (1/1)"
        in (explained["steps"][0]["outcome"])
    )
    for g in held:
        assert g is not None
        g.release()


async def test_no_limit_is_todays_behaviour(
    db: Database, pm: ProviderManager, rt: Router
) -> None:
    best = await _provider(db, "best", startup=100.0)
    await _provider(db, "other", startup=900.0)
    await rt.route("m", CHAT)
    assert not pm.capacity.any_limited()
    assert await pm.take_slot(best) is None
    for _ in range(5):
        pm.acquire(best)
    chosen = rt.route_with(await rt.current_table(), "m", CHAT)
    assert chosen is not None


async def test_queue_on_full_providers_then_429(
    db: Database,
    v1: httpx.AsyncClient,
    up: Upstream,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "upstream_queue_timeout_seconds", 0.2)
    only = await _provider(db, "only", startup=100.0, limit=1)
    key = await _key(db)
    headers = {"Authorization": f"Bearer {key}"}
    up.release.clear()
    first = asyncio.create_task(v1.post(CHAT, json=BODY, headers=headers))
    await asyncio.wait_for(up.started.wait(), 2)
    refused = await v1.post(CHAT, json=BODY, headers=headers)
    assert refused.status_code == 429
    assert refused.headers["retry-after"] == "1"
    assert "at capacity" in refused.json()["detail"]
    # A queued request is served once the slot frees.
    monkeypatch.setattr(settings, "upstream_queue_timeout_seconds", 5.0)
    queued = asyncio.create_task(v1.post(CHAT, json=BODY, headers=headers))
    for _ in range(50):
        await asyncio.sleep(0.01)
        gate = v1_deps.get_pm().capacity.gate(only)
        if gate is not None and gate.queued:
            break
    up.release.set()
    assert (await first).status_code == 200
    assert (await queued).status_code == 200
    assert up.called == [only, only]


async def test_strict_pin_queues_and_never_moves(
    db: Database,
    pm: ProviderManager,
    rt: Router,
    v1: httpx.AsyncClient,
    up: Upstream,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "upstream_queue_timeout_seconds", 0.1)
    pinned = await _provider(db, "pinned", startup=900.0, limit=1)
    await _provider(db, "roomy", startup=100.0)
    key = await _key(db, pins={"m": pinned}, strict_pins=True)
    await rt.route("m", CHAT)
    gate = await pm.take_slot(pinned)
    resp = await v1.post(CHAT, json=BODY, headers={"Authorization": f"Bearer {key}"})
    assert resp.status_code == 429
    assert up.called == []
    assert gate is not None
    gate.release()
    resp = await v1.post(CHAT, json=BODY, headers={"Authorization": f"Bearer {key}"})
    assert resp.status_code == 200
    assert up.called == [pinned]


async def test_hub_provider_request_waits_on_that_provider(
    db: Database, pm: ProviderManager, rt: Router
) -> None:
    asked = await _provider(db, "asked", startup=900.0, limit=1)
    await _provider(db, "roomy", startup=100.0)
    await rt.route("m", CHAT)
    gate = await pm.take_slot(asked)
    prefs = RoutingPreferences(hub_only=True, hub_provider=asked)
    chosen = await rt.route("m", CHAT, prefs)
    assert chosen is not None
    assert chosen.provider.id == asked
    assert gate is not None
    gate.release()


async def test_stream_holds_its_slot_until_it_ends(
    db: Database,
    pm: ProviderManager,
    v1: httpx.AsyncClient,
    up: Upstream,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    only = await _provider(db, "only", startup=100.0, limit=2)
    key = await _key(db)
    seen: list[int] = []

    def client(pm_: ProviderManager, provider_id: int) -> Any:
        class Client:
            async def chat_completions_stream(
                self, body: dict[str, Any]
            ) -> AsyncIterator[bytes]:
                for n in range(3):
                    gate = pm.capacity.gate(only)
                    seen.append(gate.in_use if gate is not None else -1)
                    yield f"data: {n}\n\n".encode()

        return Client()

    monkeypatch.setattr(proxy, "get_v1_client", client)
    resp = await v1.post(
        CHAT,
        json={**BODY, "stream": True},
        headers={"Authorization": f"Bearer {key}"},
    )
    assert resp.status_code == 200
    assert b"data: 2" in resp.content
    # Held from the first chunk to the last, released once the stream ended.
    assert seen == [1, 1, 1]
    gate = pm.capacity.gate(only)
    assert gate is not None
    assert gate.in_use == 0


async def test_capacity_metrics(db: Database, pm: ProviderManager, rt: Router) -> None:
    pid = await _provider(db, "limited", startup=100.0, limit=3)
    await rt.route("m", CHAT)
    gate = await pm.take_slot(pid)
    text = metrics.render(
        metrics.collect(await pm.list_provider_infos(), pm.stats, capacity=pm.capacity)
    )
    assert 'llama_router_provider_max_concurrent{provider="limited"' in text
    assert "llama_router_provider_queued_requests{" in text
    assert 'llama_router_provider_queue_wait_seconds_bucket{provider="limited"' in text
    assert "llama_router_provider_queue_wait_seconds_count{" in text
    assert gate is not None
    gate.release()
    info = await pm.get_provider_info(pid)
    assert info is not None
    assert info.provider.max_concurrent == 3


def test_migration_round_trip(tmp_path: Path) -> None:
    dbfile = tmp_path / "cap.db"
    cfg = _cfg(dbfile)
    command.upgrade(cfg, "035_provider_max_concurrent")

    def cols() -> set[str]:
        with contextlib.closing(sqlite3.connect(dbfile)) as con:
            return {r[1] for r in con.execute("PRAGMA table_info(providers)")}

    assert "max_concurrent" in cols()
    command.downgrade(cfg, "034_api_key_strict_pins")
    assert "max_concurrent" not in cols()
    command.upgrade(cfg, "head")
    assert "max_concurrent" in cols()
