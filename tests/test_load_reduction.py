"""Dashboard and sidecar load: bulk reads, listings from routing inputs, blob
writes off the event loop, per-scope affinity eviction, one deadline timer."""

from __future__ import annotations

import asyncio
import hashlib
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import pytest
from starlette.requests import Request

from llama_router.affinity import SessionAffinity
from llama_router.config import settings
from llama_router.database import Database
from llama_router.limits import DeadlineExceeded, upstream_slot
from llama_router.model_listing import listed_models, served_models
from llama_router.models import (
    BenchmarkResult,
    ProviderModel,
    ProviderProtocol,
    ProviderStatus,
)
from llama_router.provider_manager import ProviderManager
from llama_router.registry_cache import app as cache_app
from llama_router.router import Router
from llama_router.routing_table import load_inputs


@pytest.fixture
async def pm(db: Database) -> AsyncIterator[ProviderManager]:
    manager = ProviderManager(db)
    yield manager
    await manager.stop()


async def _provider(
    db: Database,
    name: str,
    models: tuple[str, ...],
    status: ProviderStatus = ProviderStatus.IDLE,
) -> int:
    pid = (await db.add_provider(name, ProviderProtocol.OPENAI)).require_id()
    await db.update_provider_status(pid, status)
    await db.set_provider_models(
        pid,
        [ProviderModel(provider_id=pid, name=m, size=len(name)) for m in models],
    )
    await db.add_address(pid, f"http://{name}.test:8080")
    return pid


def _refuse(*_a: Any, **_k: Any) -> Any:
    raise AssertionError("per-provider query")


# --- list_provider_infos -----------------------------------------------------


async def test_provider_infos_use_bulk_queries(
    db: Database, pm: ProviderManager, monkeypatch: pytest.MonkeyPatch
) -> None:
    a = await _provider(db, "a", ("m1", "m2"))
    b = await _provider(db, "b", ("m3",))
    await db.save_benchmark(
        BenchmarkResult(provider_id=b, model_name="m3", tokens_per_second=9.0)
    )
    for name in ("get_provider_models", "get_benchmarks_for_provider", "get_addresses"):
        monkeypatch.setattr(db, name, _refuse)

    infos = {i.provider.id: i for i in await pm.list_provider_infos()}
    assert [m.name for m in infos[a].models] == ["m1", "m2"]
    assert [m.name for m in infos[b].models] == ["m3"]
    assert [x.url for x in infos[a].addresses] == ["http://a.test:8080"]
    assert infos[a].benchmarks == []
    assert [x.tokens_per_second for x in infos[b].benchmarks] == [9.0]


# --- model listings -----------------------------------------------------------


async def test_served_models_match_the_database_listing(db: Database) -> None:
    a = await _provider(db, "a", ("shared", "only-a"))
    b = await _provider(db, "b", ("shared", "only-b"))
    await _provider(db, "down", ("only-down",), ProviderStatus.OFFLINE)
    await db.set_hub_shared_providers([a])
    inputs = await load_inputs(db)
    cases: list[dict[str, Any]] = [
        {},
        {"hub_only": True},
        {"only_id": b},
        {"hub_only": True, "exclude_ids": {a}},
    ]
    for kwargs in cases:
        assert served_models(inputs, **kwargs) == await db.list_all_models(
            hub_only=kwargs.get("hub_only", False),
            exclude_ids=kwargs.get("exclude_ids"),
            only_id=kwargs.get("only_id"),
        ), kwargs
    # The lowest model row wins a shared name: provider a's.
    assert next(m for m in served_models(inputs) if m["name"] == "shared")["size"] == 1


async def test_listing_reads_the_routing_inputs(
    db: Database, pm: ProviderManager, monkeypatch: pytest.MonkeyPatch
) -> None:
    await _provider(db, "a", ("m1",))
    rt = Router(db, pm)
    try:
        await rt.rebuild()
        monkeypatch.setattr(db, "list_all_models", _refuse)
        assert [m["name"] for m in await listed_models(db, rt)] == ["m1"]
        # A change reloads the inputs before the next listing (no task here).
        await _provider(db, "b", ("m2",))
        assert [m["name"] for m in await listed_models(db, rt)] == ["m1", "m2"]
    finally:
        await rt.stop()


async def test_listing_without_a_router_reads_the_database(db: Database) -> None:
    await _provider(db, "a", ("m1",))
    assert [m["name"] for m in await listed_models(db, None)] == ["m1"]


# --- registry cache blob writes ----------------------------------------------


async def test_blob_writer_buffers_and_hashes_in_a_thread(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(cache_app, "FLUSH_SIZE", 10)
    hops = 0
    real = asyncio.to_thread

    async def counting(fn: Any, *args: Any) -> Any:
        nonlocal hops
        hops += 1
        return await real(fn, *args)

    monkeypatch.setattr(asyncio, "to_thread", counting)
    chunks = [b"abcd"] * 7  # 28 bytes: flushes at 12 and 24, then 4 at the end
    path = tmp_path / "blob.tmp"
    async with cache_app._BlobWriter(path) as writer:
        for c in chunks:
            await writer.write(c)
        digest = await writer.digest()
    data = b"".join(chunks)
    assert path.read_bytes() == data
    assert digest == f"sha256:{hashlib.sha256(data).hexdigest()}"
    assert hops == 5  # open, three writes, close


async def test_blob_writer_refuses_writes_after_close(tmp_path: Path) -> None:
    writer = cache_app._BlobWriter(tmp_path / "blob.tmp")
    async with writer:
        await writer.write(b"x")
    with pytest.raises(RuntimeError):
        await writer.digest()


# --- session affinity ---------------------------------------------------------


def test_scope_eviction_keeps_scope_bookkeeping_in_step(clock: Any) -> None:
    store = SessionAffinity(ttl_s=60, max_sessions=3, clock=clock, max_per_key=2)
    store.bind("1:m:a", 1)
    store.bind("2:m:x", 1)
    store.bind("1:m:b", 1)
    store.bind("2:m:y", 1)  # global cap drops 1:m:a, the oldest overall
    assert store.get("1:m:a") is None
    store.bind("1:m:c", 1)  # global cap drops 2:m:x; key 1 is at its cap
    assert store.get("2:m:x") is None
    assert store.get("1:m:b") is not None
    assert {k: list(v) for k, v in store._per_scope.items()} == {
        "1": ["1:m:b", "1:m:c"],
        "2": ["2:m:y"],
    }
    store.clear()
    assert store._per_scope == {}


def test_default_session_cap_is_raised() -> None:
    assert type(settings).model_fields["session_affinity_max_sessions"].default == 10000


# --- stream deadline ----------------------------------------------------------


def _request() -> Request:
    return Request(
        {"type": "http", "headers": [], "method": "POST", "path": "/", "state": {}}
    )


async def _fast(n: int) -> AsyncIterator[bytes]:
    for i in range(n):
        yield f"chunk{i}\n".encode()


async def test_stream_arms_one_timer(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "request_deadline_seconds", 10)

    def no_timeout(*_a: Any) -> Any:
        raise AssertionError("asyncio.timeout per chunk")

    monkeypatch.setattr(asyncio, "timeout", no_timeout)
    slot = await upstream_slot(_request(), None, stream=True)
    async with slot:
        got = [c async for c in slot.stream(_fast(50))]
    assert len(got) == 50


async def test_deadline_between_chunks_ends_the_stream(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The deadline passing while the consumer is busy elsewhere does not
    cancel the consumer; the next chunk it asks for raises instead."""
    monkeypatch.setattr(settings, "request_deadline_seconds", 0.05)
    slot = await upstream_slot(_request(), None, stream=True)
    got: list[bytes] = []

    async def consume() -> None:
        async for chunk in slot.stream(_fast(5)):
            got.append(chunk)
            await asyncio.sleep(0.1)  # must finish: not cancelled

    with pytest.raises(DeadlineExceeded):
        await consume()
    assert len(got) == 1
    slot.release()


async def test_stream_done_before_deadline_leaves_no_timer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "request_deadline_seconds", 0.05)
    slot = await upstream_slot(_request(), None, stream=True)
    assert len([c async for c in slot.stream(_fast(3))]) == 3
    await asyncio.sleep(0.1)  # a stray timer would cancel this
    slot.release()


async def test_outside_cancel_is_not_a_deadline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "request_deadline_seconds", 10)
    slot = await upstream_slot(_request(), None, stream=True)

    async def slow() -> AsyncIterator[bytes]:
        await asyncio.sleep(10)
        yield b"never"

    async def consume() -> None:
        async for _ in slot.stream(slow()):
            pass

    task = asyncio.create_task(consume())
    await asyncio.sleep(0.01)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    slot.release()
