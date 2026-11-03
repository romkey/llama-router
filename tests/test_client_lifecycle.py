"""Provider clients: replaced, never closed under a running request, and the
checks on provider addresses."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

import pytest

from llama_router.database import Database
from llama_router.models import Provider, ProviderProtocol
from llama_router.ollama_client import OllamaClient
from llama_router.probe import BlockedAddressError
from llama_router.provider_manager import ProviderManager
from llama_router.v1_client import NoReachableAddressError, V1Client, get_v1_client


@pytest.fixture
async def pm(db: Database) -> AsyncIterator[ProviderManager]:
    manager = ProviderManager(db)
    yield manager
    await manager.stop()


async def _provider(db: Database, pm: ProviderManager) -> Provider:
    added = await db.add_provider("be", ProviderProtocol.OLLAMA)
    pid = added.require_id()
    await db.add_address(pid, "http://a.test:11434", is_preferred=True)
    p = await db.get_provider(pid)
    assert p is not None
    await pm._rebuild_clients(p)
    return p


def _closed(client: OllamaClient) -> bool:
    return client._http.is_closed


async def test_rebuild_waits_for_running_requests(
    db: Database, pm: ProviderManager
) -> None:
    p = await _provider(db, pm)
    pid = p.require_id()
    async with pm.acquire_provider(pid):
        old = pm.get_ollama_client(pid)
        async with pm.acquire_provider(pid):
            await pm._rebuild_clients(p)
            new = pm.get_ollama_client(pid)
            assert new is not old
            assert not _closed(old)
        # One request still has the old client.
        assert not _closed(old)
    assert _closed(old)
    assert not _closed(new)


async def test_idle_client_closes_at_once(db: Database, pm: ProviderManager) -> None:
    p = await _provider(db, pm)
    pid = p.require_id()
    old = pm.get_ollama_client(pid)
    await pm._rebuild_clients(p)
    assert _closed(old)


async def test_address_edits_keep_a_running_request_open(
    db: Database, pm: ProviderManager, monkeypatch: pytest.MonkeyPatch
) -> None:
    p = await _provider(db, pm)
    pid = p.require_id()
    monkeypatch.setattr(pm, "schedule_probe", lambda *a, **k: None)
    started = asyncio.Event()
    finish = asyncio.Event()
    seen: list[OllamaClient] = []

    async def request() -> None:
        async with pm.acquire_provider(pid):
            seen.append(pm.get_ollama_client(pid))
            started.set()
            await finish.wait()
            assert not _closed(seen[0])

    task = asyncio.create_task(request())
    await started.wait()
    addr = await pm.add_address(pid, "http://b.test:11434")
    await pm.update_address(addr.require_id(), "http://c.test:11434")
    await pm.toggle_address_preferred(addr.require_id())
    await pm.remove_address(addr.require_id())
    assert pm.get_ollama_client(pid) is not seen[0]
    finish.set()
    await task
    assert _closed(seen[0])


async def test_removing_the_last_address_leaves_no_client(
    db: Database, pm: ProviderManager
) -> None:
    p = await _provider(db, pm)
    pid = p.require_id()
    (addr,) = await db.get_addresses(pid)
    await pm.remove_address(addr.require_id())
    with pytest.raises(NoReachableAddressError):
        get_v1_client(pm, pid)
    with pytest.raises(NoReachableAddressError):
        pm.ollama_api_client(pid)


def test_v1_client_needs_a_backend() -> None:
    with pytest.raises(NoReachableAddressError):
        V1Client()


@pytest.mark.parametrize(
    "url",
    [
        "ftp://h.test/",
        "file:///etc/passwd",
        "http://user:pw@h.test:11434",
        "http://100.100.100.200/latest/meta-data",
        "http://169.254.169.254/",
    ],
)
async def test_provider_addresses_are_checked(
    db: Database, pm: ProviderManager, url: str
) -> None:
    p = await _provider(db, pm)
    pid = p.require_id()
    (addr,) = await db.get_addresses(pid)
    with pytest.raises(BlockedAddressError):
        await pm.add_provider("x", url)
    with pytest.raises(BlockedAddressError):
        await pm.add_address(pid, url)
    with pytest.raises(BlockedAddressError):
        await pm.update_address(addr.require_id(), url)
    assert [a.url for a in await db.get_addresses(pid)] == ["http://a.test:11434"]
    assert [x.name for x in await db.list_providers()] == ["be"]


async def test_stop_closes_retired_clients(db: Database) -> None:
    pm = ProviderManager(db)
    p = await _provider(db, pm)
    pid = p.require_id()
    async with pm.acquire_provider(pid):
        old = pm.get_ollama_client(pid)
        await pm._rebuild_clients(p)
        await pm.stop()
        assert _closed(old)
