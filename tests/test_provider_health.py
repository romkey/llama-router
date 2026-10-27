"""Provider health checks, manual refresh, and their Log entries."""

from __future__ import annotations

import httpx
import pytest

from llama_router.models import ProviderStatus, ProviderProtocol
from llama_router.provider_manager import ProviderManager

pytestmark = pytest.mark.httpx_mock(
    assert_all_responses_were_requested=False,
    can_send_already_matched_responses=True,
)


def _ollama_ok(request: httpx.Request) -> httpx.Response:
    path = request.url.path
    if path == "/api/version":
        return httpx.Response(200, json={"version": "0.12.0"})
    if path == "/api/tags":
        return httpx.Response(200, json={"models": [{"name": "tiny:1b", "size": 1}]})
    if path == "/api/ps":
        return httpx.Response(200, json={"models": []})
    return httpx.Response(404)


def _ollama_auth_required(request: httpx.Request) -> httpx.Response:
    # Version answers without a key, listing models does not.
    if request.url.path == "/api/version":
        return httpx.Response(200, json={"version": "0.12.0"})
    return httpx.Response(401, json={"error": "unauthorized"})


def _routes(httpx_mock, by_host: dict) -> None:
    def dispatch(request: httpx.Request):
        handler = by_host[request.url.host]
        if isinstance(handler, Exception):
            raise handler
        return handler(request)

    httpx_mock.add_callback(dispatch, is_reusable=True, is_optional=True)


async def _provider(db, name: str, url: str, status: ProviderStatus):
    p = await db.add_provider(name, ProviderProtocol.OLLAMA)
    await db.add_address(p.id, url, True)
    await db.update_provider_status(p.id, status)
    return await db.get_provider(p.id)


async def _system_logs(db):
    return [e for e in await db.get_request_logs(100, 0) if e.protocol == "system"]


async def test_refresh_marks_addresses_live_and_logs(db, httpx_mock) -> None:
    _routes(httpx_mock, {"good": _ollama_ok})
    p = await _provider(db, "good", "http://good:11434", ProviderStatus.OFFLINE)
    pm = ProviderManager(db)

    status = await pm.refresh_provider(p.id)

    assert status == ProviderStatus.IDLE
    assert (await db.get_provider(p.id)).status == ProviderStatus.IDLE
    assert all(a.is_live for a in await db.get_addresses(p.id))
    assert [m.name for m in await db.get_provider_models(p.id)] == ["tiny:1b"]
    logs = await _system_logs(db)
    assert [(e.endpoint, e.status) for e in logs] == [("refresh", "ok")]
    await pm.stop()


async def test_failing_provider_does_not_block_others(db, httpx_mock) -> None:
    _routes(httpx_mock, {"locked": _ollama_auth_required, "good": _ollama_ok})
    locked = await _provider(
        db, "locked", "http://locked:11434", ProviderStatus.OFFLINE
    )
    good = await _provider(db, "good", "http://good:11434", ProviderStatus.OFFLINE)
    pm = ProviderManager(db)

    await pm._run_health_checks()

    assert (await db.get_provider(good.id)).status == ProviderStatus.IDLE
    # Reachable but unusable: address is live, provider stays out of routing.
    assert (await db.get_provider(locked.id)).status == ProviderStatus.OFFLINE
    assert all(a.is_live for a in await db.get_addresses(locked.id))
    detail = pm.status_detail(locked.id)
    assert detail and "authentication required" in detail
    info = await pm.get_provider_info(locked.id)
    assert info.status_detail == detail
    await pm.stop()


async def test_unreachable_provider_goes_offline_and_logs_once(db, httpx_mock) -> None:
    _routes(httpx_mock, {"gone": httpx.ConnectError("refused")})
    p = await _provider(db, "gone", "http://gone:11434", ProviderStatus.IDLE)
    pm = ProviderManager(db)

    await pm._run_health_checks()
    await pm._run_health_checks()

    assert (await db.get_provider(p.id)).status == ProviderStatus.OFFLINE
    assert not any(a.is_live for a in await db.get_addresses(p.id))
    assert pm.status_detail(p.id) is None
    logs = await _system_logs(db)
    assert [(e.endpoint, e.status) for e in logs] == [("health", "error")]
    await pm.stop()


async def test_health_logs_only_transitions(db, httpx_mock) -> None:
    _routes(httpx_mock, {"good": _ollama_ok})
    p = await _provider(db, "good", "http://good:11434", ProviderStatus.OFFLINE)
    pm = ProviderManager(db)

    await pm._run_health_checks()
    await pm._run_health_checks()

    assert (await db.get_provider(p.id)).status == ProviderStatus.IDLE
    logs = await _system_logs(db)
    assert [(e.endpoint, e.status) for e in logs] == [("health", "ok")]
    await pm.stop()


async def test_restart_does_not_relog_already_offline_provider(db, httpx_mock) -> None:
    _routes(httpx_mock, {"gone": httpx.ConnectError("refused")})
    # Offline in the database from the previous run, still unreachable now.
    await _provider(db, "gone", "http://gone:11434", ProviderStatus.OFFLINE)
    pm = ProviderManager(db)

    await pm._check_many(await db.list_providers(), rediscover=True, source="startup")
    await pm._run_health_checks()

    assert await _system_logs(db) == []
    await pm.stop()


async def test_rebuild_uses_this_cycles_liveness(db, httpx_mock) -> None:
    _routes(httpx_mock, {"dead": httpx.ConnectError("refused"), "backup": _ollama_ok})
    p = await _provider(db, "two", "http://dead:11434", ProviderStatus.IDLE)
    dead = (await db.get_addresses(p.id))[0]
    await db.set_address_live(dead.id, True)  # stale: it was live last cycle
    await db.add_address(p.id, "http://backup:11434", False)
    pm = ProviderManager(db)
    await pm._rebuild_clients(p)
    assert pm._active_urls[p.id] == "http://dead:11434"

    status = await pm.check_provider(p)

    assert status == ProviderStatus.IDLE
    assert pm._active_urls[p.id] == "http://backup:11434"
    await pm.stop()
