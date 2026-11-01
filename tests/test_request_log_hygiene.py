"""Request log retention and noise, circuit-breaker samples, and API hardening."""

from __future__ import annotations

from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta
from typing import Any

import httpx
import pytest
from fastapi import FastAPI
from pytest_httpx import HTTPXMock
from starlette.requests import Request

from llama_router import request_logger
from llama_router.api.app import app as ollama_app
from llama_router.auth import (
    generate_api_key,
    invalidate_auth_cache,
    key_hash,
    key_hash_legacy,
    key_prefix,
    routing_preferences_from_request,
)
from llama_router.background import drain
from llama_router.config import settings
from llama_router.database import Database
from llama_router.llamacpp_api import deps as v1_deps
from llama_router.llamacpp_api.app import app as v1_app
from llama_router.models import (
    Provider,
    ProviderModel,
    ProviderProtocol,
    ProviderStatus,
    RequestLog,
)
from llama_router.provider_manager import ProviderManager
from llama_router.router import Router

BASE = "http://be.test:8080"


def _client(app: FastAPI) -> httpx.AsyncClient:
    transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
    return httpx.AsyncClient(transport=transport, base_url="http://router")


def _request(api_key_id: int | None = None) -> Request:
    req = Request({"type": "http", "headers": [], "path": "/x"})
    if api_key_id is not None:
        req.state.api_key_id = api_key_id
    return req


async def _key(db: Database, legacy: bool = False) -> tuple[int, str]:
    raw = generate_api_key()
    digest = key_hash_legacy(raw) if legacy else key_hash(raw)
    key_id = await db.create_api_key(key_prefix(raw), digest, "balanced", True)
    return key_id, raw


@pytest.fixture
async def v1_provider(db: Database, pm: ProviderManager) -> Provider:
    added = await db.add_provider("be", ProviderProtocol.OPENAI)
    pid = added.require_id()
    await db.add_address(pid, BASE, is_preferred=True)
    await db.update_provider_status(pid, ProviderStatus.IDLE)
    await db.set_provider_models(pid, [ProviderModel(provider_id=pid, name="m")])
    p = await db.get_provider(pid)
    assert p is not None
    await pm._rebuild_clients(p)
    v1_deps.init(db, pm, Router(db, pm))
    return p


# --- Anonymous 404s ---


def test_refusals_worth_logging() -> None:
    assert request_logger.refusal_worth_logging(_request(), 401)
    assert request_logger.refusal_worth_logging(_request(), 400)
    assert not request_logger.refusal_worth_logging(_request(), 404)
    assert request_logger.refusal_worth_logging(_request(5), 404)
    assert not request_logger.refusal_worth_logging(_request(5), 500)


async def test_anonymous_404_is_not_logged(db: Database, router: Router) -> None:
    async with _client(v1_app) as c:
        assert (await c.get("/wp-login.php")).status_code == 404
    async with _client(ollama_app) as c:
        assert (await c.get("/.env")).status_code == 404
    assert await db.count_request_logs() == 0


# --- Docs pages ---


@pytest.mark.parametrize("app", [v1_app, ollama_app])
@pytest.mark.parametrize("path", ["/docs", "/redoc", "/openapi.json"])
async def test_public_apis_have_no_docs(
    app: FastAPI, path: str, db: Database, router: Router
) -> None:
    async with _client(app) as c:
        assert (await c.get(path)).status_code == 404


# --- Retention ---


async def test_old_request_logs_are_purged(
    db: Database, pm: ProviderManager, monkeypatch: pytest.MonkeyPatch
) -> None:
    for _ in range(2):
        await db.save_request_log(
            RequestLog(protocol="v1", endpoint="/v1/x", source_ip="1.2.3.4")
        )
    await db._execute(
        "UPDATE request_log SET created_at = ? "
        "WHERE id = (SELECT MIN(id) FROM request_log)",
        ((datetime.now(UTC) - timedelta(days=40)).strftime("%Y-%m-%d %H:%M:%S"),),
    )
    monkeypatch.setattr(settings, "request_log_retention_days", 30.0)
    assert await pm.purge_request_logs() == 1
    assert await db.count_request_logs() == 1
    # Runs at most hourly unless forced.
    assert await pm.purge_request_logs() == 0
    monkeypatch.setattr(settings, "request_log_retention_days", 0.0)
    assert await pm.purge_request_logs(force=True) == 0
    assert await db.count_request_logs() == 1


# --- Circuit breaker samples ---


def _samples(pm: ProviderManager, pid: int) -> int:
    return pm.stats.provider_health(pid).request_samples


async def test_oversized_max_tokens_timeout_is_not_a_provider_failure(
    pm: ProviderManager, v1_provider: Provider, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(settings, "circuit_timeout_max_tokens", 4096)
    pid = v1_provider.require_id()
    timeout = "ReadTimeout for http://be.test/v1/chat/completions: "
    for body in (
        {"max_tokens": 100_000},
        {"options": {"num_predict": -1}},
        {"max_completion_tokens": 5000},
    ):
        request_logger._record_outcome(
            v1_provider,
            "m",
            status="error",
            error_detail=timeout,
            request_body=body,
        )
    assert _samples(pm, pid) == 0
    # A modest request that timed out does count, as does any other failure.
    request_logger._record_outcome(
        v1_provider,
        "m",
        status="error",
        error_detail=timeout,
        request_body={"max_tokens": 100},
    )
    request_logger._record_outcome(
        v1_provider,
        "m",
        status="error",
        error_detail="ConnectError: refused",
        request_body={"max_tokens": 100_000},
    )
    assert _samples(pm, pid) == 2


async def test_cancelled_request_is_not_sampled(
    pm: ProviderManager, v1_provider: Provider
) -> None:
    request_logger._record_outcome(
        v1_provider, "m", status=request_logger.STATUS_CANCELLED, error_detail=None
    )
    assert _samples(pm, v1_provider.require_id()) == 0


async def test_client_disconnect_mid_stream_is_logged_as_cancelled(
    db: Database, pm: ProviderManager, v1_provider: Provider
) -> None:
    async def chunks() -> AsyncIterator[bytes]:
        yield b"data: one\n\n"
        yield b"data: two\n\n"

    logged = request_logger.StreamLogger(
        chunks(),
        db=db,
        provider=v1_provider,
        protocol="v1",
        endpoint="/v1/chat/completions",
        request=_request(),
        model="m",
        request_body={"model": "m"},
        start_time=0.0,
    )
    stream = logged.__aiter__()
    assert await stream.__anext__() == b"data: one\n\n"
    await stream.aclose()  # what Starlette does when the client goes away
    await drain()
    assert _samples(pm, v1_provider.require_id()) == 0
    [entry] = await db.get_request_logs(10)
    assert entry.status == request_logger.STATUS_CANCELLED


# --- /v1/audio/voices ---


async def test_voices_needs_a_key(db: Database, router: Router) -> None:
    await db.set_allow_unauthenticated(False)
    invalidate_auth_cache(db)
    async with _client(v1_app) as c:
        assert (await c.get("/v1/audio/voices")).status_code == 401
        bad = await c.get("/v1/audio/voices", headers={"API-KEY": "lrk_nope"})
        assert bad.status_code == 401


async def test_voices_are_cached(
    db: Database, v1_provider: Provider, httpx_mock: HTTPXMock
) -> None:
    await db.set_allow_unauthenticated(False)
    invalidate_auth_cache(db)
    _, raw = await _key(db)
    httpx_mock.add_response(
        url=BASE + "/v1/audio/voices", json={"voices": [{"voice_id": "a"}]}
    )
    async with _client(v1_app) as c:
        for _ in range(3):
            resp = await c.get("/v1/audio/voices", headers={"API-KEY": raw})
            assert resp.json() == {"voices": [{"voice_id": "a"}]}
    assert len(httpx_mock.get_requests()) == 1


# --- Legacy key hashes ---


async def test_legacy_key_hash_is_upgraded(db: Database) -> None:
    key_id, raw = await _key(db, legacy=True)
    req = Request(
        {"type": "http", "headers": [(b"api-key", raw.encode())], "path": "/"}
    )
    prefs = await routing_preferences_from_request(db, req)
    assert prefs is not None
    assert prefs.key_id == key_id
    assert await db.lookup_api_key(key_hash_legacy(raw)) is None
    record: dict[str, Any] | None = await db.lookup_api_key(key_hash(raw))
    assert record is not None
    assert record["id"] == key_id
    # Still accepted after the upgrade, once the cache forgets it.
    invalidate_auth_cache(db)
    again = await routing_preferences_from_request(db, req)
    assert again is not None
    assert again.key_id == key_id
