"""Request limits: body size, n/max_tokens caps, upstream slots and deadlines."""

from __future__ import annotations

import asyncio
import gc
import json
import time
from collections.abc import AsyncIterator
from typing import Any

import httpx
import pytest
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pytest_httpx import HTTPXMock

from llama_router.api import deps as api_deps
from llama_router.api.app import app as ollama_app
from llama_router.config import settings
from llama_router.dashboard.app import app as dashboard_app
from llama_router.database import Database
from llama_router.decision_policy import _decide_in_slot
from llama_router.limits import (
    BodySizeLimitMiddleware,
    DeadlineExceeded,
    UpstreamSlot,
    clamp_generation_params,
    limiter,
    upstream_slot,
)
from llama_router.llamacpp_api import deps as v1_deps
from llama_router.llamacpp_api.app import app as v1_app
from llama_router.models import (
    Provider,
    ProviderModel,
    ProviderProtocol,
    ProviderStatus,
)
from llama_router.provider_manager import ProviderManager
from llama_router.request_logger import stream_response
from llama_router.router import Router, RoutingPreferences

MODEL = "m"
BASE = "http://be.test:11434"


def _client(app: Any) -> httpx.AsyncClient:
    transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
    return httpx.AsyncClient(transport=transport, base_url="http://router")


def _echo_app() -> FastAPI:
    app = FastAPI()
    app.add_middleware(BodySizeLimitMiddleware)

    @app.post("/{path:path}")
    async def echo(request: Request) -> JSONResponse:
        return JSONResponse({"size": len(await request.body())})

    return app


def _request(state: dict[str, Any] | None = None) -> Request:
    return Request(
        {
            "type": "http",
            "headers": [],
            "method": "POST",
            "path": "/",
            "state": state or {},
        }
    )


def _prefs(key_id: int) -> RoutingPreferences:
    return RoutingPreferences(key_id=key_id)


async def _sleep_in(slot: UpstreamSlot, seconds: float) -> None:
    async with slot:
        await asyncio.sleep(seconds)


# --- body size ---------------------------------------------------------------


def test_middleware_installed_on_every_app() -> None:
    for app in (ollama_app, v1_app, dashboard_app):
        classes: list[object] = [m.cls for m in app.user_middleware]
        assert BodySizeLimitMiddleware in classes


async def test_declared_length_over_limit_is_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "max_body_bytes", 100)
    async with _client(_echo_app()) as c:
        ok = await c.post("/api/chat", content=b"x" * 100)
        big = await c.post("/api/chat", content=b"x" * 101)
    assert ok.status_code == 200
    assert ok.json() == {"size": 100}
    assert big.status_code == 413


async def test_chunked_body_is_cut_off_at_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "max_body_bytes", 100)

    async def chunks() -> AsyncIterator[bytes]:
        for _ in range(10):
            yield b"x" * 30

    async with _client(_echo_app()) as c:
        resp = await c.post("/api/chat", content=chunks())
    assert resp.status_code == 413


async def test_media_endpoints_get_the_larger_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "max_body_bytes", 100)
    monkeypatch.setattr(settings, "max_media_body_bytes", 1000)
    async with _client(_echo_app()) as c:
        media = await c.post("/v1/images/edits", content=b"x" * 500)
        other = await c.post("/v1/chat/completions", content=b"x" * 500)
        too_big = await c.post("/v1/audio/transcriptions", content=b"x" * 1001)
    assert media.status_code == 200
    assert other.status_code == 413
    assert too_big.status_code == 413


async def test_zero_turns_the_body_limit_off(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "max_body_bytes", 0)
    async with _client(_echo_app()) as c:
        resp = await c.post("/api/chat", content=b"x" * 5000)
    assert resp.status_code == 200


async def test_api_app_refuses_oversized_body(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "max_body_bytes", 10)
    async with _client(v1_app) as c:
        resp = await c.post("/v1/chat/completions", content=b"x" * 11)
    assert resp.status_code == 413


# --- n / max_tokens ----------------------------------------------------------


def test_clamp_generation_params() -> None:
    body: dict[str, Any] = {
        "n": 10,
        "best_of": 9,
        "max_tokens": 10**9,
        "max_completion_tokens": 10**9,
        "max_output_tokens": 50,
        "options": {"num_predict": -1},
    }
    clamp_generation_params(body)
    assert body == {
        "n": settings.max_n,
        "best_of": settings.max_n,
        "max_tokens": settings.max_tokens_cap,
        "max_completion_tokens": settings.max_tokens_cap,
        "max_output_tokens": 50,
        "options": {"num_predict": settings.max_tokens_cap},
    }


def test_clamp_leaves_absent_and_odd_fields_alone() -> None:
    body: dict[str, Any] = {"n": True, "max_tokens": "lots", "options": None}
    clamp_generation_params(body)
    assert body == {"n": True, "max_tokens": "lots", "options": None}
    clamp_generation_params(["not", "a", "dict"])  # no error


def test_clamp_can_be_turned_off(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "max_n", 0)
    monkeypatch.setattr(settings, "max_tokens_cap", 0)
    body: dict[str, Any] = {"n": 10, "max_tokens": 10**9}
    clamp_generation_params(body)
    assert body == {"n": 10, "max_tokens": 10**9}


# --- routes ------------------------------------------------------------------


@pytest.fixture
async def pm(db: Database) -> AsyncIterator[ProviderManager]:
    manager = ProviderManager(db)
    yield manager
    await manager.stop()


@pytest.fixture
async def provider(db: Database, pm: ProviderManager) -> Provider:
    await db.set_allow_unauthenticated(True)
    added = await db.add_provider("be", ProviderProtocol.OLLAMA)
    pid = added.id
    assert pid is not None
    await db.add_address(pid, BASE, is_preferred=True)
    await db.update_provider_status(pid, ProviderStatus.IDLE)
    await db.set_provider_models(pid, [ProviderModel(provider_id=pid, name=MODEL)])
    p = await db.get_provider(pid)
    assert p is not None
    await pm._rebuild_clients(p)
    rt = Router(db, pm)
    api_deps.init(db, pm, rt)
    v1_deps.init(db, pm, rt)
    return p


async def test_chat_request_is_clamped_upstream(
    provider: Provider, httpx_mock: HTTPXMock
) -> None:
    httpx_mock.add_response(url=BASE + "/v1/chat/completions", json={"choices": []})
    body = {"model": MODEL, "messages": [], "n": 50, "max_tokens": 10**7}
    async with _client(v1_app) as c:
        resp = await c.post("/v1/chat/completions", json=body)
    assert resp.status_code == 200
    (sent,) = httpx_mock.get_requests()
    forwarded = json.loads(sent.content)
    assert forwarded["n"] == settings.max_n
    assert forwarded["max_tokens"] == settings.max_tokens_cap


@pytest.mark.parametrize(
    ("app", "path"),
    [
        (ollama_app, "/api/chat"),
        (ollama_app, "/api/generate"),
        (ollama_app, "/api/embed"),
        (v1_app, "/v1/chat/completions"),
        (v1_app, "/v1/completions"),
        (v1_app, "/v1/responses"),
        (v1_app, "/v1/embeddings"),
        (v1_app, "/v1/images/generations"),
        (v1_app, "/v1/images/edits"),
        (v1_app, "/v1/audio/speech"),
        (v1_app, "/v1/audio/transcriptions"),
        (v1_app, "/v1/decisions"),
    ],
)
async def test_key_is_checked_before_body_is_parsed(
    provider: Provider, db: Database, app: FastAPI, path: str
) -> None:
    await db.set_allow_unauthenticated(False)
    async with _client(app) as c:
        resp = await c.post(path, content=b"{not json")
    assert resp.status_code == 401


# --- upstream slots ----------------------------------------------------------


async def test_global_limit_waits_then_429(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "upstream_max_concurrency", 1)
    monkeypatch.setattr(settings, "upstream_queue_timeout_seconds", 0.05)
    first = await upstream_slot(_request(), None)
    with pytest.raises(HTTPException) as exc:
        await upstream_slot(_request(), None)
    assert exc.value.status_code == 429
    first.release()
    second = await upstream_slot(_request(), None)
    second.release()


async def test_waiter_gets_a_slot_when_one_frees(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "upstream_max_concurrency", 1)
    monkeypatch.setattr(settings, "upstream_queue_timeout_seconds", 5)
    first = await upstream_slot(_request(), None)
    waiter = asyncio.create_task(upstream_slot(_request(), None))
    await asyncio.sleep(0.01)
    assert not waiter.done()
    async with first:
        pass
    second = await asyncio.wait_for(waiter, 1)
    second.release()


async def test_per_key_limit_is_per_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "upstream_max_concurrency_per_key", 1)
    monkeypatch.setattr(settings, "upstream_queue_timeout_seconds", 0)
    a = await upstream_slot(_request(), _prefs(1))
    with pytest.raises(HTTPException) as exc:
        await upstream_slot(_request(), _prefs(1))
    assert exc.value.status_code == 429
    b = await upstream_slot(_request(), _prefs(2))
    # Requests without a key share only the global limit.
    anon = [await upstream_slot(_request(), None) for _ in range(3)]
    for slot in (a, b, *anon):
        slot.release()


async def test_stream_cap_per_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "max_streams_per_key", 2)
    streams = [
        await upstream_slot(_request(), _prefs(1), stream=True) for _ in range(2)
    ]
    with pytest.raises(HTTPException) as exc:
        await upstream_slot(_request(), _prefs(1), stream=True)
    assert exc.value.status_code == 429
    # Non-streamed calls and other keys are not affected.
    (await upstream_slot(_request(), _prefs(1))).release()
    (await upstream_slot(_request(), _prefs(2), stream=True)).release()
    streams[0].release()
    streams[0].release()  # releasing twice is harmless
    assert limiter.active_streams(1) == 1
    (await upstream_slot(_request(), _prefs(1), stream=True)).release()


async def test_unused_slot_is_released_when_collected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "max_streams_per_key", 1)
    slot = await upstream_slot(_request(), _prefs(7), stream=True)
    assert limiter.active_streams(7) == 1
    del slot
    gc.collect()
    assert limiter.active_streams(7) == 0


# --- deadlines ---------------------------------------------------------------


async def test_non_streamed_call_past_deadline_is_504(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "request_deadline_seconds", 0.05)
    slot = await upstream_slot(_request(), _prefs(1))
    with pytest.raises(DeadlineExceeded) as exc:
        await _sleep_in(slot, 1)
    assert exc.value.status_code == 504
    assert limiter._per_key[1]._value == settings.upstream_max_concurrency_per_key


async def test_deadline_counts_from_arrival(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "request_deadline_seconds", 10)
    arrived = time.monotonic() - 9.95
    slot = await upstream_slot(_request({"received_at": arrived}), None)
    with pytest.raises(DeadlineExceeded):
        await _sleep_in(slot, 1)


async def _slow(n: int, delay: float) -> AsyncIterator[bytes]:
    for i in range(n):
        await asyncio.sleep(delay)
        yield f"chunk{i}\n".encode()


async def test_stream_past_deadline_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "request_deadline_seconds", 0.1)
    slot = await upstream_slot(_request(), None, stream=True)
    got: list[bytes] = []

    async def consume() -> None:
        async with slot:
            async for chunk in slot.stream(_slow(100, 0.03)):
                got.append(chunk)

    with pytest.raises(DeadlineExceeded):
        await consume()
    assert 0 < len(got) < 100


class _Logged:
    def __init__(self, inner: AsyncIterator[bytes]) -> None:
        self._inner = inner

    def __aiter__(self) -> AsyncIterator[bytes]:
        return self._inner


async def test_stream_response_ends_body_at_deadline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "request_deadline_seconds", 0.1)
    slot = await upstream_slot(_request(), None, stream=True)

    async def gen() -> AsyncIterator[bytes]:
        async with slot:
            async for chunk in slot.stream(_slow(100, 0.03)):
                yield chunk

    resp = await stream_response(_Logged(gen()), "text/plain")  # type: ignore[arg-type]
    assert resp.status_code == 200
    body = b"".join([c async for c in resp.body_iterator])  # type: ignore[attr-defined]
    assert body.startswith(b"chunk0\n")


async def test_stream_past_deadline_before_first_chunk_is_504(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "request_deadline_seconds", 0.05)
    slot = await upstream_slot(_request(), None, stream=True)

    async def gen() -> AsyncIterator[bytes]:
        async with slot:
            async for chunk in slot.stream(_slow(1, 1)):
                yield chunk

    with pytest.raises(DeadlineExceeded):
        await stream_response(_Logged(gen()), "text/plain")  # type: ignore[arg-type]


async def test_429_carries_retry_after(
    provider: Provider, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(settings, "upstream_max_concurrency", 1)
    monkeypatch.setattr(settings, "upstream_queue_timeout_seconds", 0)
    held = await upstream_slot(_request(), None)
    body = {"model": MODEL, "messages": [], "stream": False}
    async with _client(v1_app) as c:
        resp = await c.post("/v1/chat/completions", json=body)
    held.release()
    assert resp.status_code == 429
    assert resp.headers["retry-after"] == "1"


async def test_decision_policy_call_takes_a_slot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "upstream_max_concurrency", 1)
    monkeypatch.setattr(settings, "upstream_queue_timeout_seconds", 0)
    held = await upstream_slot(_request(), None)
    with pytest.raises(HTTPException) as exc:
        # Refused before any of the other arguments are used.
        await _decide_in_slot(None, None, None, None, _request(), None)  # type: ignore[arg-type]
    held.release()
    assert exc.value.status_code == 429
