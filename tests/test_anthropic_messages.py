"""Anthropic Messages pass-through (``/v1/messages``)."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import httpx
import pytest

from llama_router import proxy, request_logger
from llama_router.auth import generate_api_key, key_prefix
from llama_router.database import Database
from llama_router.llamacpp_api import deps as v1_deps
from llama_router.llamacpp_api.app import app as v1_app
from llama_router.llamacpp_client import OpenAIClient
from llama_router.models import (
    BenchmarkResult,
    ProviderModel,
    ProviderProtocol,
    ProviderStatus,
)
from llama_router.provider_manager import ProviderManager
from llama_router.provider_secrets import key_hash
from llama_router.request_logger import usage_from_body, usage_from_messages_stream
from llama_router.router import Router
from llama_router.v1_client import V1Client
from tests.conftest import asgi_client

MSG = "/v1/messages"
BODY = {
    "model": "claude-x",
    "max_tokens": 64,
    "system": "be brief",
    "messages": [{"role": "user", "content": [{"type": "text", "text": "hi"}]}],
}
REPLY = {
    "id": "msg_1",
    "type": "message",
    "role": "assistant",
    "content": [{"type": "text", "text": "hello"}],
    "usage": {
        "input_tokens": 10,
        "output_tokens": 5,
        "cache_read_input_tokens": 100,
        "cache_creation_input_tokens": 3,
    },
}
STREAM = [
    b'event: message_start\ndata: {"type":"message_start","message":{"usage":'
    b'{"input_tokens":12,"cache_read_input_tokens":8,"output_tokens":1}}}\n\n',
    b'event: content_block_delta\ndata: {"type":"content_block_delta",'
    b'"delta":{"type":"text_delta","text":"Hel"}}\n\n',
    b'event: message_delta\ndata: {"type":"message_delta","usage":'
    b'{"output_tokens":7}}\n\n',
    b'event: message_stop\ndata: {"type":"message_stop"}\n\n',
]


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


class Upstream:
    def __init__(self) -> None:
        self.calls: list[tuple[int, dict[str, Any], dict[str, str], str]] = []

    def client(self, pm: ProviderManager, provider_id: int) -> Any:
        up = self

        class Client:
            async def messages(
                self, body: dict[str, Any], headers: dict[str, str], path: str = MSG
            ) -> dict[str, Any]:
                up.calls.append((provider_id, dict(body), dict(headers), path))
                if path.endswith("count_tokens"):
                    return {"input_tokens": 42}
                return REPLY

            async def messages_stream(
                self, body: dict[str, Any], headers: dict[str, str]
            ) -> AsyncIterator[bytes]:
                up.calls.append((provider_id, dict(body), dict(headers), MSG))
                for chunk in STREAM:
                    yield chunk

        return Client()


@pytest.fixture
def up(monkeypatch: pytest.MonkeyPatch) -> Upstream:
    u = Upstream()
    monkeypatch.setattr(proxy, "get_v1_client", u.client)
    return u


@pytest.fixture
def logged(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    monkeypatch.setattr(
        request_logger, "log_request_background", lambda db, **kw: rows.append(kw)
    )
    monkeypatch.setattr(
        proxy, "log_request_background", lambda db, **kw: rows.append(kw)
    )
    return rows


async def _provider(
    db: Database,
    name: str,
    *,
    messages: str | None = "available",
    locality: str = "local",
    startup: float = 100.0,
) -> int:
    p = await db.add_provider(name, ProviderProtocol.OPENAI)
    pid = p.require_id()
    await db.update_provider_status(pid, ProviderStatus.IDLE)
    await db.update_provider_settings(pid, locality=locality)
    await db.set_provider_models(
        pid, [ProviderModel(provider_id=pid, name="claude-x", raw_name="backend-x")]
    )
    if messages is not None:
        await db.set_provider_capabilities(
            pid,
            {
                "endpoints": [
                    {"path": MSG, "method": "POST", "state": messages},
                    {
                        "path": "/v1/messages/count_tokens",
                        "method": "POST",
                        "state": messages,
                    },
                ]
            },
        )
    await db.save_benchmark(
        BenchmarkResult(
            provider_id=pid,
            model_name="claude-x",
            protocol="openai",
            startup_time_ms=startup,
        )
    )
    return pid


async def _key(db: Database, **kw: Any) -> str:
    plaintext = generate_api_key()
    await db.create_api_key(
        key_prefix(plaintext), key_hash(plaintext), "balanced", True, **kw
    )
    return plaintext


async def test_provider_without_the_route_is_never_chosen(
    db: Database, v1: httpx.AsyncClient, up: Upstream, logged: list[Any]
) -> None:
    serves = await _provider(db, "serves", startup=900.0)
    await _provider(db, "unknown", messages=None, startup=10.0)
    await _provider(db, "missing", messages="missing", startup=10.0)
    key = await _key(db)
    for _ in range(3):
        resp = await v1.post(MSG, json=BODY, headers={"x-api-key": key})
        assert resp.status_code == 200, resp.text
    assert {c[0] for c in up.calls} == {serves}


async def test_x_api_key_authenticates_and_is_not_forwarded(
    db: Database, v1: httpx.AsyncClient, up: Upstream, logged: list[Any]
) -> None:
    await _provider(db, "serves")
    key = await _key(db)
    resp = await v1.post(
        MSG,
        json=BODY,
        headers={
            "x-api-key": key,
            "anthropic-version": "2023-06-01",
            "anthropic-beta": "tools-2024",
        },
    )
    assert resp.status_code == 200
    assert resp.json() == REPLY
    _, body, headers, _ = up.calls[0]
    assert headers == {
        "anthropic-version": "2023-06-01",
        "anthropic-beta": "tools-2024",
    }
    assert body["model"] == "backend-x"
    assert body["system"] == "be brief"
    assert body["messages"] == BODY["messages"]
    assert key not in str(up.calls)


async def test_the_client_send_path_never_carries_the_client_key() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json=REPLY)

    lcpp = OpenAIClient(
        "http://up.test/v1",
        transport=httpx.MockTransport(handler),
        headers={"x-api-key": "provider-secret"},
    )
    out = await V1Client(llamacpp=lcpp).messages(
        dict(BODY), {"anthropic-version": "2023-06-01"}
    )
    assert out == REPLY
    req = seen[0]
    assert req.url.path == "/v1/messages"
    assert req.headers["x-api-key"] == "provider-secret"
    assert req.headers["anthropic-version"] == "2023-06-01"
    await lcpp.close()


async def test_stream_bytes_are_relayed_unchanged_and_logged(
    db: Database, v1: httpx.AsyncClient, up: Upstream, logged: list[dict[str, Any]]
) -> None:
    await _provider(db, "serves")
    key = await _key(db)
    resp = await v1.post(MSG, json={**BODY, "stream": True}, headers={"x-api-key": key})
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/event-stream")
    assert resp.content == b"".join(STREAM)
    row = logged[-1]
    assert (row["prompt_tokens"], row["completion_tokens"]) == (20, 7)
    assert row["ttfb_ms"] is not None


def test_usage_parsing() -> None:
    assert usage_from_body(REPLY) == (113, 5)
    assert usage_from_messages_stream(STREAM[0], b"".join(STREAM[2:])) == (20, 7)
    assert usage_from_body({"usage": {"prompt_tokens": 4, "completion_tokens": 2}}) == (
        4,
        2,
    )


async def test_router_errors_have_anthropics_shape(
    db: Database, v1: httpx.AsyncClient, up: Upstream, logged: list[Any]
) -> None:
    unauth = await v1.post(MSG, json=BODY)
    assert unauth.status_code == 401
    assert unauth.json()["type"] == "error"
    assert unauth.json()["error"]["type"] == "authentication_error"
    key = await _key(db)
    await _provider(db, "unknown", messages=None)
    missing = await v1.post(MSG, json=BODY, headers={"x-api-key": key})
    assert missing.status_code == 404
    err = missing.json()["error"]
    assert err["type"] == "not_found_error"
    assert "Messages API" in err["message"]
    assert "claude-x" in err["message"]
    bad = await v1.post(MSG, json={"messages": []}, headers={"x-api-key": key})
    assert bad.status_code == 400
    assert bad.json()["error"]["type"] == "invalid_request_error"


async def test_count_tokens_is_forwarded_and_not_metered(
    db: Database, v1: httpx.AsyncClient, up: Upstream, logged: list[Any]
) -> None:
    await _provider(db, "serves")
    uid = await db.create_dashboard_user("ann", "x", False)
    await db.set_dashboard_user_limits(uid, {"requests_per_minute": 1})
    key = await _key(db, owner_user_id=uid)
    body = {"model": "claude-x", "messages": BODY["messages"]}
    for _ in range(3):
        resp = await v1.post(
            "/v1/messages/count_tokens", json=body, headers={"x-api-key": key}
        )
        assert resp.json() == {"input_tokens": 42}
    assert up.calls[0][3] == "/v1/messages/count_tokens"
    # Messages themselves are metered: the second gets 429, Anthropic-shaped.
    assert (
        await v1.post(MSG, json=BODY, headers={"x-api-key": key})
    ).status_code == 200
    limited = await v1.post(MSG, json=BODY, headers={"x-api-key": key})
    assert limited.status_code == 429
    assert limited.json()["error"]["type"] == "rate_limit_error"


async def test_local_only_key_never_reaches_the_cloud_provider(
    db: Database, v1: httpx.AsyncClient, up: Upstream, logged: list[Any]
) -> None:
    cloud = await _provider(db, "anthropic", locality="cloud", startup=1.0)
    local = await _provider(db, "local", startup=900.0)
    key = await _key(db, locality="local")
    for _ in range(3):
        assert (
            await v1.post(MSG, json=BODY, headers={"x-api-key": key})
        ).status_code == 200
    assert {c[0] for c in up.calls} == {local}
    assert cloud not in {c[0] for c in up.calls}


async def test_capacity_applies(
    db: Database,
    pm: ProviderManager,
    rt: Router,
    v1: httpx.AsyncClient,
    up: Upstream,
    logged: list[Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from llama_router.config import settings

    monkeypatch.setattr(settings, "upstream_queue_timeout_seconds", 0.05)
    pid = await _provider(db, "serves")
    await db.update_provider_settings(pid, max_concurrent=1)
    await rt.route("claude-x", MSG)
    gate = await pm.take_slot(pid)
    key = await _key(db)
    full = await v1.post(MSG, json=BODY, headers={"x-api-key": key})
    assert full.status_code == 429
    assert full.json()["error"]["type"] == "rate_limit_error"
    assert gate is not None
    gate.release()


async def test_affinity_fingerprint_includes_system() -> None:
    from llama_router.affinity import session_key

    a = session_key({}, BODY, 1, None)
    b = session_key({}, {**BODY, "system": "be verbose"}, 1, None)
    assert a is not None
    assert b is not None
    assert a != b


def test_count_tokens_is_never_charged() -> None:
    from llama_router import request_logger as rl
    from llama_router.models import Provider
    from llama_router.pricing import Price

    rl.attach_pricer(lambda pid, model: Price(1_000_000.0, 1_000_000.0))
    try:
        entry = rl._build_entry(
            Database("sqlite+aiosqlite:///:memory:"),
            provider=Provider(id=1, name="anthropic", locality="cloud"),
            protocol="v1",
            endpoint="/v1/messages/count_tokens",
            model="claude-x",
            request_body={"model": "claude-x", "messages": []},
            response_body={"input_tokens": 42},
            duration_ms=1.0,
        )
    finally:
        rl.attach_pricer(None)
    assert entry.cost_usd is None


async def test_large_bodies_reach_the_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    from llama_router import json_io

    monkeypatch.setattr(json_io, "LARGE_JSON_BYTES", 10)
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        if request.headers.get("accept") == "text/event-stream":
            return httpx.Response(200, content=b"".join(STREAM))
        return httpx.Response(200, json=REPLY)

    lcpp = OpenAIClient("http://up.test/v1", transport=httpx.MockTransport(handler))
    client = V1Client(llamacpp=lcpp)
    hdrs = {"anthropic-version": "2023-06-01"}
    assert await client.messages(dict(BODY), hdrs) == REPLY
    got = b"".join([c async for c in client.messages_stream(dict(BODY), hdrs)])
    assert got
    for req in seen:
        assert req.headers["content-type"] == "application/json"
        assert req.headers["anthropic-version"] == "2023-06-01"
    await lcpp.close()
