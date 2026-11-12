"""Failover within one request: another provider after an upstream failure."""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from typing import Any

import httpx
import pytest

from llama_router import proxy, request_logger
from llama_router.auth import generate_api_key, key_prefix
from llama_router.config import settings
from llama_router.database import Database
from llama_router.llamacpp_api import deps as v1_deps
from llama_router.llamacpp_api.app import app as v1_app
from llama_router.models import (
    BenchmarkResult,
    Provider,
    ProviderModel,
    ProviderProtocol,
    ProviderStatus,
)
from llama_router.provider_manager import ProviderManager
from llama_router.provider_secrets import key_hash
from llama_router.router import Router, RoutingPreferences
from llama_router.spend import tracker_for
from tests.conftest import asgi_client

CHAT = "/v1/chat/completions"
BODY = {"model": "m", "messages": [{"role": "user", "content": "hi"}]}
URL = "http://upstream.test/v1/chat/completions"


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
    db: Database, name: str, *, startup: float, locality: str = "local"
) -> int:
    p = await db.add_provider(name, ProviderProtocol.OPENAI)
    pid = p.require_id()
    await db.update_provider_settings(pid, locality=locality)
    await db.update_provider_status(pid, ProviderStatus.IDLE)
    await db.set_provider_models(pid, [ProviderModel(provider_id=pid, name="m")])
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


async def _pair(db: Database, **second: Any) -> tuple[int, int]:
    """The first choice and the runner-up, as the router ranks them."""
    a = await _provider(db, "first", startup=100.0)
    b = await _provider(db, "second", startup=900.0, **second)
    chosen = await Router(db, ProviderManager(db)).route("m", CHAT)
    assert chosen is not None
    return (a, b) if chosen.provider.id == a else (b, a)


async def _key(db: Database, **settings_: Any) -> str:
    plaintext = generate_api_key()
    pins = settings_.pop("pins", {})
    key_id = await db.create_api_key(
        key_prefix(plaintext), key_hash(plaintext), "balanced", True, **settings_
    )
    for model, pid in pins.items():
        await db.set_api_key_model_pin(key_id, model, pid)
    return plaintext


def _status(code: int) -> httpx.HTTPStatusError:
    request = httpx.Request("POST", URL)
    return httpx.HTTPStatusError(
        f"HTTP {code}",
        request=request,
        response=httpx.Response(code, request=request, text="busy"),
    )


Failure = Callable[[], BaseException] | None


class Upstreams:
    """Fake providers: each fails as told, or answers."""

    def __init__(self) -> None:
        self.failures: dict[int, Failure] = {}
        self.called: list[int] = []
        self.stream_chunks: dict[int, list[bytes]] = {}
        self.during_call: Callable[[int], None] | None = None
        self.bodies: list[dict[str, Any]] = []

    def client(self, pm: ProviderManager, provider_id: int) -> Any:
        up = self

        class Client:
            async def chat_completions(self, body: dict[str, Any]) -> dict[str, Any]:
                up.called.append(provider_id)
                if up.during_call is not None:
                    up.during_call(provider_id)
                fail = up.failures.get(provider_id)
                if fail is not None:
                    raise fail()
                return {"choices": [{"message": {"content": f"from {provider_id}"}}]}

            async def chat_completions_stream(
                self, body: dict[str, Any]
            ) -> AsyncIterator[bytes]:
                up.called.append(provider_id)
                up.bodies.append(dict(body))
                chunks = up.stream_chunks.get(provider_id, [b"data: ok\n\n"])
                fail = up.failures.get(provider_id)
                if fail is not None and not chunks:
                    raise fail()
                for chunk in chunks:
                    yield chunk
                if fail is not None:
                    raise fail()

        return Client()


@pytest.fixture
def up(monkeypatch: pytest.MonkeyPatch) -> Upstreams:
    upstreams = Upstreams()
    monkeypatch.setattr(proxy, "get_v1_client", upstreams.client)
    return upstreams


@pytest.fixture
def logged(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    def log(db: Database, **kw: Any) -> None:
        # What the real one does synchronously: take the reservation.
        if kw.get("request") is not None:
            request_logger._take_reservation(kw["request"])
        rows.append(kw)

    monkeypatch.setattr(proxy, "log_request_background", log)
    monkeypatch.setattr(request_logger, "log_request_background", log)
    return rows


def _connect_error() -> BaseException:
    return httpx.ConnectError("refused", request=httpx.Request("POST", URL))


async def _post(
    v1: httpx.AsyncClient, key: str, body: dict[str, Any] | None = None
) -> httpx.Response:
    return await v1.post(
        CHAT, json=body or BODY, headers={"Authorization": f"Bearer {key}"}
    )


async def test_connect_error_fails_over_and_logs_both_attempts(
    db: Database, v1: httpx.AsyncClient, up: Upstreams, logged: list[dict[str, Any]]
) -> None:
    first, second = await _pair(db)
    up.failures[first] = _connect_error
    resp = await _post(v1, await _key(db))
    assert resp.status_code == 200, resp.text
    assert resp.json()["choices"][0]["message"]["content"] == f"from {second}"
    assert up.called == [first, second]
    outcomes = [(r["provider"].id, r.get("status", "ok")) for r in logged]
    assert outcomes == [(first, "error"), (second, "ok")]
    assert "refused" in logged[0]["error_detail"]


@pytest.mark.parametrize("code", [429, 502, 503])
async def test_busy_or_bad_gateway_fails_over(
    db: Database,
    v1: httpx.AsyncClient,
    up: Upstreams,
    logged: list[dict[str, Any]],
    code: int,
) -> None:
    first, second = await _pair(db)
    up.failures[first] = lambda: _status(code)
    resp = await _post(v1, await _key(db))
    assert resp.status_code == 200, resp.text
    assert up.called == [first, second]


@pytest.mark.parametrize("code", [400, 401, 403, 404, 422, 500])
async def test_other_statuses_are_answered_not_retried(
    db: Database,
    v1: httpx.AsyncClient,
    up: Upstreams,
    logged: list[dict[str, Any]],
    code: int,
) -> None:
    first, _ = await _pair(db)
    up.failures[first] = lambda: _status(code)
    resp = await _post(v1, await _key(db))
    assert resp.status_code != 200
    assert up.called == [first]


async def test_read_timeout_is_not_retried(
    db: Database, v1: httpx.AsyncClient, up: Upstreams, logged: list[dict[str, Any]]
) -> None:
    first, _ = await _pair(db)
    up.failures[first] = lambda: httpx.ReadTimeout(
        "slow", request=httpx.Request("POST", URL)
    )
    resp = await _post(v1, await _key(db))
    assert resp.status_code == 504
    assert up.called == [first]


async def test_failover_can_be_turned_off(
    db: Database,
    v1: httpx.AsyncClient,
    up: Upstreams,
    logged: list[dict[str, Any]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "upstream_failover_attempts", 0)
    first, _ = await _pair(db)
    up.failures[first] = _connect_error
    resp = await _post(v1, await _key(db))
    assert resp.status_code == 502
    assert up.called == [first]


async def test_attempts_are_limited(
    db: Database, v1: httpx.AsyncClient, up: Upstreams, logged: list[dict[str, Any]]
) -> None:
    first, second = await _pair(db)
    third = await _provider(db, "third", startup=2000.0)
    for pid in (first, second):
        up.failures[pid] = _connect_error
    resp = await _post(v1, await _key(db))
    # One extra attempt by default: the third provider is never tried.
    assert resp.status_code == 502
    assert up.called == [first, second]
    assert third not in up.called


async def test_local_only_key_never_fails_over_to_cloud(
    db: Database, v1: httpx.AsyncClient, up: Upstreams, logged: list[dict[str, Any]]
) -> None:
    lan = await _provider(db, "lan", startup=900.0)
    cloud = await _provider(db, "cloud", startup=100.0, locality="cloud")
    up.failures[lan] = _connect_error
    resp = await _post(v1, await _key(db, locality="local"))
    assert resp.status_code == 502
    assert up.called == [lan]
    assert cloud not in up.called


async def test_strict_pin_is_not_retried(
    db: Database, v1: httpx.AsyncClient, up: Upstreams, logged: list[dict[str, Any]]
) -> None:
    first, second = await _pair(db)
    up.failures[first] = _connect_error
    key = await _key(db, strict_pins=True, pins={"m": first})
    resp = await _post(v1, key)
    assert resp.status_code == 502
    assert up.called == [first]

    # A lenient pin fails over like any other request.
    up.called.clear()
    lenient = await _key(db, pins={"m": first})
    assert (await _post(v1, lenient)).status_code == 200
    assert up.called == [first, second]


async def test_capped_key_holds_one_estimate_at_a_time(
    db: Database, v1: httpx.AsyncClient, up: Upstreams, logged: list[dict[str, Any]]
) -> None:
    a = await _provider(db, "first", startup=100.0, locality="cloud")
    b = await _provider(db, "second", startup=900.0, locality="cloud")
    chosen = await Router(db, ProviderManager(db)).route("m", CHAT)
    assert chosen is not None
    first, second = (a, b) if chosen.provider.id == a else (b, a)
    for pid in (first, second):
        await db.set_model_price(pid, "m", 1.0, 1.0)
    key = await _key(db, daily_spend_cap_usd=100.0)
    key_id = (await db.list_api_keys())[0]["id"]
    up.failures[first] = _connect_error
    held: list[float] = []
    tracker = tracker_for(db)
    up.during_call = lambda pid: held.append(tracker._reserved.get(key_id, 0.0))

    resp = await _post(v1, key)
    assert resp.status_code == 200, resp.text
    assert up.called == [first, second]
    assert held[0] > 0
    # The second attempt's estimate replaced the first; never both at once.
    assert held[1] == pytest.approx(held[0])


async def test_stream_fails_over_before_the_first_chunk(
    db: Database, v1: httpx.AsyncClient, up: Upstreams, logged: list[dict[str, Any]]
) -> None:
    first, second = await _pair(db)
    up.failures[first] = _connect_error
    up.stream_chunks[first] = []
    resp = await _post(v1, await _key(db), {**BODY, "stream": True})
    assert resp.status_code == 200, resp.text
    assert resp.text == "data: ok\n\n"
    assert up.called == [first, second]
    statuses = [(r["provider"].id, r["status"]) for r in logged]
    assert statuses == [(first, "error"), (second, "ok")]


async def test_stream_failing_after_the_first_chunk_is_not_retried(
    db: Database, v1: httpx.AsyncClient, up: Upstreams, logged: list[dict[str, Any]]
) -> None:
    first, second = await _pair(db)
    up.failures[first] = _connect_error
    up.stream_chunks[first] = [b"data: partial\n\n"]
    # As before failover: the body has started, so the error ends the
    # connection (the test client re-raises it) rather than a reply.
    with pytest.raises((httpx.ConnectError, RuntimeError)):
        await _post(v1, await _key(db), {**BODY, "stream": True})
    assert up.called == [first]
    assert second not in up.called


async def test_route_excludes_failed_providers(db: Database, rt: Router) -> None:
    first, second = await _pair(db)
    third = await _provider(db, "third", startup=2000.0)
    result = await rt.route("m", CHAT)
    assert result is not None
    assert result.provider.id == first
    result = await rt.route("m", CHAT, exclude=frozenset({first}))
    assert result is not None
    assert result.provider.id == second
    result = await rt.route("m", CHAT, exclude=frozenset({first, second}))
    assert result is not None
    assert result.provider.id == third
    assert await rt.route("m", CHAT, exclude=frozenset({first, second, third})) is None
    # A lenient pin to a failed provider lets another serve it.
    pinned = RoutingPreferences(pinned_providers={"m": first})
    result = await rt.route("m", CHAT, pinned, exclude=frozenset({first}))
    assert result is not None
    assert result.provider.id == second


async def test_hub_request_for_one_provider_is_not_retried(db: Database) -> None:
    first, _ = await _pair(db)
    prefs = RoutingPreferences(hub_only=True, hub_provider=first)
    fo = proxy._Failover(
        Router(db, ProviderManager(db)), db, _fake_request(), "m", prefs, {}, None
    )
    failed = Provider(id=first, name="first")
    from llama_router.router import RouteResult

    assert await fo.next(_connect_error(), RouteResult(failed, "m"), None) is None


def _fake_request() -> Any:
    from starlette.requests import Request

    return Request(
        {
            "type": "http",
            "method": "POST",
            "path": CHAT,
            "headers": [],
            "query_string": b"",
            "scheme": "http",
            "server": ("router", 8080),
        }
    )


async def test_retried_stream_is_sent_as_the_client_sent_it(
    db: Database, v1: httpx.AsyncClient, up: Upstreams, logged: list[dict[str, Any]]
) -> None:
    cloud = await _provider(db, "cloud", startup=100.0, locality="cloud")
    lan = await _provider(db, "lan", startup=900.0)
    up.failures[cloud] = _connect_error
    up.stream_chunks[cloud] = []
    resp = await _post(v1, await _key(db), {**BODY, "stream": True})
    assert resp.status_code == 200, resp.text
    assert up.called == [cloud, lan]
    # The usage request added for the cloud provider is not passed on.
    assert up.bodies[0]["stream_options"] == {"include_usage": True}
    assert "stream_options" not in up.bodies[1]
