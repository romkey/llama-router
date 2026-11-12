"""Strict pins: a key whose pinned provider can't serve the model is refused."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any, cast

import httpx
import pytest
from fastapi import HTTPException
from starlette.requests import Request

from llama_router import proxy, routing_errors
from llama_router.auth import generate_api_key, key_prefix, route_within_cap
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
from llama_router.pricing import Price
from llama_router.provider_manager import ProviderManager
from llama_router.provider_secrets import key_hash
from llama_router.router import (
    PinUnavailableError,
    Router,
    RouteResult,
    RoutingPreferences,
)
from tests.conftest import asgi_client

CHAT = "/v1/chat/completions"


@pytest.fixture
def pm(db: Database) -> ProviderManager:
    return ProviderManager(db)


async def _provider(
    db: Database,
    name: str,
    models: tuple[str, ...] = ("m",),
    *,
    locality: str = "local",
    status: ProviderStatus = ProviderStatus.IDLE,
    startup: float = 100.0,
) -> Provider:
    p = await db.add_provider(name, ProviderProtocol.OPENAI)
    await db.update_provider_settings(p.require_id(), locality=locality)
    await db.update_provider_status(p.require_id(), status)
    await db.set_provider_models(
        p.require_id(),
        [ProviderModel(provider_id=p.require_id(), name=m) for m in models],
    )
    for m in models:
        await db.save_benchmark(
            BenchmarkResult(
                provider_id=p.require_id(),
                model_name=m,
                protocol="openai",
                startup_time_ms=startup,
                tokens_per_second=50.0,
            )
        )
    found = await db.get_provider(p.require_id())
    assert found is not None
    return found


def _prefs(pins: dict[str, int], *, strict: bool, **kw: Any) -> RoutingPreferences:
    return RoutingPreferences(pinned_providers=pins, strict_pins=strict, **kw)


async def _explain(rt: Router, model: str, prefs: RoutingPreferences) -> dict[str, Any]:
    return await rt.explain(model, CHAT, prefs)


# --- Routing ---


async def test_strict_pin_to_offline_provider_is_refused(
    db: Database, pm: ProviderManager
) -> None:
    pinned = await _provider(db, "pinned", status=ProviderStatus.OFFLINE)
    await _provider(db, "other")
    rt = Router(db, pm)
    prefs = _prefs({"m": pinned.require_id()}, strict=True)

    with pytest.raises(PinUnavailableError) as exc:
        await rt.route("m", CHAT, prefs)
    assert (exc.value.model, exc.value.provider_id) == ("m", pinned.require_id())
    assert not exc.value.outside_locality

    out = await _explain(rt, "m", prefs)
    assert out["chosen"] is None
    assert out["refused"]["status"] == 503
    assert out["request"]["strict_pins"] is True
    step = out["steps"][-1]
    assert step["outcome"] == (
        "pinned provider unavailable; this key's pins are strict"
    )
    assert all(c["excluded"] for c in step["candidates"])
    assert "503" in out["summary"]


async def test_lenient_pin_to_offline_provider_falls_through(
    db: Database, pm: ProviderManager
) -> None:
    pinned = await _provider(db, "pinned", status=ProviderStatus.OFFLINE)
    other = await _provider(db, "other")
    rt = Router(db, pm)
    prefs = _prefs({"m": pinned.require_id()}, strict=False)

    result = await rt.route("m", CHAT, prefs)
    assert result is not None
    assert result.provider.id == other.require_id()
    out = await _explain(rt, "m", prefs)
    assert out["chosen"]["provider_id"] == other.require_id()
    assert "refused" not in out


async def test_strict_pin_outside_locality_is_refused(
    db: Database, pm: ProviderManager
) -> None:
    cloud = await _provider(db, "cloud", locality="cloud")
    await _provider(db, "lan")
    rt = Router(db, pm)
    prefs = _prefs({"m": cloud.require_id()}, strict=True, locality="local")

    with pytest.raises(PinUnavailableError) as exc:
        await rt.route("m", CHAT, prefs)
    assert exc.value.outside_locality

    out = await _explain(rt, "m", prefs)
    assert out["chosen"] is None
    assert out["refused"]["pinned_provider_id"] == cloud.require_id()
    # Never a cloud provider for a local-only key, strict or not.
    lenient = await rt.route("m", CHAT, _prefs({"m": cloud.require_id()}, strict=False))
    assert lenient is not None


async def test_strict_pin_stops_the_fallback_chain(
    db: Database, pm: ProviderManager
) -> None:
    pinned = await _provider(db, "pinned", ("big",), status=ProviderStatus.OFFLINE)
    small = await _provider(db, "small", ("small",))
    await db.set_model_fallback("big", "small")
    rt = Router(db, pm)

    lenient = _prefs({"big": pinned.require_id()}, strict=False, allow_fallback=True)
    result = await rt.route("big", CHAT, lenient)
    assert result is not None
    assert (result.provider.id, result.resolved_model) == (small.require_id(), "small")

    strict = _prefs({"big": pinned.require_id()}, strict=True, allow_fallback=True)
    with pytest.raises(PinUnavailableError):
        await rt.route("big", CHAT, strict)
    out = await _explain(rt, "big", strict)
    assert out["chosen"] is None
    assert [s["model"] for s in out["steps"]] == ["big"]


async def test_strict_pin_that_is_available_routes_normally(
    db: Database, pm: ProviderManager
) -> None:
    pinned = await _provider(db, "pinned", startup=900.0)
    await _provider(db, "faster", startup=10.0)
    rt = Router(db, pm)
    prefs = _prefs({"m": pinned.require_id()}, strict=True)
    result = await rt.route("m", CHAT, prefs)
    assert result is not None
    assert result.provider.id == pinned.require_id()
    out = await _explain(rt, "m", prefs)
    assert out["chosen"]["provider_id"] == pinned.require_id()


async def test_strict_pins_leave_unpinned_models_alone(
    db: Database, pm: ProviderManager
) -> None:
    pinned = await _provider(db, "pinned", ("a",), status=ProviderStatus.OFFLINE)
    other = await _provider(db, "other", ("b",))
    rt = Router(db, pm)
    result = await rt.route("b", CHAT, _prefs({"a": pinned.require_id()}, strict=True))
    assert result is not None
    assert result.provider.id == other.require_id()


async def test_hub_request_for_another_provider_is_refused_when_strict(
    db: Database, pm: ProviderManager
) -> None:
    pinned = await _provider(db, "pinned")
    other = await _provider(db, "other")
    for p in (pinned, other):
        await db.update_provider_settings(p.require_id(), hub_shared=True)
    rt = Router(db, pm)
    hub = {"hub_only": True, "hub_provider": other.require_id()}
    assert (
        await rt.route(
            "m", CHAT, _prefs({"m": pinned.require_id()}, strict=False, **hub)
        )
        is None
    )
    with pytest.raises(PinUnavailableError):
        await rt.route(
            "m", CHAT, _prefs({"m": pinned.require_id()}, strict=True, **hub)
        )


# --- Spend cap ---


def _request(path: str = CHAT) -> Request:
    return Request(
        {
            "type": "http",
            "method": "POST",
            "path": path,
            "headers": [],
            "query_string": b"",
            "scheme": "http",
            "server": ("router", 8080),
        }
    )


async def test_strict_key_already_over_cap_with_cloud_pin_gets_429(
    db: Database, pm: ProviderManager
) -> None:
    cloud = await _provider(db, "cloud", locality="cloud")
    await _provider(db, "lan")
    rt = Router(db, pm)
    # What _apply_spend_cap leaves a key past its cap with.
    prefs = _prefs(
        {"m": cloud.require_id()},
        strict=True,
        key_id=5,
        daily_spend_cap_usd=1.0,
        locality="local",
        spend_capped=True,
    )
    with pytest.raises(HTTPException) as exc:
        await route_within_cap(rt, db, _request(), "m", prefs, {})
    assert exc.value.status_code == 429


class _CloudOnly:
    """Routes everything to the cloud provider; counts what it was asked."""

    cloud = Provider(id=1, name="cloud", locality="cloud")
    local = Provider(id=2, name="lan", locality="local")

    def __init__(self) -> None:
        self.asked: list[str | None] = []

    async def route(
        self,
        model: str,
        endpoint: str,
        prefs: RoutingPreferences | None = None,
        *,
        session: str | None = None,
    ) -> RouteResult | None:
        locality = prefs.locality if prefs else None
        self.asked.append(locality)
        return RouteResult(self.local if locality == "local" else self.cloud, model)

    def price_for(self, provider_id: int, model: str) -> Price | None:
        return Price(1000.0, 1000.0) if provider_id == self.cloud.require_id() else None


async def test_strict_key_whose_estimate_passes_the_cap_is_not_rerouted(
    db: Database,
) -> None:
    body = {"messages": [{"role": "user", "content": "hello"}], "max_tokens": 1000}
    rt = _CloudOnly()
    lenient = RoutingPreferences(
        key_id=5, daily_spend_cap_usd=0.01, pinned_providers={"m": 1}
    )
    result = await route_within_cap(
        cast(Router, rt), db, _request(), "m", lenient, body
    )
    assert result is not None
    assert result.provider is rt.local
    assert rt.asked == [None, "local"]

    rt = _CloudOnly()
    strict = RoutingPreferences(
        key_id=5, daily_spend_cap_usd=0.01, pinned_providers={"m": 1}, strict_pins=True
    )
    with pytest.raises(HTTPException) as exc:
        await route_within_cap(cast(Router, rt), db, _request(), "m", strict, body)
    assert exc.value.status_code == 429
    assert rt.asked == [None]


# --- Handlers ---


@pytest.fixture
async def v1(db: Database, pm: ProviderManager) -> AsyncIterator[httpx.AsyncClient]:
    rt = Router(db, pm)
    v1_deps.init(db, pm, rt)
    async with asgi_client(v1_app) as client:
        yield client
    v1_deps._db, v1_deps._pm, v1_deps._router = None, None, None


async def _key(db: Database, *, strict: bool, pins: dict[str, int]) -> str:
    plaintext = generate_api_key()
    key_id = await db.create_api_key(
        key_prefix(plaintext),
        key_hash(plaintext),
        "balanced",
        True,
        strict_pins=strict,
    )
    for model, pid in pins.items():
        await db.set_api_key_model_pin(key_id, model, pid)
    return plaintext


class _Upstream:
    def __init__(self) -> None:
        self.called: list[int] = []

    def client(self, pm: ProviderManager, provider_id: int) -> Any:
        upstream = self

        class Client:
            async def chat_completions(self, body: dict[str, Any]) -> dict[str, Any]:
                upstream.called.append(provider_id)
                return {"choices": [{"message": {"content": "hi"}}]}

        return Client()


async def test_handler_answers_503_and_calls_no_provider(
    db: Database, v1: httpx.AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    pinned = await _provider(db, "pinned", status=ProviderStatus.OFFLINE)
    await _provider(db, "other")
    upstream = _Upstream()
    monkeypatch.setattr(proxy, "get_v1_client", upstream.client)
    logged: list[dict[str, Any]] = []
    monkeypatch.setattr(
        routing_errors, "log_request_background", lambda db, **kw: logged.append(kw)
    )
    monkeypatch.setattr(proxy, "log_request_background", lambda *a, **k: None)
    key = await _key(db, strict=True, pins={"m": pinned.require_id()})

    resp = await v1.post(
        CHAT,
        json={"model": "m", "messages": [{"role": "user", "content": "hi"}]},
        headers={"Authorization": f"Bearer {key}"},
    )
    assert resp.status_code == 503, resp.text
    assert int(resp.headers["Retry-After"]) >= 1
    assert "strict" in resp.json()["detail"]
    assert upstream.called == []
    assert len(logged) == 1
    assert logged[0]["model"] == "m"
    assert logged[0]["status"] == "error"
    assert logged[0]["error_detail"].startswith("HTTP 503")


async def test_handler_serves_elsewhere_when_pins_are_not_strict(
    db: Database, v1: httpx.AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    pinned = await _provider(db, "pinned", status=ProviderStatus.OFFLINE)
    other = await _provider(db, "other")
    upstream = _Upstream()
    monkeypatch.setattr(proxy, "get_v1_client", upstream.client)
    monkeypatch.setattr(proxy, "log_request_background", lambda *a, **k: None)
    key = await _key(db, strict=False, pins={"m": pinned.require_id()})

    resp = await v1.post(
        CHAT,
        json={"model": "m", "messages": [{"role": "user", "content": "hi"}]},
        headers={"Authorization": f"Bearer {key}"},
    )
    assert resp.status_code == 200, resp.text
    assert upstream.called == [other.require_id()]


class _PinnedFallback(_CloudOnly):
    """Cloud for the requested model; locally, its fallback is strictly
    pinned to a cloud provider."""

    async def route(
        self,
        model: str,
        endpoint: str,
        prefs: RoutingPreferences | None = None,
        *,
        session: str | None = None,
    ) -> RouteResult | None:
        locality = prefs.locality if prefs else None
        self.asked.append(locality)
        if locality == "local":
            raise PinUnavailableError(
                "fallback", 1, "it is cloud", outside_locality=True
            )
        return RouteResult(self.cloud, model)


async def test_cap_reroute_hitting_a_strict_fallback_pin_is_still_429(
    db: Database,
) -> None:
    body = {"messages": [{"role": "user", "content": "hello"}], "max_tokens": 1000}
    rt = _PinnedFallback()
    prefs = RoutingPreferences(
        key_id=5,
        daily_spend_cap_usd=0.01,
        pinned_providers={"fallback": 1},
        strict_pins=True,
    )
    with pytest.raises(HTTPException) as exc:
        await route_within_cap(cast(Router, rt), db, _request(), "m", prefs, body)
    assert exc.value.status_code == 429
    assert rt.asked == [None, "local"]
