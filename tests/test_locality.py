"""Local and cloud providers, and API keys limited to one of them."""

from __future__ import annotations

import pytest
from starlette.requests import Request

from llama_router.auth import (
    generate_api_key,
    key_hash,
    key_prefix,
    routing_preferences_from_request,
)
from llama_router.models import (
    BenchmarkResult,
    ProviderModel,
    ProviderProtocol,
    ProviderStatus,
)
from llama_router.provider_manager import ProviderManager
from llama_router.request_logger import log_request
from llama_router.router import Router, RoutingPreferences
from llama_router.upstream_url import guess_locality

CHAT = "/v1/chat/completions"


@pytest.fixture
def pm(db) -> ProviderManager:
    return ProviderManager(db)


async def _provider(db, name: str, locality: str, models=("m",), startup=100.0):
    p = await db.add_provider(name, ProviderProtocol.OPENAI)
    await db.update_provider_settings(p.id, locality=locality)
    await db.update_provider_status(p.id, ProviderStatus.IDLE)
    await db.set_provider_models(
        p.id, [ProviderModel(provider_id=p.id, name=m) for m in models]
    )
    for m in models:
        await db.save_benchmark(
            BenchmarkResult(
                provider_id=p.id,
                model_name=m,
                protocol="openai",
                startup_time_ms=startup,
                tokens_per_second=50.0,
            )
        )
    return await db.get_provider(p.id)


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        ("http://192.168.1.5:11434", "local"),
        ("http://gpu:8080", "local"),
        ("http://box.tail0.ts.net:8080", "local"),
        ("http://100.70.1.2:11434", "local"),
        ("worker://edge", "local"),
        ("https://api.openai.com/v1", "cloud"),
        ("http://203.0.113.9:8080", "cloud"),
    ],
)
def test_guess_locality(url: str, expected: str) -> None:
    assert guess_locality(url) == expected


async def test_key_locality_limits_candidates(db, pm) -> None:
    near = await _provider(db, "near", "local", startup=5000)
    far = await _provider(db, "far", "cloud", startup=60)
    rt = Router(db, pm)
    assert (await rt.route("m", CHAT)).provider.id == far.id
    local = RoutingPreferences(locality="local")
    assert (await rt.route("m", CHAT, local)).provider.id == near.id
    cloud_only = RoutingPreferences(locality="cloud")
    assert (await rt.route("m", CHAT, cloud_only)).provider.id == far.id


async def test_local_key_reaches_a_worse_tier_local_provider(db, pm) -> None:
    near = await _provider(db, "near", "local")
    await _provider(db, "far", "cloud")
    # The local provider's probe found the endpoint missing: it is in a worse
    # tier, so the unrestricted ranking holds only the cloud provider.
    await db.set_provider_capabilities(
        near.id, {"endpoints": [{"path": CHAT, "state": "missing"}]}
    )
    rt = Router(db, pm)
    assert (await rt.route("m", CHAT)).provider.name == "far"
    r = await rt.route("m", CHAT, RoutingPreferences(locality="local"))
    assert r.provider.id == near.id


async def test_no_local_provider_means_no_route(db, pm) -> None:
    await _provider(db, "far", "cloud")
    rt = Router(db, pm)
    assert await rt.route("m", CHAT, RoutingPreferences(locality="local")) is None
    out = await rt.explain("m", CHAT, RoutingPreferences(locality="local"))
    assert out["chosen"] is None
    (row,) = out["steps"][0]["candidates"]
    assert row["excluded"] == "provider is cloud; the key requires local"
    assert out["steps"][0]["locality"] == "key requires local; 1 cloud provider skipped"
    assert out["request"]["locality"] == "local"


async def test_pin_to_wrong_locality_is_ignored(db, pm) -> None:
    near = await _provider(db, "near", "local")
    far = await _provider(db, "far", "cloud")
    rt = Router(db, pm)
    prefs = RoutingPreferences(locality="local", pinned_providers={"m": far.id})
    assert (await rt.route("m", CHAT, prefs)).provider.id == near.id
    out = await rt.explain("m", CHAT, prefs)
    assert "far is cloud" in out["steps"][0]["pin_ignored"]
    # A pin within the key's locality still wins.
    prefs = RoutingPreferences(locality="cloud", pinned_providers={"m": far.id})
    assert (await rt.route("m", CHAT, prefs)).provider.id == far.id


async def test_chaos_and_fallback_respect_locality(db, pm) -> None:
    await _provider(db, "near", "local", models=("small",))
    await _provider(db, "far", "cloud", models=("big",))
    await db.set_model_fallback("big", "small")
    rt = Router(db, pm)
    local = RoutingPreferences(locality="local")
    r = await rt.route("big", CHAT, local)
    assert (r.provider.name, r.resolved_model) == ("near", "small")
    chaos = RoutingPreferences(mode="chaos", locality="local")
    for _ in range(10):
        assert (await rt.route("x", CHAT, chaos)).provider.name == "near"


async def test_session_moves_when_its_provider_is_the_wrong_locality(db, pm) -> None:
    await _provider(db, "near", "local", startup=5000)
    far = await _provider(db, "far", "cloud", startup=60)
    rt = Router(db, pm)
    session = "s1"
    assert (await rt.route("m", CHAT, session=session)).provider.id == far.id
    local = RoutingPreferences(locality="local")
    assert (await rt.route("m", CHAT, local, session=session)).provider.name == "near"


def _request(headers: dict[str, str]) -> Request:
    raw = [(k.lower().encode(), v.encode()) for k, v in headers.items()]
    return Request({"type": "http", "headers": raw, "method": "POST", "path": "/"})


async def test_locality_header_narrows_but_never_widens(db) -> None:
    async def key(locality):
        plain = generate_api_key()
        await db.create_api_key(
            key_prefix(plain), key_hash(plain), "balanced", True, locality=locality
        )
        return plain

    any_key, cloud_key = await key(None), await key("cloud")
    h = "X-Llama-Router-Locality"
    prefs = await routing_preferences_from_request(db, _request({"API-KEY": any_key}))
    assert prefs.locality is None
    prefs = await routing_preferences_from_request(
        db, _request({"API-KEY": any_key, h: "local"})
    )
    assert prefs.locality == "local"
    # Cached preferences are not changed by one request's header.
    prefs = await routing_preferences_from_request(db, _request({"API-KEY": any_key}))
    assert prefs.locality is None
    prefs = await routing_preferences_from_request(
        db, _request({"API-KEY": cloud_key, h: "local"})
    )
    assert prefs.locality == "cloud"
    prefs = await routing_preferences_from_request(
        db, _request({"API-KEY": cloud_key, h: "bogus"})
    )
    assert prefs.locality == "cloud"


async def test_log_records_locality(db) -> None:
    far = await _provider(db, "far", "cloud")
    await log_request(
        db, provider=far, protocol="v1", endpoint=CHAT, model="m", duration_ms=5
    )
    (entry,) = await db.get_request_logs(10, 0)
    assert entry.locality == "cloud"


async def test_add_provider_guesses_locality(db, httpx_mock) -> None:
    import httpx

    httpx_mock.add_exception(httpx.ConnectError("offline"), is_reusable=True)
    pm = ProviderManager(db)
    await pm.add_provider("a", "http://10.0.0.2:11434")
    await pm.add_provider("b", "https://api.openai.com/v1", ProviderProtocol.OPENAI)
    await pm.add_provider(
        "c",
        "https://home.example.com",
        provider_settings={"locality": "local"},
    )
    await pm.stop()
    got = {p.name: p.locality for p in await db.list_providers()}
    assert got == {"a": "local", "b": "cloud", "c": "local"}


async def test_dashboard_key_locality(dash_client) -> None:
    client, db = dash_client
    r = await client.post(
        "/api/keys/generate", json={"routing_mode": "balanced", "locality": "local"}
    )
    assert r.status_code == 200
    assert r.json()["locality"] == "local"
    (k,) = await db.list_api_keys()
    assert k["locality"] == "local"
    r = await client.post(f"/api/keys/{k['id']}/locality", json={"locality": "any"})
    assert r.status_code == 200
    assert (await db.list_api_keys())[0]["locality"] is None
    r = await client.post(f"/api/keys/{k['id']}/locality", json={"locality": "moon"})
    assert r.status_code == 400
    r = await client.post("/api/keys/999/locality", json={"locality": "cloud"})
    assert r.status_code == 404
    page = await client.get("/")
    assert "setKeyLocality(" in page.text
