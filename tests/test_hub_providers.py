"""Which of a worker's providers its hub may use (Provider.hub_shared)."""

from __future__ import annotations

from typing import Any

import httpx
import pytest
from starlette.requests import Request

from llama_router.api import deps as api_deps
from llama_router.api.app import app as api_app
from llama_router.auth import from_hub, routing_preferences_from_request
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
from llama_router.router import Router, RouteResult, RoutingPreferences
from llama_router.worker_link.protocol import HUB_HEADER, HUB_HEADER_VALUE
from llama_router.worker_link.relay import (
    RELAY_HEADER,
    HubRelayMiddleware,
    relay_secret,
)

CHAT = "/v1/chat/completions"
# What this router's own worker link sends with a relayed job.
VIA = {HUB_HEADER: HUB_HEADER_VALUE, RELAY_HEADER: relay_secret()}


@pytest.fixture
def pm(db: Database) -> ProviderManager:
    return ProviderManager(db)


async def _provider(
    db: Database,
    name: str,
    *,
    models: tuple[str, ...] = ("m",),
    startup: float = 100.0,
    **settings_: Any,
) -> Provider:
    pid = (await db.add_provider(name, ProviderProtocol.OPENAI)).require_id()
    if settings_:
        await db.update_provider_settings(pid, **settings_)
    await db.update_provider_status(pid, ProviderStatus.IDLE)
    await db.set_provider_models(
        pid, [ProviderModel(provider_id=pid, name=m) for m in models]
    )
    for m in models:
        await db.save_benchmark(
            BenchmarkResult(
                provider_id=pid,
                model_name=m,
                protocol="openai",
                startup_time_ms=startup,
                tokens_per_second=50.0,
            )
        )
    got = await db.get_provider(pid)
    assert got is not None
    return got


def _pid(result: RouteResult | None) -> int | None:
    return result.provider.id if result is not None else None


async def _request(headers: dict[str, str]) -> Request:
    """A request as the routes see it, after HubRelayMiddleware."""
    raw = [(k.lower().encode(), v.encode()) for k, v in headers.items()]
    seen: list[Any] = []

    async def app(scope: Any, receive: Any, send: Any) -> None:
        seen.append(scope)

    scope = {"type": "http", "headers": raw, "method": "GET", "path": "/"}
    await HubRelayMiddleware(app)(scope, None, None)  # type: ignore[arg-type]
    return Request(seen[0])


async def test_providers_are_shared_by_default_and_set_as_a_whole(
    db: Database,
) -> None:
    a = await _provider(db, "a")
    b = await _provider(db, "b", models=("m", "only-b"))
    assert a.hub_shared
    assert b.hub_shared

    await db.set_hub_shared_providers([a.require_id()])
    a2 = await db.get_provider(a.require_id())
    b2 = await db.get_provider(b.require_id())
    assert a2 is not None
    assert b2 is not None
    assert (a2.hub_shared, b2.hub_shared) == (True, False)
    names = {m["name"] for m in await db.list_all_models(hub_only=True)}
    assert names == {"m"}
    names = {m["name"] for m in await db.list_all_models()}
    assert names == {"m", "only-b"}

    await db.set_hub_shared_providers([])
    assert await db.list_all_models(hub_only=True) == []
    a3 = await db.get_provider(a.require_id())
    assert a3 is not None
    assert not a3.hub_shared


async def test_hub_requests_route_only_to_shared_providers(
    db: Database, pm: ProviderManager
) -> None:
    fast = await _provider(db, "fast", startup=100.0, hub_shared=False)
    slow = await _provider(db, "slow", startup=5000.0)
    rt = Router(db, pm)
    hub = RoutingPreferences(hub_only=True)

    assert _pid(await rt.route("m", CHAT)) == fast.id
    assert _pid(await rt.route("m", CHAT, hub)) == slow.id
    out = await rt.explain("m", CHAT, hub)
    assert out["request"]["hub_only"] is True
    rows = {c["provider"]: c for c in out["steps"][0]["candidates"]}
    assert rows["fast"]["excluded"] == "provider is not shared with the hub"
    assert rows["slow"]["excluded"] is None

    # A pin to a provider kept from the hub is ignored, like a wrong locality.
    pins = {"m": fast.require_id()}
    pinned = RoutingPreferences(hub_only=True, pinned_providers=pins)
    assert _pid(await rt.route("m", CHAT, pinned)) == slow.id
    # Chaos mode draws from the shared providers only.
    chaos = RoutingPreferences(mode="chaos", hub_only=True)
    for _ in range(10):
        assert _pid(await rt.route("m", CHAT, chaos)) == slow.id

    await db.set_hub_shared_providers([])
    assert await rt.route("m", CHAT, hub) is None
    assert _pid(await rt.route("m", CHAT)) == fast.id


async def test_hub_scope_combines_with_key_locality(
    db: Database, pm: ProviderManager
) -> None:
    await _provider(db, "lan", hub_shared=False)
    cloud = await _provider(db, "api", locality="cloud")
    rt = Router(db, pm)
    assert (
        await rt.route("m", CHAT, RoutingPreferences(hub_only=True, locality="local"))
        is None
    )
    got = await rt.route("m", CHAT, RoutingPreferences(hub_only=True, locality="cloud"))
    assert _pid(got) == cloud.id


async def test_via_header_sets_hub_scope(db: Database) -> None:
    await db.set_allow_unauthenticated(True)
    assert from_hub(await _request(VIA))
    assert not from_hub(await _request({HUB_HEADER: "client"}))
    assert not from_hub(await _request({}))
    prefs = await routing_preferences_from_request(db, await _request(VIA))
    assert prefs is not None
    assert prefs.hub_only
    assert await routing_preferences_from_request(db, await _request({})) is None


async def test_model_listings_hide_unshared_providers_from_the_hub(
    db: Database, pm: ProviderManager
) -> None:
    await db.set_allow_unauthenticated(True)
    await _provider(db, "mine", models=("private",), hub_shared=False)
    await _provider(db, "shared", models=("public",))
    rt = Router(db, pm)
    api_deps.init(db, pm, rt)
    v1_deps.init(db, pm, rt)

    async def get(
        app: Any, path: str, headers: dict[str, str] | None = None
    ) -> httpx.Response:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://r") as c:
            return await c.get(path, headers=headers)

    tags = (await get(api_app, "/api/tags")).json()["models"]
    assert {m["name"] for m in tags} == {"private", "public"}
    tags = (await get(api_app, "/api/tags", VIA)).json()["models"]
    assert {m["name"] for m in tags} == {"public"}
    models = (await get(v1_app, "/v1/models", VIA)).json()["data"]
    assert {m["id"] for m in models} == {"public"}
    assert (await get(v1_app, "/v1/models/private", VIA)).status_code == 404
    assert (await get(v1_app, "/v1/models/private")).status_code == 200

    # /api/ps lists loaded models of busy providers: not the unshared one.
    for p in await db.list_providers():
        pm.acquire(p.require_id())
    try:
        ps = (await get(api_app, "/api/ps")).json()["models"]
        assert {m["name"] for m in ps} == {"private", "public"}
        ps = (await get(api_app, "/api/ps", VIA)).json()["models"]
        assert {m["name"] for m in ps} == {"public"}
    finally:
        for p in await db.list_providers():
            pm.release(p.require_id())


async def test_dashboard_hub_provider_api(dash_client: Any) -> None:
    client, db = dash_client
    lan = await _provider(db, "lan")
    cloud = await _provider(db, "api", locality="cloud")
    remote = await db.add_provider("edge", worker_name="edge")

    rows = (await client.get("/api/hub-providers")).json()["providers"]
    assert {r["name"]: r["kind"] for r in rows} == {
        "lan": "local",
        "api": "cloud",
        "edge": "remote",
    }
    assert all(r["hub_shared"] for r in rows)

    resp = await client.post("/api/hub-providers", json={"provider_ids": [lan.id]})
    assert resp.status_code == 200
    shared = {r["name"]: r["hub_shared"] for r in resp.json()["providers"]}
    assert shared == {"lan": True, "api": False, "edge": False}
    for pid in (cloud.id, remote.id):
        assert pid is not None
        row = await db.get_provider(pid)
        assert row is not None
        assert not row.hub_shared

    resp = await client.post("/api/hub-providers", json={"provider_ids": [9999]})
    assert resp.status_code == 400

    page = await client.get("/?tab=workers")
    assert "Providers the hub may use" in page.text


async def test_provider_header_narrows_hub_jobs_to_one_provider(
    db: Database, pm: ProviderManager
) -> None:
    """A worker that shares providers individually names the one a job is for."""
    from llama_router.auth import hub_provider
    from llama_router.worker_link.protocol import PROVIDER_HEADER

    await db.set_allow_unauthenticated(True)
    fast = await _provider(db, "fast", models=("m", "only-fast"), startup=100.0)
    slow = await _provider(db, "slow", startup=5000.0)
    kept = await _provider(db, "kept", models=("m", "secret"), hub_shared=False)
    rt = Router(db, pm)
    api_deps.init(db, pm, rt)
    v1_deps.init(db, pm, rt)

    to_slow = {**VIA, PROVIDER_HEADER: str(slow.require_id())}
    assert hub_provider(await _request(to_slow)) == slow.id
    # Without the Via header, or with junk, it means nothing.
    assert hub_provider(await _request({PROVIDER_HEADER: str(slow.id)})) is None
    assert hub_provider(await _request({**VIA, PROVIDER_HEADER: "x"})) is None

    prefs = await routing_preferences_from_request(db, await _request(to_slow))
    assert prefs is not None
    assert prefs.hub_provider == slow.id
    assert _pid(await rt.route("m", CHAT, prefs)) == slow.id
    # No falling back to another provider for a model it lacks.
    assert await rt.route("only-fast", CHAT, prefs) is None
    # A provider kept from the hub stays kept, even when named.
    to_kept = RoutingPreferences(hub_only=True, hub_provider=kept.require_id())
    assert await rt.route("m", CHAT, to_kept) is None
    to_fast = RoutingPreferences(hub_only=True, hub_provider=fast.require_id())
    assert _pid(await rt.route("only-fast", CHAT, to_fast)) == fast.id

    async def get(app: Any, path: str, headers: dict[str, str]) -> httpx.Response:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://r") as c:
            return await c.get(path, headers=headers)

    tags = (await get(api_app, "/api/tags", to_slow)).json()["models"]
    assert {m["name"] for m in tags} == {"m"}
    to_fast_h = {**VIA, PROVIDER_HEADER: str(fast.id)}
    models = (await get(v1_app, "/v1/models", to_fast_h)).json()["data"]
    assert {m["id"] for m in models} == {"m", "only-fast"}
    to_kept_h = {**VIA, PROVIDER_HEADER: str(kept.id)}
    assert (await get(api_app, "/api/tags", to_kept_h)).json()["models"] == []

    for p in await db.list_providers():
        pm.acquire(p.require_id())
    try:
        ps = (await get(api_app, "/api/ps", to_slow)).json()["models"]
        assert {m["name"] for m in ps} == {"m"}
    finally:
        for p in await db.list_providers():
            pm.release(p.require_id())


async def test_hub_headers_from_clients_count_for_nothing(
    db: Database, pm: ProviderManager
) -> None:
    """Only the worker link in this process can mark a job as the hub's: the
    headers without its secret are ignored, and stripped either way."""
    from llama_router.auth import hub_provider
    from llama_router.worker_link.protocol import PROVIDER_HEADER
    from llama_router.worker_link.relay import SCOPE_KEY

    await db.set_allow_unauthenticated(True)
    forged = {HUB_HEADER: HUB_HEADER_VALUE, PROVIDER_HEADER: "1"}
    for headers in (forged, {**forged, RELAY_HEADER: "guess"}):
        req = await _request(headers)
        assert not from_hub(req)
        assert hub_provider(req) is None
        assert await routing_preferences_from_request(db, req) is None
        assert HUB_HEADER not in req.headers
        assert PROVIDER_HEADER not in req.headers
        assert RELAY_HEADER not in req.headers

    genuine = await _request({**VIA, PROVIDER_HEADER: "1"})
    assert from_hub(genuine)
    assert hub_provider(genuine) == 1
    assert RELAY_HEADER not in genuine.headers
    assert HUB_HEADER not in genuine.headers

    # A scope that arrives already marked is not believed either.
    seen: list[Any] = []

    async def app(scope: Any, receive: Any, send: Any) -> None:
        seen.append(scope)

    marked = {"type": "http", "headers": [], SCOPE_KEY: {"provider": 1}}
    await HubRelayMiddleware(app)(marked, None, None)  # type: ignore[arg-type]
    assert SCOPE_KEY not in seen[0]

    # Over the API: a client that forges the headers sees everything, as
    # it would without them, and is not narrowed to a provider it names.
    await _provider(db, "mine", models=("private",), hub_shared=False)
    shared = await _provider(db, "shared", models=("public",))
    rt = Router(db, pm)
    api_deps.init(db, pm, rt)
    v1_deps.init(db, pm, rt)
    transport = httpx.ASGITransport(app=api_app)
    async with httpx.AsyncClient(transport=transport, base_url="http://r") as c:
        to_shared = {HUB_HEADER: HUB_HEADER_VALUE, PROVIDER_HEADER: str(shared.id)}
        tags = (await c.get("/api/tags", headers=to_shared)).json()["models"]
    assert {m["name"] for m in tags} == {"private", "public"}


async def test_hub_provider_jobs_keep_the_keys_pins(
    db: Database, pm: ProviderManager
) -> None:
    fast = await _provider(db, "fast", startup=100.0)
    slow = await _provider(db, "slow", startup=5000.0)
    rt = Router(db, pm)
    to_slow = RoutingPreferences(hub_only=True, hub_provider=slow.require_id())
    assert _pid(await rt.route("m", CHAT, to_slow)) == slow.id
    pinned_away = RoutingPreferences(
        hub_only=True,
        hub_provider=slow.require_id(),
        pinned_providers={"m": fast.require_id()},
    )
    assert await rt.route("m", CHAT, pinned_away) is None
    pinned_here = RoutingPreferences(
        hub_only=True,
        hub_provider=slow.require_id(),
        pinned_providers={"m": slow.require_id()},
    )
    assert _pid(await rt.route("m", CHAT, pinned_here)) == slow.id


@pytest.mark.parametrize(
    ("url", "ok"),
    [
        ("http://127.0.0.1:11434", True),
        ("http://localhost:8080", True),
        ("http://[::1]:8080", True),
        ("http://router.localhost", True),
        ("http://192.168.1.5:11434", False),
        ("https://ollama.example.org", False),
        ("not a url", False),
    ],
)
def test_relay_secret_only_goes_to_this_machine(url: str, ok: bool) -> None:
    from llama_router.worker_link.relay import is_loopback_url

    assert is_loopback_url(url) is ok
