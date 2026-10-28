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
from tests.test_dashboard_auth import dash_client  # noqa: F401  (fixture)

CHAT = "/v1/chat/completions"
VIA = {HUB_HEADER: HUB_HEADER_VALUE}


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
    p = await db.add_provider(name, ProviderProtocol.OPENAI)
    if settings_:
        await db.update_provider_settings(p.id, **settings_)
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
    got = await db.get_provider(p.id)
    assert got is not None
    return got


def _id(p: Provider) -> int:
    assert p.id is not None
    return p.id


def _pid(result: RouteResult | None) -> int | None:
    return result.provider.id if result is not None else None


def _request(headers: dict[str, str]) -> Request:
    raw = [(k.lower().encode(), v.encode()) for k, v in headers.items()]
    return Request({"type": "http", "headers": raw, "method": "GET", "path": "/"})


async def test_providers_are_shared_by_default_and_set_as_a_whole(
    db: Database,
) -> None:
    a = await _provider(db, "a")
    b = await _provider(db, "b", models=("m", "only-b"))
    assert a.hub_shared and b.hub_shared

    await db.set_hub_shared_providers([_id(a)])
    a2, b2 = await db.get_provider(_id(a)), await db.get_provider(_id(b))
    assert a2 is not None and b2 is not None
    assert (a2.hub_shared, b2.hub_shared) == (True, False)
    names = {m["name"] for m in await db.list_all_models(hub_only=True)}
    assert names == {"m"}
    names = {m["name"] for m in await db.list_all_models()}
    assert names == {"m", "only-b"}

    await db.set_hub_shared_providers([])
    assert await db.list_all_models(hub_only=True) == []
    a3 = await db.get_provider(_id(a))
    assert a3 is not None and not a3.hub_shared


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
    pinned = RoutingPreferences(hub_only=True, pinned_providers={"m": _id(fast)})
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
    assert from_hub(_request(VIA))
    assert not from_hub(_request({HUB_HEADER: "client"}))
    assert not from_hub(_request({}))
    prefs = await routing_preferences_from_request(db, _request(VIA))
    assert prefs is not None and prefs.hub_only
    assert await routing_preferences_from_request(db, _request({})) is None


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
        assert row is not None and not row.hub_shared

    resp = await client.post("/api/hub-providers", json={"provider_ids": [9999]})
    assert resp.status_code == 400

    page = await client.get("/?tab=workers")
    assert "Providers the hub may use" in page.text
