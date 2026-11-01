"""Model groups as clients see them: listings, proxying, logs and metrics."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import httpx
import pytest
from fastapi import FastAPI
from pytest_httpx import HTTPXMock

from llama_router import metrics
from llama_router.api import deps as api_deps
from llama_router.api.app import app as ollama_app
from llama_router.background import drain
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
from llama_router.router import Router
from llama_router.worker_link.protocol import HUB_HEADER, HUB_HEADER_VALUE
from llama_router.worker_link.relay import RELAY_HEADER, relay_token

FAST = "http://fast.test:8080"
SLOW = "http://slow.test:8080"


@pytest.fixture
async def pm(db: Database) -> AsyncIterator[ProviderManager]:
    manager = ProviderManager(db)
    yield manager
    await manager.stop()


async def _provider(
    db: Database, pm: ProviderManager, name: str, url: str, model: str, startup: float
) -> Provider:
    added = await db.add_provider(name, ProviderProtocol.OPENAI)
    pid = added.require_id()
    await db.add_address(pid, url, is_preferred=True)
    await db.update_provider_status(pid, ProviderStatus.IDLE)
    await db.set_provider_models(pid, [ProviderModel(provider_id=pid, name=model)])
    await db.save_benchmark(
        BenchmarkResult(
            provider_id=pid,
            model_name=model,
            protocol="openai",
            startup_time_ms=startup,
            tokens_per_second=50.0,
        )
    )
    p = await db.get_provider(pid)
    assert p is not None
    await pm._rebuild_clients(p)
    return p


@pytest.fixture
async def setup(db: Database, pm: ProviderManager) -> tuple[Router, Provider, Provider]:
    await db.set_allow_unauthenticated(True)
    fast = await _provider(db, pm, "fast", FAST, "small-coder", 100.0)
    slow = await _provider(db, pm, "slow", SLOW, "big-coder", 5000.0)
    await db.set_hub_shared_providers([slow.require_id()])
    g = await db.create_model_group("coding", "models for code")
    assert g.id is not None
    await db.add_model_group_member(g.id, "big-coder")
    await db.add_model_group_member(g.id, "small-coder")
    empty = await db.create_model_group("nothing")
    assert empty.id is not None
    await db.add_model_group_member(empty.id, "nobody-serves-this")
    rt = Router(db, pm)
    await rt.rebuild()
    api_deps.init(db, pm, rt)
    v1_deps.init(db, pm, rt)
    return rt, fast, slow


def _client(app: FastAPI) -> httpx.AsyncClient:
    transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
    return httpx.AsyncClient(transport=transport, base_url="http://router")


async def test_tags_and_v1_models_list_usable_groups(
    setup: tuple[Router, Provider, Provider],
) -> None:
    async with _client(ollama_app) as c:
        tags = (await c.get("/api/tags")).json()["models"]
    assert [m["name"] for m in tags] == ["big-coder", "coding", "small-coder"]
    coding = next(m for m in tags if m["name"] == "coding")
    assert coding["details"] == {
        "family": "group",
        "group": True,
        "members": ["big-coder", "small-coder"],
    }

    async with _client(v1_app) as c:
        listing = (await c.get("/v1/models")).json()["data"]
        one = await c.get("/v1/models/coding")
        empty = await c.get("/v1/models/nothing")
    assert [m["id"] for m in listing] == ["big-coder", "coding", "small-coder"]
    assert one.json()["id"] == "coding"
    assert empty.status_code == 404


async def test_hub_lists_a_group_only_through_shared_providers(
    setup: tuple[Router, Provider, Provider], db: Database
) -> None:
    hub = {HUB_HEADER: HUB_HEADER_VALUE, RELAY_HEADER: relay_token()}
    async with _client(ollama_app) as c:
        tags = (await c.get("/api/tags", headers=hub)).json()["models"]
    assert [m["name"] for m in tags] == ["big-coder", "coding"]

    await db.set_hub_shared_providers([])
    rt, _, _ = setup
    await rt.rebuild()
    async with _client(ollama_app) as c:
        tags = (await c.get("/api/tags", headers=hub)).json()["models"]
    assert tags == []


async def test_a_group_hides_a_model_of_its_name_even_when_unusable(
    setup: tuple[Router, Provider, Provider], db: Database
) -> None:
    rt, fast, _ = setup
    # A provider starts serving a model named like the empty group: routing
    # sends that name to the group, so listings must not offer the model.
    fid = fast.require_id()
    await db.set_provider_models(
        fid,
        [
            ProviderModel(provider_id=fid, name="small-coder"),
            ProviderModel(provider_id=fid, name="nothing"),
        ],
    )
    await rt.rebuild()
    assert await rt.route("nothing", "/v1/chat/completions") is None
    async with _client(v1_app) as c:
        listing = (await c.get("/v1/models")).json()["data"]
        one = await c.get("/v1/models/nothing")
    assert "nothing" not in [m["id"] for m in listing]
    assert one.status_code == 404


async def test_group_request_runs_the_member_and_logs_both_names(
    setup: tuple[Router, Provider, Provider], db: Database, httpx_mock: HTTPXMock
) -> None:
    httpx_mock.add_response(
        url=FAST + "/v1/chat/completions", json={"choices": [], "model": "x"}
    )
    async with _client(v1_app) as c:
        resp = await c.post(
            "/v1/chat/completions",
            json={"model": "coding", "messages": [], "stream": False},
        )
    assert resp.status_code == 200
    sent = httpx_mock.get_requests()[0]
    assert b'"model":"small-coder"' in sent.content.replace(b" ", b"")
    await drain()
    logs = await db.get_request_logs(10, 0)
    assert [(e.model, e.routed_model) for e in logs] == [("coding", "small-coder")]


async def test_direct_request_logs_no_routed_model(
    setup: tuple[Router, Provider, Provider], db: Database, httpx_mock: HTTPXMock
) -> None:
    httpx_mock.add_response(url=FAST + "/v1/chat/completions", json={"choices": []})
    async with _client(v1_app) as c:
        await c.post(
            "/v1/chat/completions",
            json={"model": "small-coder", "messages": [], "stream": False},
        )
    await drain()
    logs = await db.get_request_logs(10, 0)
    assert [(e.model, e.routed_model) for e in logs] == [("small-coder", None)]


async def test_metrics_count_group_routes(
    setup: tuple[Router, Provider, Provider], pm: ProviderManager
) -> None:
    rt, fast, _ = setup
    for _ in range(3):
        await rt.route("coding", "/v1/chat/completions")
    await rt.explain("coding", "/v1/chat/completions")  # not counted
    text = metrics.render(
        metrics.collect(await pm.list_provider_infos(), pm.stats, router=rt)
    )
    labels = (
        f'provider="fast",provider_id="{fast.id}",protocol="openai",'
        'locality="local",group="coding",model="small-coder"'
    )
    assert f"llama_router_model_group_routes_total{{{labels}}} 3" in text
    assert 'llama_router_model_group_available{group="coding"} 1' in text
    assert 'llama_router_model_group_available{group="nothing"} 0' in text


def test_listing_without_a_table_falls_back_to_models() -> None:
    from llama_router.routing_table import RoutingTable

    table = RoutingTable()
    assert table.group_listing() == []
    rows: list[dict[str, Any]] = table.group_listing(hub_only=True)
    assert rows == []
