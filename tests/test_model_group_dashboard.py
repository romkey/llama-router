"""Dashboard API and page for model groups."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import httpx
import pytest
from fastapi import FastAPI

from llama_router.dashboard import deps as dash_deps
from llama_router.dashboard.model_groups import router as groups_router
from llama_router.database import Database
from llama_router.models import (
    BenchmarkResult,
    Provider,
    ProviderModel,
    ProviderProtocol,
    ProviderStatus,
)
from llama_router.provider_manager import ProviderManager
from llama_router.router import Router


@pytest.fixture
async def client(
    db: Database, pm: ProviderManager
) -> AsyncIterator[tuple[httpx.AsyncClient, Router]]:
    rt = Router(db, pm)
    dash_deps.init(db, pm)
    app = FastAPI()
    app.include_router(groups_router)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://dash") as c:
        yield c, rt


async def _provider(db: Database, name: str, models: dict[str, float]) -> Provider:
    p = await db.add_provider(name, ProviderProtocol.OPENAI)
    pid = p.require_id()
    await db.update_provider_status(pid, ProviderStatus.IDLE)
    await db.set_provider_models(
        pid, [ProviderModel(provider_id=pid, name=m) for m in models]
    )
    for m, startup in models.items():
        await db.save_benchmark(
            BenchmarkResult(
                provider_id=pid,
                model_name=m,
                protocol="openai",
                startup_time_ms=startup,
                tokens_per_second=50.0,
            )
        )
    return p


async def _listing(c: httpx.AsyncClient, rt: Router) -> dict[str, Any]:
    await rt.rebuild()
    resp = await c.get("/api/model-groups")
    assert resp.status_code == 200
    body: dict[str, Any] = resp.json()
    return body


async def test_create_list_and_route(
    client: tuple[httpx.AsyncClient, Router], db: Database
) -> None:
    c, rt = client
    gpu = await _provider(db, "gpu", {"big": 5000.0, "small": 100.0})
    resp = await c.post(
        "/api/model-groups",
        json={
            "name": "coding",
            "description": "for code",
            "members": [
                {"model_name": "big"},
                {"model_name": "small", "provider_id": gpu.id},
                {"model_name": "ghost"},
            ],
        },
    )
    assert resp.status_code == 201
    gid = resp.json()["id"]

    body = await _listing(c, rt)
    assert body["warnings"] == []
    [g] = body["groups"]
    assert (g["id"], g["name"], g["description"]) == (gid, "coding", "for code")
    assert [(m["label"], m["available"]) for m in g["members"]] == [
        ("big", True),
        ("small@gpu", True),
        ("ghost", False),
    ]
    assert g["endpoint"] == "/v1/chat/completions"
    assert g["choices"]["latency"] == "small on gpu"

    r = await rt.route("coding", "/v1/chat/completions")
    assert r is not None
    assert r.model_key == "small"


async def test_members_can_be_added_and_removed(
    client: tuple[httpx.AsyncClient, Router], db: Database
) -> None:
    c, rt = client
    gid = (await c.post("/api/model-groups", json={"name": "vision"})).json()["id"]
    resp = await c.post(f"/api/model-groups/{gid}/members", json={"model_name": "v"})
    assert resp.status_code == 201
    mid = resp.json()["id"]
    [g] = (await _listing(c, rt))["groups"]
    assert [m["model_name"] for m in g["members"]] == ["v"]
    assert g["choices"] == {}

    assert (await c.delete(f"/api/model-groups/{gid}/members/{mid}")).status_code == 200
    assert (await c.delete(f"/api/model-groups/{gid}/members/{mid}")).status_code == 404
    assert (await c.delete(f"/api/model-groups/{gid}")).status_code == 200
    assert (await c.delete(f"/api/model-groups/{gid}")).status_code == 404
    assert (await _listing(c, rt))["groups"] == []


async def test_rename_and_describe(
    client: tuple[httpx.AsyncClient, Router], db: Database
) -> None:
    c, _rt = client
    gid = (await c.post("/api/model-groups", json={"name": "code"})).json()["id"]
    resp = await c.patch(f"/api/model-groups/{gid}", json={"name": "coding"})
    assert resp.status_code == 200
    resp = await c.patch(f"/api/model-groups/{gid}", json={"description": "d"})
    assert resp.json() == {"id": gid, "name": "coding", "description": "d"}
    resp = await c.patch(f"/api/model-groups/{gid}", json={"description": ""})
    assert resp.json()["description"] is None
    assert (await c.patch("/api/model-groups/999", json={})).status_code == 404


@pytest.mark.parametrize(
    ("body", "status", "detail"),
    [
        ({}, 400, "name is required"),
        ({"name": "x" * 256}, 400, "name is too long"),
        ({"name": "served"}, 409, "a provider serves a model named served"),
        ({"name": "nick"}, 409, "nick is a model alias"),
        ({"name": "real"}, 409, "real is a model alias"),
        ({"name": "taken"}, 409, "group taken already exists"),
        ({"name": "g", "members": [{"model_name": "taken"}]}, 400, "another model"),
        ({"name": "g", "members": [{"model_name": "g"}]}, 400, "contain itself"),
        ({"name": "g", "members": [{}]}, 400, "model_name is required"),
        ({"name": "g", "members": "x"}, 400, "members must be a list"),
        (
            {"name": "g", "members": [{"model_name": "m", "provider_id": "x"}]},
            400,
            "provider_id must be a number",
        ),
        (
            {"name": "g", "members": [{"model_name": "m", "provider_id": 999}]},
            404,
            "provider not found",
        ),
        ({"name": "g", "description": "d" * 1001}, 400, "description is too long"),
    ],
)
async def test_create_rejects_bad_input(
    client: tuple[httpx.AsyncClient, Router],
    db: Database,
    body: dict[str, Any],
    status: int,
    detail: str,
) -> None:
    c, _rt = client
    await _provider(db, "gpu", {"served": 100.0})
    await db.set_model_alias("nick", "real")
    await db.create_model_group("taken")
    resp = await c.post("/api/model-groups", json=body)
    assert resp.status_code == status
    assert detail in resp.json()["detail"]


async def test_bad_bodies_and_self_membership(
    client: tuple[httpx.AsyncClient, Router],
) -> None:
    c, _rt = client
    gid = (await c.post("/api/model-groups", json={"name": "g"})).json()["id"]
    resp = await c.post(f"/api/model-groups/{gid}/members", json={"model_name": "g"})
    assert resp.status_code == 400
    resp = await c.post(
        "/api/model-groups",
        content=b"[1]",
        headers={"content-type": "application/json"},
    )
    assert resp.status_code == 400
    resp = await c.post(
        "/api/model-groups", content=b"{", headers={"content-type": "application/json"}
    )
    assert resp.status_code == 400
    resp = await c.post("/api/model-groups/999/members", json={"model_name": "m"})
    assert resp.status_code == 404
    resp = await c.patch(f"/api/model-groups/{gid}", json={"name": "g"})
    assert resp.status_code == 200


async def test_listing_without_a_router(db: Database) -> None:
    pm = ProviderManager(db)
    dash_deps.init(db, pm)
    app = FastAPI()
    app.include_router(groups_router)
    await db.create_model_group("g")
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://dash") as c:
        body = (await c.get("/api/model-groups")).json()
    assert body["warnings"] == []
    assert body["groups"][0]["choices"] == {}
    await pm.stop()
