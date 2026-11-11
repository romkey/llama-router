"""Reranking: /v1/rerank and its aliases, routed to the provider's rerank route."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

import httpx
import pytest
from pytest_httpx import HTTPXMock

from llama_router.background import drain
from llama_router.database import Database
from llama_router.endpoints import (
    RERANK,
    RERANK_PATH,
    endpoint_class,
    normalize_endpoint,
    resolve_endpoint,
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
from llama_router.router import Router
from llama_router.routing_table import (
    TIER_MISSING,
    rerank_support,
    rerank_upstream_path,
)
from llama_router.worker_link import worker as worker_mod

MODEL = "bge-reranker"
BASE = "http://rr.test:8080"
BODY = {"model": MODEL, "query": "q", "documents": ["a", "b"], "top_n": 1}
ANSWER = {
    "model": MODEL,
    "results": [{"index": 1, "relevance_score": 0.9}],
    "usage": {"prompt_tokens": 7, "total_tokens": 7},
}
ALIASES = ["/v1/rerank", "/rerank", "/v1/reranking", "/reranking", "/v2/rerank"]


def _probed(*paths: str, state: str = "available") -> dict[str, Any]:
    return {"endpoints": [{"path": p, "state": state} for p in paths]}


async def _add(db: Database, name: str, caps: dict[str, Any] | None = None) -> int:
    p = await db.add_provider(name, ProviderProtocol.OPENAI)
    pid = p.id
    assert pid is not None
    await db.add_address(pid, BASE if name == "rr" else f"http://{name}.test:8080")
    await db.update_provider_status(pid, ProviderStatus.IDLE)
    await db.set_provider_models(pid, [ProviderModel(provider_id=pid, name=MODEL)])
    if caps is not None:
        await db.set_provider_capabilities(pid, caps)
    return pid


@pytest.fixture
async def pm(db: Database) -> AsyncIterator[ProviderManager]:
    manager = ProviderManager(db)
    yield manager
    await manager.stop()


async def _serve(
    db: Database, pm: ProviderManager, caps: dict[str, Any] | None
) -> None:
    await db.set_allow_unauthenticated(True)
    pid = await _add(db, "rr", caps)
    p = await db.get_provider(pid)
    assert p is not None
    await pm._rebuild_clients(p)
    v1_deps.init(db, pm, Router(db, pm))


def _client() -> httpx.AsyncClient:
    transport = httpx.ASGITransport(app=v1_app, raise_app_exceptions=False)
    return httpx.AsyncClient(transport=transport, base_url="http://router")


# --- endpoint table ---


@pytest.mark.parametrize("path", ALIASES)
def test_rerank_aliases_resolve_to_one_route(path: str) -> None:
    assert normalize_endpoint(path) == RERANK_PATH
    assert resolve_endpoint(path) == RERANK_PATH
    assert endpoint_class(path) == RERANK


def test_worker_runs_rerank_jobs() -> None:
    for path in ALIASES:
        assert ("POST", path) in worker_mod._INFERENCE_PATHS


# --- upstream route choice ---


def _provider(caps: dict[str, Any] | None) -> Provider:
    return Provider(id=1, name="p", protocol=ProviderProtocol.OPENAI, capabilities=caps)


@pytest.mark.parametrize(
    ("caps", "upstream", "support"),
    [
        (None, None, "unknown"),
        (_probed("/v1/rerank"), "/v1/rerank", "supported"),
        # A bare route counts for its /v1 form.
        (_probed("/rerank"), "/v1/rerank", "supported"),
        (_probed("/reranking"), "/v1/reranking", "supported"),
        (_probed("/v2/rerank"), "/v2/rerank", "supported"),
        (
            {
                "endpoints": [
                    {"path": "/v1/rerank", "state": "missing"},
                    {"path": "/v2/rerank", "state": "available"},
                ]
            },
            "/v2/rerank",
            "supported",
        ),
        (_probed("/v1/rerank", "/rerank", state="missing"), None, "missing"),
    ],
)
def test_rerank_upstream_follows_probe(
    caps: dict[str, Any] | None, upstream: str | None, support: str
) -> None:
    p = _provider(caps)
    assert rerank_upstream_path(p) == upstream
    assert rerank_support(p) == support


async def test_provider_without_rerank_is_ranked_missing(
    db: Database, pm: ProviderManager
) -> None:
    good = await _add(db, "good", _probed("/v1/rerank"))
    await _add(db, "none", _probed("/v1/rerank", "/v1/reranking", state="missing"))
    rt = Router(db, pm)
    table = await rt.rebuild()
    tiers = {
        c.provider.name: c.tier for c in table.all_candidates[(MODEL, RERANK_PATH)]
    }
    assert tiers["none"] == TIER_MISSING
    for _ in range(3):
        result = await rt.route(MODEL, "/rerank")
        assert result is not None
        assert result.provider.id == good


# --- the API ---


@pytest.mark.parametrize("path", ALIASES)
async def test_rerank_proxies_and_logs(
    db: Database, pm: ProviderManager, httpx_mock: HTTPXMock, path: str
) -> None:
    await _serve(db, pm, None)
    httpx_mock.add_response(url=BASE + "/v1/rerank", json=ANSWER)
    async with _client() as c:
        resp = await c.post(path, json=BODY)
    assert resp.status_code == 200, resp.text
    assert resp.json() == ANSWER
    sent = httpx_mock.get_requests()[-1]
    assert json.loads(sent.read()) == BODY
    await drain()
    assert any(
        e.endpoint == path and e.status == "ok"
        for e in await db.get_request_logs(20, 0)
    )


async def test_rerank_goes_to_the_route_the_probe_found(
    db: Database, pm: ProviderManager, httpx_mock: HTTPXMock
) -> None:
    await _serve(db, pm, _probed("/v2/rerank"))
    httpx_mock.add_response(url=BASE + "/v2/rerank", json=ANSWER)
    async with _client() as c:
        resp = await c.post("/v1/rerank", json=BODY)
    assert resp.status_code == 200, resp.text


async def test_rerank_forwards_upstream_error(
    db: Database, pm: ProviderManager, httpx_mock: HTTPXMock
) -> None:
    await _serve(db, pm, None)
    httpx_mock.add_response(
        url=BASE + "/v1/rerank", status_code=400, json={"error": "not a reranker"}
    )
    async with _client() as c:
        resp = await c.post("/v1/rerank", json=BODY)
    assert resp.status_code == 400
    assert resp.json() == {"error": "not a reranker"}


@pytest.mark.parametrize(
    "body",
    [
        {"model": MODEL, "documents": ["a"]},
        {"model": MODEL, "query": "", "documents": ["a"]},
        {"model": MODEL, "query": "q"},
        {"model": MODEL, "query": "q", "documents": []},
        {"model": MODEL, "query": "q", "documents": "a"},
    ],
)
async def test_rerank_refuses_bad_bodies(
    db: Database, pm: ProviderManager, body: dict[str, Any]
) -> None:
    await _serve(db, pm, None)
    async with _client() as c:
        resp = await c.post("/v1/rerank", json=body)
    assert resp.status_code == 400, resp.text


async def test_rerank_needs_a_known_model(db: Database, pm: ProviderManager) -> None:
    await _serve(db, pm, None)
    async with _client() as c:
        missing = await c.post("/v1/rerank", json={"query": "q", "documents": ["a"]})
        unknown = await c.post("/v1/rerank", json={**BODY, "model": "nope"})
    assert missing.status_code == 400
    assert unknown.status_code == 404
