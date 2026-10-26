"""Provider endpoint probing."""

from __future__ import annotations

import json

import httpx
import pytest

from llama_router.probe import (
    CATALOG,
    normalize_base_url,
    probe_url,
)
from tests.test_dashboard_auth import dash_client  # noqa: F401  (fixture)


def _server(routes: dict[tuple[str, str], httpx.Response], default=None):
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        key = (request.method, request.url.path)
        if key in routes:
            return routes[key]
        return default or httpx.Response(404, text="404 page not found")

    return httpx.MockTransport(handler), seen


def _state(result, method: str, path: str) -> str:
    return next(
        e.state for e in result.endpoints if e.method == method and e.path == path
    )


OLLAMA = {
    ("GET", "/api/version"): httpx.Response(200, json={"version": "0.12.6"}),
    ("GET", "/api/tags"): httpx.Response(
        200, json={"models": [{"name": "qwen3:8b"}, {"name": "llama3.2:latest"}]}
    ),
    ("GET", "/api/ps"): httpx.Response(200, json={"models": []}),
    ("POST", "/api/chat"): httpx.Response(400, json={"error": "model is required"}),
    ("GET", "/v1/models"): httpx.Response(
        200, json={"object": "list", "data": [{"id": "qwen3:8b"}]}
    ),
    ("POST", "/v1/chat/completions"): httpx.Response(400, json={"error": "bad"}),
}

LLAMACPP = {
    ("GET", "/health"): httpx.Response(200, json={"status": "ok"}),
    ("GET", "/props"): httpx.Response(
        200,
        json={"default_generation_settings": {}, "build_info": "b6000-abc"},
    ),
    ("GET", "/v1/models"): httpx.Response(
        200, json={"object": "list", "data": [{"id": "gemma-3-4b"}]}
    ),
    ("POST", "/completion"): httpx.Response(400, json={"error": "prompt missing"}),
    ("POST", "/rerank"): httpx.Response(400, json={"error": "query missing"}),
    ("POST", "/v1/messages"): httpx.Response(400, json={"error": "bad"}),
}


async def test_detects_ollama_and_lists_models() -> None:
    transport, seen = _server(OLLAMA)
    r = await probe_url("http://gpu:11434/", transport=transport)
    assert r.base_url == "http://gpu:11434"
    assert r.reachable and not r.auth_required
    assert "ollama" in r.servers and "openai-compatible" in r.servers
    assert r.version == "0.12.6"
    assert r.suggested_type == "ollama"
    assert r.models == ["qwen3:8b", "llama3.2:latest"]
    assert _state(r, "GET", "/api/tags") == "available"
    assert _state(r, "POST", "/api/chat") == "exists"
    assert _state(r, "POST", "/v1/decisions") == "missing"
    # Every catalog entry was tried, and POSTs only ever carried "{}".
    assert len(seen) == len(CATALOG)
    for req in seen:
        if req.method == "POST":
            assert json.loads(req.content) == {}


async def test_detects_llamacpp() -> None:
    transport, _ = _server(LLAMACPP)
    r = await probe_url("http://box:8080", transport=transport)
    assert "llama.cpp" in r.servers
    assert r.suggested_type == "llamacpp"
    assert r.version == "b6000-abc"
    assert r.models == ["gemma-3-4b"]
    assert _state(r, "POST", "/rerank") == "exists"
    assert _state(r, "POST", "/v1/messages") == "exists"
    d = r.to_dict()
    assert "POST /completion" in d["supported"]
    assert d["families"]["llamacpp"] >= 3


async def test_litellm_decisions_and_auth() -> None:
    routes = {
        ("GET", "/health/liveliness"): httpx.Response(200, text='"I\'m alive!"'),
        ("GET", "/v1/models"): httpx.Response(401, json={"error": "auth"}),
        ("POST", "/v1/decisions"): httpx.Response(401, json={"error": "auth"}),
    }
    transport, _ = _server(routes)
    r = await probe_url("https://llm.example.org", transport=transport)
    assert "litellm" in r.servers
    assert r.auth_required
    assert _state(r, "POST", "/v1/decisions") == "needs_key"

    def keyed(request: httpx.Request) -> httpx.Response:
        if request.headers.get("Authorization") != "Bearer sk-1":
            return httpx.Response(401)
        if request.url.path == "/v1/decisions":
            return httpx.Response(422, json={"error": "input required"})
        if request.url.path == "/v1/models":
            return httpx.Response(200, json={"data": [{"id": "decider"}]})
        return httpx.Response(404)

    r2 = await probe_url(
        "https://llm.example.org", api_key="sk-1", transport=httpx.MockTransport(keyed)
    )
    assert _state(r2, "POST", "/v1/decisions") == "exists"
    assert r2.models == ["decider"]


async def test_unexpected_endpoint_200_counts_as_missing() -> None:
    lmstudio_default = httpx.Response(
        200, json={"error": "Unexpected endpoint or method. (GET /api/tags)"}
    )
    routes = {
        ("GET", "/v1/models"): httpx.Response(200, json={"data": [{"id": "m"}]}),
    }
    transport, _ = _server(routes, default=lmstudio_default)
    r = await probe_url("http://studio:1234", transport=transport)
    assert _state(r, "GET", "/api/tags") == "missing"
    assert _state(r, "GET", "/v1/models") == "available"
    assert r.servers == ["openai-compatible"]


async def test_unreachable() -> None:
    def boom(request):
        raise httpx.ConnectError("refused")

    r = await probe_url("http://nothing:1", transport=httpx.MockTransport(boom))
    assert not r.reachable and r.error
    assert all(e.state == "error" for e in r.endpoints)


@pytest.mark.parametrize(
    "bad",
    ["ftp://x", "http://", "http://u:p@host", "http://host/?q=1", "nonsense"],
)
def test_normalize_rejects(bad: str) -> None:
    with pytest.raises(ValueError):
        normalize_base_url(bad)


def _mock_ollama_everywhere(httpx_mock) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        resp = OLLAMA.get((request.method, request.url.path))
        if resp is not None:
            return httpx.Response(resp.status_code, content=resp.content)
        return httpx.Response(404)

    httpx_mock.add_callback(handler, is_reusable=True, is_optional=True)


@pytest.mark.httpx_mock(assert_all_responses_were_requested=False)
async def test_probe_routes_store_capabilities(
    dash_client, httpx_mock  # noqa: F811
) -> None:
    _mock_ollama_everywhere(httpx_mock)
    client, db = dash_client

    r = await client.post("/api/providers/probe", json={"url": "http://gpu:11434"})
    assert r.status_code == 200
    body = r.json()
    assert body["suggested_type"] == "ollama"
    assert body["family_labels"]["decisions"] == "Decision models"

    bad = await client.post("/api/providers/probe", json={"url": "file:///etc"})
    assert bad.status_code == 400

    await client.post("/providers/add", data={"name": "gpu", "url": "http://gpu:11434"})
    p = (await db.list_providers())[0]
    r = await client.post(f"/api/providers/{p.id}/probe")
    assert r.status_code == 200
    stored = (await db.get_provider(p.id)).capabilities
    assert stored and "GET /api/tags" in stored["supported"]
    assert (await db.get_provider(p.id)).probed_at is not None

    page = await client.get(f"/providers/{p.id}")
    assert "supported endpoints" in page.text and "ollama" in page.text

    missing = await client.post("/api/providers/999/probe")
    assert missing.status_code == 404
