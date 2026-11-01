"""Ollama (/api/*) and OpenAI-style (/v1/*) proxy routes against fake backends."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

import httpx
import pytest
from fastapi import FastAPI
from pytest_httpx import HTTPXMock

from llama_router.api import deps as api_deps
from llama_router.api.app import app as ollama_app
from llama_router.background import drain
from llama_router.database import Database
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

MODEL = "m"
BASE = "http://be.test:11434"


@pytest.fixture
async def pm(db: Database) -> AsyncIterator[ProviderManager]:
    manager = ProviderManager(db)
    yield manager
    await manager.stop()


@pytest.fixture
async def provider(db: Database, pm: ProviderManager) -> Provider:
    await db.set_allow_unauthenticated(True)
    added = await db.add_provider("be", ProviderProtocol.OLLAMA)
    pid = added.id
    assert pid is not None
    await db.add_address(pid, BASE, is_preferred=True)
    await db.update_provider_status(pid, ProviderStatus.IDLE)
    await db.set_provider_models(pid, [ProviderModel(provider_id=pid, name=MODEL)])
    p = await db.get_provider(pid)
    assert p is not None
    await pm._rebuild_clients(p)
    rt = Router(db, pm)
    api_deps.init(db, pm, rt)
    v1_deps.init(db, pm, rt)
    return p


def _pid(p: Provider) -> int:
    assert p.id is not None
    return p.id


def _client(app: FastAPI) -> httpx.AsyncClient:
    transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
    return httpx.AsyncClient(transport=transport, base_url="http://router")


JSON_ROUTES = [
    # (app, path, upstream path, request body, upstream answer)
    (
        ollama_app,
        "/api/chat",
        "/api/chat",
        {"model": MODEL, "stream": False},
        {"message": {"content": "hi"}},
    ),
    (
        ollama_app,
        "/api/generate",
        "/api/generate",
        {"model": MODEL, "prompt": "p", "stream": False},
        {"response": "hi"},
    ),
    (
        ollama_app,
        "/api/embed",
        "/api/embed",
        {"model": MODEL, "input": "x"},
        {"embeddings": [[0.1]]},
    ),
    (
        ollama_app,
        "/api/embeddings",
        "/api/embeddings",
        {"model": MODEL, "prompt": "x"},
        {"embedding": [0.1]},
    ),
    (
        v1_app,
        "/v1/chat/completions",
        "/v1/chat/completions",
        {"model": MODEL, "messages": [], "stream": False},
        {"choices": []},
    ),
    (
        v1_app,
        "/v1/completions",
        "/v1/completions",
        {"model": MODEL, "prompt": "p", "stream": False},
        {"choices": []},
    ),
    (
        v1_app,
        "/v1/embeddings",
        "/v1/embeddings",
        {"model": MODEL, "input": "x"},
        {"data": []},
    ),
    (
        v1_app,
        "/v1/responses",
        "/v1/responses",
        {"model": MODEL, "input": "x", "stream": False},
        {"output": []},
    ),
    (
        v1_app,
        "/v1/images/generations",
        "/v1/images/generations",
        {"model": MODEL, "prompt": "cat"},
        {"data": [{"url": "u"}]},
    ),
]
IDS = [r[1] for r in JSON_ROUTES]


@pytest.mark.parametrize(("app", "path", "up", "body", "answer"), JSON_ROUTES, ids=IDS)
async def test_json_route_proxies_and_logs(
    provider: Provider,
    db: Database,
    httpx_mock: HTTPXMock,
    app: FastAPI,
    path: str,
    up: str,
    body: dict[str, Any],
    answer: dict[str, Any],
) -> None:
    httpx_mock.add_response(url=BASE + up, json=answer)
    async with _client(app) as c:
        resp = await c.post(path, json=body)
    assert resp.status_code == 200
    assert resp.json() == answer
    await drain()
    assert any(
        e.endpoint == path and e.status == "ok"
        for e in await db.get_request_logs(20, 0)
    )


@pytest.mark.parametrize(("app", "path", "up", "body", "answer"), JSON_ROUTES, ids=IDS)
async def test_json_route_forwards_upstream_error(
    provider: Provider,
    db: Database,
    httpx_mock: HTTPXMock,
    app: FastAPI,
    path: str,
    up: str,
    body: dict[str, Any],
    answer: dict[str, Any],
) -> None:
    httpx_mock.add_response(url=BASE + up, status_code=429, json={"error": "slow down"})
    async with _client(app) as c:
        resp = await c.post(path, json=body)
    assert resp.status_code == 429
    assert resp.json() == {"error": "slow down"}
    await drain()
    (entry,) = [e for e in await db.get_request_logs(20, 0) if e.endpoint == path]
    assert entry.status == "error"
    assert "HTTP 429" in (entry.error_detail or "")


@pytest.mark.parametrize(("app", "path", "up", "body", "answer"), JSON_ROUTES, ids=IDS)
async def test_json_route_connection_failure_is_logged_and_released(
    provider: Provider,
    pm: ProviderManager,
    db: Database,
    httpx_mock: HTTPXMock,
    app: FastAPI,
    path: str,
    up: str,
    body: dict[str, Any],
    answer: dict[str, Any],
) -> None:
    httpx_mock.add_exception(httpx.ConnectError("refused"))
    async with _client(app) as c:
        resp = await c.post(path, json=body)
    assert resp.status_code == 500
    assert pm.active_requests(_pid(provider)) == 0
    await drain()
    (entry,) = [e for e in await db.get_request_logs(20, 0) if e.endpoint == path]
    assert entry.status == "error"
    assert "ConnectError" in (entry.error_detail or "")


@pytest.mark.parametrize(("app", "path", "up", "body", "answer"), JSON_ROUTES, ids=IDS)
async def test_json_route_validates_model(
    provider: Provider,
    app: FastAPI,
    path: str,
    up: str,
    body: dict[str, Any],
    answer: dict[str, Any],
) -> None:
    async with _client(app) as c:
        missing = await c.post(
            path, json={k: v for k, v in body.items() if k != "model"}
        )
        unknown = await c.post(path, json={**body, "model": "nope"})
    assert missing.status_code == 400
    assert unknown.status_code == 404
    assert "nope" in unknown.json()["detail"]


@pytest.mark.parametrize(
    "path",
    [*IDS, "/v1/audio/speech", "/v1/systemone"],
)
async def test_json_route_rejects_bad_bodies(provider: Provider, path: str) -> None:
    app = ollama_app if path.startswith("/api/") else v1_app
    async with _client(app) as c:
        not_json = await c.post(
            path, content=b"{nope", headers={"content-type": "application/json"}
        )
        not_object = await c.post(path, json=["model", MODEL])
        bad_model = await c.post(path, json={"model": ["m"]})
    for resp in (not_json, not_object, bad_model):
        assert resp.status_code == 400, resp.text
    assert "string" in bad_model.json()["detail"]


async def test_v1_passthrough_strips_router_only_fields(
    provider: Provider, httpx_mock: HTTPXMock
) -> None:
    httpx_mock.add_response(url=BASE + "/v1/chat/completions", json={"choices": []})
    body = {
        "model": MODEL,
        "messages": [],
        "stream": False,
        "temperature": 0.5,
        "provider": {"order": ["x"]},
        "models": ["a", "b"],
        "route": "fallback",
        "transforms": ["middle-out"],
        "extra_body": {"x": 1},
        "base_url": "http://evil.test",
        "api_key": "sk-other",
    }
    async with _client(v1_app) as c:
        resp = await c.post("/v1/chat/completions", json=body)
    assert resp.status_code == 200
    sent = json.loads(httpx_mock.get_requests()[0].content)
    assert sent == {"model": MODEL, "messages": [], "stream": False, "temperature": 0.5}


STREAM_ROUTES = [
    (ollama_app, "/api/chat", "/api/chat", {"model": MODEL, "messages": []}),
    (ollama_app, "/api/generate", "/api/generate", {"model": MODEL, "prompt": "p"}),
    (
        v1_app,
        "/v1/chat/completions",
        "/v1/chat/completions",
        {"model": MODEL, "messages": [], "stream": True},
    ),
    (
        v1_app,
        "/v1/completions",
        "/v1/completions",
        {"model": MODEL, "prompt": "p", "stream": True},
    ),
    (
        v1_app,
        "/v1/responses",
        "/v1/responses",
        {"model": MODEL, "input": "x", "stream": True},
    ),
]


@pytest.mark.parametrize(
    ("app", "path", "up", "body"), STREAM_ROUTES, ids=[r[1] for r in STREAM_ROUTES]
)
async def test_stream_route_relays_chunks(
    provider: Provider,
    pm: ProviderManager,
    db: Database,
    httpx_mock: HTTPXMock,
    app: FastAPI,
    path: str,
    up: str,
    body: dict[str, Any],
) -> None:
    httpx_mock.add_response(url=BASE + up, content=b"chunk-1\nchunk-2\n")
    async with _client(app) as c:
        resp = await c.post(path, json=body)
    assert resp.status_code == 200
    assert resp.content == b"chunk-1\nchunk-2\n"
    assert pm.active_requests(_pid(provider)) == 0
    await drain()
    assert any(
        e.endpoint == path and e.status == "ok"
        for e in await db.get_request_logs(20, 0)
    )


async def test_audio_speech_streams_with_format_media_type(
    provider: Provider, httpx_mock: HTTPXMock
) -> None:
    httpx_mock.add_response(url=BASE + "/v1/audio/speech", content=b"RIFF")
    async with _client(v1_app) as c:
        resp = await c.post(
            "/v1/audio/speech",
            json={"model": MODEL, "input": "hi", "response_format": "wav"},
        )
    assert resp.headers["content-type"] == "audio/wav"
    assert resp.content == b"RIFF"


async def test_audio_speech_requires_known_model(provider: Provider) -> None:
    async with _client(v1_app) as c:
        assert (await c.post("/v1/audio/speech", json={})).status_code == 400
        assert (
            await c.post("/v1/audio/speech", json={"model": "x"})
        ).status_code == 404


async def test_audio_transcriptions_proxy(
    provider: Provider, db: Database, httpx_mock: HTTPXMock
) -> None:
    httpx_mock.add_response(
        url=BASE + "/v1/audio/transcriptions", json={"text": "hello"}
    )
    async with _client(v1_app) as c:
        resp = await c.post(
            "/v1/audio/transcriptions",
            data={"model": MODEL},
            files={"file": ("a.wav", b"\x00\x01", "audio/wav")},
        )
    assert resp.status_code == 200
    assert resp.json() == {"text": "hello"}
    sent = httpx_mock.get_requests()[0]
    assert sent.headers["content-type"].startswith("multipart/form-data")
    await drain()
    assert any(
        e.endpoint == "/v1/audio/transcriptions"
        for e in await db.get_request_logs(20, 0)
    )


async def test_audio_transcriptions_errors(
    provider: Provider, httpx_mock: HTTPXMock
) -> None:
    async with _client(v1_app) as c:
        no_model = await c.post(
            "/v1/audio/transcriptions", files={"file": ("a.wav", b"x", "audio/wav")}
        )
        unknown = await c.post("/v1/audio/transcriptions", data={"model": "zz"})
        httpx_mock.add_response(status_code=500, json={"error": "bad"})
        upstream = await c.post("/v1/audio/transcriptions", data={"model": MODEL})
        httpx_mock.add_exception(httpx.ConnectError("x"))
        down = await c.post("/v1/audio/transcriptions", data={"model": MODEL})
    assert no_model.status_code == 400
    assert unknown.status_code == 404
    assert upstream.status_code == 500
    assert upstream.json() == {"error": "upstream service returned an error"}
    assert down.status_code == 500


async def test_audio_voices_aggregates_and_skips_failures(
    provider: Provider, httpx_mock: HTTPXMock
) -> None:
    httpx_mock.add_response(
        url=BASE + "/v1/audio/voices",
        json={"voices": [{"voice_id": "a"}, {"id": "b"}, {"voice_id": "a"}, {}]},
    )
    async with _client(v1_app) as c:
        resp = await c.get("/v1/audio/voices")
    assert [v.get("voice_id") or v.get("id") for v in resp.json()["voices"]] == [
        "a",
        "b",
    ]


async def test_audio_voices_ignores_failing_provider(
    provider: Provider, httpx_mock: HTTPXMock
) -> None:
    httpx_mock.add_response(status_code=500)
    async with _client(v1_app) as c:
        resp = await c.get("/v1/audio/voices")
    assert resp.json() == {"voices": []}


async def test_images_edits_proxy(provider: Provider, httpx_mock: HTTPXMock) -> None:
    httpx_mock.add_response(url=BASE + "/v1/images/edits", json={"data": []})
    async with _client(v1_app) as c:
        resp = await c.post(
            "/v1/images/edits",
            data={"model": MODEL, "prompt": "p"},
            files={"image": ("i.png", b"\x89PNG", "image/png")},
        )
    assert resp.status_code == 200, resp.text
    assert resp.json() == {"data": []}


async def test_images_edits_requires_known_model(provider: Provider) -> None:
    async with _client(v1_app) as c:
        assert (
            await c.post(
                "/v1/images/edits", files={"image": ("i.png", b"x", "image/png")}
            )
        ).status_code == 400
        assert (
            await c.post("/v1/images/edits", data={"model": "zz"})
        ).status_code == 404


async def test_ollama_misc_routes(provider: Provider, pm: ProviderManager) -> None:
    async with _client(ollama_app) as c:
        tags = (await c.get("/api/tags")).json()
        version = (await c.get("/api/version")).json()
        ps = (await c.get("/api/ps")).json()
    assert [m["name"] for m in tags["models"]] == [MODEL]
    assert "version" in version
    assert ps == {"models": []}
    pm.acquire(_pid(provider))
    try:
        async with _client(ollama_app) as c:
            running = (await c.get("/api/ps")).json()
    finally:
        pm.release(_pid(provider))
    assert [m["name"] for m in running["models"]] == [MODEL]


async def test_v1_models_and_health(provider: Provider) -> None:
    async with _client(v1_app) as c:
        listing = (await c.get("/v1/models")).json()
        one = await c.get(f"/v1/models/{MODEL}")
        missing = await c.get("/v1/models/ghost")
        health = (await c.get("/health")).json()
    assert [m["id"] for m in listing["data"]] == [MODEL]
    assert one.json()["id"] == MODEL
    assert missing.status_code == 404
    assert health["status"] == "ok"
