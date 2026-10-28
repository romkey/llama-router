"""TypeSafe's hosted Jev as a provider: a preset that serves decisions only."""

from __future__ import annotations

import json

import httpx
import pytest

from llama_router.auto_benchmark import AutoBenchmarker
from llama_router.cloud_presets import PRESETS, get_preset
from llama_router.database import Database
from llama_router.llamacpp_api import deps
from llama_router.llamacpp_api.app import app as api_app
from llama_router.models import (
    ENDPOINT_MISSING,
    ENDPOINT_SUPPORTED,
    ProviderModel,
    ProviderStatus,
)
from llama_router.probe import probe_url
from llama_router.provider_manager import ProviderManager
from llama_router.router import Router, RoutingPreferences
from llama_router.routing_table import decides_natively, native_decision_path

KEY = "ts-test-key"
JEV = "https://api.typesafe.ai/v1"
QUESTION = {"is_urgent": {"type": "noul", "instructions": "The message is urgent"}}
ANSWER = {
    "model": "jev-1.14.0",
    "answers": {"is_urgent": {"type": "noul", "noul": 0.97}},
    "usage": {"input_tokens": 12},
}


@pytest.fixture
async def pm(db: Database, monkeypatch):
    for p in PRESETS:
        monkeypatch.delenv(p.env_var, raising=False)
    monkeypatch.setenv("LLAMA_ROUTER_PROVIDER_TYPESAFE_API_KEY", KEY)
    manager = ProviderManager(db)
    yield manager
    await manager.stop()


async def _jev(db: Database, pm: ProviderManager, *models: str):
    (made,) = await pm.provision_from_env()
    await db.update_provider_status(made.id, ProviderStatus.IDLE)
    await db.set_provider_models(
        made.id,
        [ProviderModel(provider_id=made.id, name=m) for m in models or ("jev-latest",)],
    )
    await pm._rebuild_clients(await db.get_provider(made.id))
    return await db.get_provider(made.id)


@pytest.fixture
async def router(db: Database, pm: ProviderManager):
    await db.set_allow_unauthenticated(True)
    rt = Router(db, pm)
    deps.init(db, pm, rt)
    return rt


async def _post(path: str, body: dict) -> httpx.Response:
    transport = httpx.ASGITransport(app=api_app)
    async with httpx.AsyncClient(transport=transport, base_url="http://router") as c:
        return await c.post(path, json=body)


async def test_preset_makes_a_decision_only_cloud_provider(db, pm) -> None:
    preset = get_preset("typesafe")
    assert preset.base_url == JEV
    assert preset.api_key_header is None
    assert preset.env_var == "LLAMA_ROUTER_PROVIDER_TYPESAFE_API_KEY"
    p = await _jev(db, pm)
    assert (p.preset, p.locality, p.manual_models) == (
        "typesafe",
        "cloud",
        ["jev-latest"],
    )
    (addr,) = await db.get_addresses(p.id)
    assert addr.url == JEV
    assert not p.auto_benchmarks
    # Declared routes, with no probe needed.
    assert p.capabilities is None
    for path in ("/v1/systemone", "/systemone", "/v1/models"):
        assert p.endpoint_state(path) == ENDPOINT_SUPPORTED, path
    for path in ("/v1/chat/completions", "/v1/embeddings", "/v1/decisions", "/props"):
        assert p.endpoint_state(path) == ENDPOINT_MISSING, path
    assert native_decision_path(p, "/v1/systemone") == "/v1/systemone"
    assert native_decision_path(p, "/v1/decisions") == "/v1/systemone"


async def test_discovery_reads_typesafes_model_list(db, pm, httpx_mock) -> None:
    httpx_mock.add_response(
        url=f"{JEV}/models",
        json={
            "models": [
                {"name": "jev-latest", "description": "x", "release_date": "d"},
                {"name": "jev-preview", "description": "y", "release_date": "d"},
            ]
        },
    )
    p = await _jev(db, pm)
    await pm._discover_provider(p)
    names = {m.name for m in await db.get_provider_models(p.id)}
    assert names == {"jev-latest", "jev-preview"}
    (sent,) = httpx_mock.get_requests()
    assert sent.headers["authorization"] == f"Bearer {KEY}"


@pytest.mark.parametrize("path", ["/v1/systemone", "/v1/decisions", "/decisions"])
async def test_decisions_pass_through_with_key_and_backend_name(
    db, pm, router, httpx_mock, path
) -> None:
    await _jev(db, pm)
    # The client asks by another name; Jev must get its own.
    await db.set_model_fallback("decider", "jev-latest")
    httpx_mock.add_response(url=f"{JEV}/systemone", method="POST", json=ANSWER)
    body = {"model": "decider", "state": "Prod is down.", "questions": QUESTION}
    resp = await _post(path, body)
    assert resp.status_code == 200, resp.text
    assert resp.json() == ANSWER
    (sent,) = httpx_mock.get_requests()
    assert sent.headers["authorization"] == f"Bearer {KEY}"
    assert json.loads(sent.content) == {**body, "model": "jev-latest"}


async def test_never_offered_for_anything_but_decisions(db, pm, router) -> None:
    p = await _jev(db, pm)
    for path in (
        "/v1/chat/completions",
        "/v1/completions",
        "/v1/responses",
        "/v1/embeddings",
        "/api/chat",
        "/api/generate",
        "/api/embed",
    ):
        assert await router.route("jev-latest", path) is None, path
    chaos = RoutingPreferences(mode="chaos")
    assert await router.route("jev-latest", "/v1/chat/completions", chaos) is None
    for path in ("/v1/systemone", "/v1/decisions"):
        chosen = await router.route("jev-latest", path)
        assert chosen is not None
        assert chosen.provider.id == p.id


async def test_chat_request_for_a_jev_model_is_a_404(db, pm, router) -> None:
    await _jev(db, pm)
    resp = await _post(
        "/v1/chat/completions",
        {"model": "jev-latest", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert resp.status_code == 404


async def test_decision_check_skips_it(db, pm, httpx_mock) -> None:
    p = await _jev(db, pm)
    assert decides_natively(p, "/v1/systemone")
    assert await AutoBenchmarker(db, pm).check_decisions(p.id, "jev-latest") == (
        "skipped"
    )
    assert httpx_mock.get_requests() == []


async def test_probe_asks_only_the_declared_routes(httpx_mock) -> None:
    httpx_mock.add_callback(
        lambda r: httpx.Response(401), is_reusable=True, is_optional=True
    )
    result = await probe_url(JEV)
    assert result.preset == "typesafe"
    probed = {r.url.path for r in httpx_mock.get_requests()}
    assert probed
    assert probed <= {
        "/v1/models",
        "/v1/systemone",
        "/v1/v1/models",
        "/v1/v1/systemone",
        "/models",
        "/systemone",
    }
    assert not any("chat" in p or "embed" in p for p in probed)
