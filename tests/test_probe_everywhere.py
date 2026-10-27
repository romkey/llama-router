"""Probing drives provider setup: API bases, liveness, flavors, cloud hosts."""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone

import httpx
import pytest

from llama_router.config import settings
from llama_router.llamacpp_client import LlamaCppClient, OpenAIClient
from llama_router.models import Provider, ProviderProtocol, ProviderStatus
from llama_router.probe import probe_url
from llama_router.provider_manager import ProviderManager
from llama_router.provider_secrets import auth_headers, normalize_header
from llama_router.upstream_url import split_api_base

pytestmark = pytest.mark.httpx_mock(
    assert_all_responses_were_requested=False,
    can_send_already_matched_responses=True,
)


@pytest.mark.parametrize(
    "url,root,api",
    [
        ("http://gpu:8080", "http://gpu:8080", "/v1"),
        ("http://gpu:8080/", "http://gpu:8080", "/v1"),
        ("https://api.openai.com/v1", "https://api.openai.com", "/v1"),
        ("https://openrouter.ai/api/v1", "https://openrouter.ai", "/api/v1"),
        (
            "https://generativelanguage.googleapis.com/v1beta/openai",
            "https://generativelanguage.googleapis.com",
            "/v1beta/openai",
        ),
        ("https://api.groq.com/openai/v1", "https://api.groq.com", "/openai/v1"),
        (
            "https://api.deepinfra.com/v1/openai",
            "https://api.deepinfra.com",
            "/v1/openai",
        ),
        ("http://host/llama", "http://host/llama", "/llama/v1"),
        ("http://host/llama/v1", "http://host/llama", "/llama/v1"),
    ],
)
def test_split_api_base(url: str, root: str, api: str) -> None:
    assert split_api_base(url) == (root, api)


async def test_openai_client_routes_land_on_the_api_base() -> None:
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(str(request.url))
        return httpx.Response(200, json={"data": [{"id": "m"}], "status": "ok"})

    assert LlamaCppClient is OpenAIClient
    c = OpenAIClient(
        "https://openrouter.ai/api/v1", transport=httpx.MockTransport(handler)
    )
    await c.get_models()
    await c.chat_completions({"model": "m"})
    await c.get_props()
    await c.close()
    assert seen == [
        "https://openrouter.ai/api/v1/models",
        "https://openrouter.ai/api/v1/chat/completions",
        "https://openrouter.ai/props",
    ]


async def test_openai_client_reachable_without_health_route() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/models":
            return httpx.Response(200, json={"data": []})
        return httpx.Response(404)

    c = OpenAIClient(
        "https://api.openai.com/v1", transport=httpx.MockTransport(handler)
    )
    assert await c.is_reachable()
    await c.close()


def test_several_key_headers() -> None:
    assert normalize_header("Authorization ,x-api-key") == "Authorization, x-api-key"
    assert auth_headers("k", "Authorization, x-api-key") == {
        "Authorization": "Bearer k",
        "x-api-key": "k",
    }
    with pytest.raises(ValueError):
        normalize_header("Authorization, bad header")


def test_endpoint_state_counts_bare_openai_routes() -> None:
    p = Provider(
        name="cloud",
        protocol=ProviderProtocol.OPENAI,
        capabilities={
            "endpoints": [
                {"path": "/v1/chat/completions", "state": "missing"},
                {"path": "/chat/completions", "state": "exists"},
                {"path": "/props", "state": "missing"},
            ],
            "liveness": "/models",
        },
    )
    assert p.endpoint_state("/v1/chat/completions") == "supported"
    assert p.lacks("/props") and not p.lacks("/api/pull")
    assert p.liveness_path == "/models"


def _mock(routes: dict[tuple[str, str], httpx.Response], default=None):
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return routes.get(
            (request.method, request.url.path), default or httpx.Response(404)
        )

    return httpx.MockTransport(handler), seen


async def test_cloud_host_gets_preset_and_short_catalog() -> None:
    transport, seen = _mock(
        {("GET", "/v1/models"): httpx.Response(200, json={"data": [{"id": "gpt-x"}]})}
    )
    r = await probe_url("https://api.openai.com/v1", transport=transport)
    assert (r.preset, r.flavor, r.suggested_type) == ("openai", "openai", "openai")
    # Only OpenAI- and Anthropic-style routes were tried, not self-hosted ones.
    assert not {e.family for e in r.endpoints} & {"ollama", "vllm", "litellm", "sglang"}
    assert not any(req.url.path.startswith("/api/") for req in seen)
    assert r.liveness == "/models"
    assert r.models == ["gpt-x"]


async def test_key_rejected_everywhere() -> None:
    transport, _ = _mock({}, default=httpx.Response(401, json={"error": "bad key"}))
    r = await probe_url("http://gpu:8000", api_key="nope", transport=transport)
    assert r.key_rejected and r.auth_required
    r = await probe_url("http://gpu:8000", transport=transport)
    assert not r.key_rejected


@pytest.mark.parametrize(
    "routes,flavor",
    [
        (
            {
                ("GET", "/api/v1/health"): httpx.Response(
                    200, json={"status": "ok", "model_loaded": None}
                ),
                ("GET", "/v1/models"): httpx.Response(200, json={"data": []}),
            },
            "lemonade",
        ),
        (
            {
                ("GET", "/get_server_info"): httpx.Response(
                    200, json={"version": "0.5"}
                ),
                ("GET", "/v1/models"): httpx.Response(200, json={"data": []}),
            },
            "sglang",
        ),
        (
            {
                ("GET", "/info"): httpx.Response(
                    200, json={"model_id": "m", "router": "text-generation-router"}
                ),
            },
            "tgi",
        ),
        (
            {
                ("GET", "/api/extra/version"): httpx.Response(
                    200, json={"result": "KoboldCpp", "version": "1.80"}
                ),
                ("GET", "/v1/models"): httpx.Response(200, json={"data": []}),
            },
            "koboldcpp",
        ),
        (
            {
                ("GET", "/health"): httpx.Response(200, json={"status": "ok"}),
                ("GET", "/props"): httpx.Response(200, json={"total_slots": 1}),
                ("GET", "/v1/models"): httpx.Response(200, json={"data": []}),
            },
            "llama.cpp",
        ),
    ],
)
async def test_flavor_detection(routes, flavor) -> None:
    transport, _ = _mock(routes)
    r = await probe_url("http://box:8000", transport=transport)
    assert r.flavor == flavor
    assert r.suggested_type == "openai"


# --- provider manager ---


def _host_routes(httpx_mock, handler) -> None:
    httpx_mock.add_callback(handler, is_reusable=True, is_optional=True)


async def _openai_provider(db, url: str, caps: dict | None = None):
    p = await db.add_provider("srv", ProviderProtocol.OPENAI)
    await db.add_address(p.id, url, True)
    if caps is not None:
        await db.set_provider_capabilities(p.id, caps)
    return await db.get_provider(p.id)


async def test_health_check_uses_probed_liveness(db, httpx_mock) -> None:
    paths: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        paths.append(request.url.path)
        if request.url.path == "/v1/models":
            return httpx.Response(200, json={"data": [{"id": "gpt-x"}]})
        return httpx.Response(404)

    _host_routes(httpx_mock, handler)
    p = await _openai_provider(
        db,
        "https://api.openai.com/v1",
        {"liveness": "/models", "endpoints": [{"path": "/props", "state": "missing"}]},
    )
    pm = ProviderManager(db)
    status = await pm.check_provider(p, rediscover=True)
    await pm.stop()
    assert status == ProviderStatus.IDLE
    assert [m.name for m in await db.get_provider_models(p.id)] == ["gpt-x"]
    # The probed GET was the health check, and no /props was asked for.
    assert "/props" not in paths and "/health" not in paths
    assert paths.count("/v1/models") >= 2


async def test_add_probes_in_background_and_stores_flavor(
    db, httpx_mock, monkeypatch
) -> None:
    monkeypatch.setattr(settings, "auto_probe_enabled", True)

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path == "/health":
            return httpx.Response(200, json={"status": "ok"})
        if path == "/version":
            return httpx.Response(200, json={"version": "0.11.0"})
        if path == "/ping":
            return httpx.Response(200)
        if path in ("/v1/models", "/models"):
            return httpx.Response(
                200, json={"data": [{"id": "qwen", "owned_by": "vllm"}]}
            )
        return httpx.Response(404)

    _host_routes(httpx_mock, handler)
    pm = ProviderManager(db)
    p = await pm.add_provider("vl", "http://vl:8000", ProviderProtocol.OPENAI)
    await asyncio.gather(*list(pm._probing.values()))
    fresh = await db.get_provider(p.id)
    await pm.stop()
    assert fresh.flavor == "vllm"
    assert fresh.liveness_path == "/health"
    assert fresh.probed_at is not None


async def test_stale_probe_is_rescheduled(db, monkeypatch) -> None:
    pm = ProviderManager(db)
    old = datetime.now(timezone.utc) - timedelta(hours=25)
    p = Provider(name="x", probed_at=old)
    assert pm._probe_is_stale(p)
    assert not pm._probe_is_stale(
        Provider(name="y", probed_at=datetime.now(timezone.utc))
    )
    assert pm._probe_is_stale(Provider(name="z"))
    monkeypatch.setattr(settings, "probe_max_age_hours", 0)
    assert not pm._probe_is_stale(p)
    await pm.stop()


async def test_schedule_probe_respects_the_setting(db, monkeypatch) -> None:
    pm = ProviderManager(db)
    pm.schedule_probe(1)
    assert pm._probing == {}
    await pm.stop()
