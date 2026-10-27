from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from llama_router.auth import key_hash
from llama_router.dashboard import deps as dash_deps
from llama_router.dashboard.app import app as dash_app
from llama_router.database import Database
from llama_router.models import (
    BenchmarkResult,
    ProviderModel,
    ProviderStatus,
    ProviderType,
)
from llama_router.provider_manager import ProviderManager
from llama_router.router import (
    ENDPOINT_PROTOCOLS,
    Router,
    RoutingPreferences,
    protocol_for_endpoint,
)


async def _provider(db: Database, name: str, models: list[str], **kw):
    status = kw.pop("status", ProviderStatus.IDLE)
    p = await db.add_provider(name, kw.pop("ptype", ProviderType.OLLAMA))
    await db.update_provider_status(p.id, status)
    await db.set_provider_models(
        p.id, [ProviderModel(provider_id=p.id, name=m) for m in models]
    )
    return p


async def _bench(
    db: Database,
    pid: int,
    model: str,
    startup: float,
    tps: float,
    protocol: str | None = "ollama",
):
    await db.save_benchmark(
        BenchmarkResult(
            provider_id=pid,
            model_name=model,
            protocol=protocol,
            startup_time_ms=startup,
            tokens_per_second=tps,
        )
    )


def _by_name(step: dict) -> dict[str, dict]:
    return {c["provider"]: c for c in step["candidates"]}


@pytest.fixture
def pm(db: Database) -> ProviderManager:
    return ProviderManager(db)


def test_endpoint_protocols():
    assert protocol_for_endpoint("/api/chat") == "ollama"
    assert protocol_for_endpoint("/chat/completions") is None
    assert protocol_for_endpoint("/v1/audio/speech") is None
    assert "/v1/images/edits" in ENDPOINT_PROTOCOLS


@pytest.mark.parametrize("mode", ["latency", "throughput", "balanced"])
async def test_explanation_matches_real_decision(
    db: Database, pm: ProviderManager, mode: str
):
    a = await _provider(db, "a", ["m"])
    b = await _provider(db, "b", ["m"])
    c = await _provider(db, "c", ["m"])
    await _bench(db, a.id, "m", startup=100, tps=10)
    await _bench(db, b.id, "m", startup=900, tps=200)
    await _bench(db, c.id, "m", startup=400, tps=60)
    pm._active_requests[b.id] = 1
    rt = Router(db, pm)
    prefs = RoutingPreferences(mode=mode)
    real = await rt.route("m", "/v1/chat/completions", preferences=prefs)
    out = await rt.explain("m", "/v1/chat/completions", preferences=prefs)
    assert out["chosen"]["provider_id"] == real.provider.id
    step = out["steps"][0]
    ranked = [c for c in step["candidates"] if c["rank"]]
    assert ranked[0]["provider_id"] == real.provider.id
    assert [c["rank"] for c in ranked] == [1, 2, 3]
    # Totals include the live queue.
    row_b = _by_name(step)["b"]
    assert row_b["active_requests"] == 1
    assert row_b["total_score"] == pytest.approx(
        row_b["static_score"] + row_b["queue_cost"]
    )
    assert real.provider.name in out["summary"]


async def test_explains_unusable_providers(db: Database, pm: ProviderManager):
    good = await _provider(db, "good", ["m"])
    await _provider(db, "down", ["m"], status=ProviderStatus.OFFLINE)
    await _provider(db, "lcpp", ["m"], ptype=ProviderType.LLAMACPP)
    flaky = await _provider(db, "flaky", ["m"])
    for _ in range(5):
        pm.stats.record_request(flaky.id, "m", ok=False)
    rt = Router(db, pm)
    out = await rt.explain("m", "/api/chat")
    rows = _by_name(out["steps"][0])
    assert out["chosen"]["provider_id"] == good.id
    assert rows["down"]["excluded"] == "provider is offline"
    assert rows["lcpp"]["excluded"] == "provider does not serve the Ollama API"
    assert "circuit open" in rows["flaky"]["excluded"]
    assert rows["good"]["rank"] == 1


async def test_explains_missing_endpoint(db: Database, pm: ProviderManager):
    speechless = await _provider(db, "speechless", ["m"], ptype=ProviderType.LLAMACPP)
    full = await _provider(db, "full", ["m"], ptype=ProviderType.LLAMACPP)
    await _bench(db, speechless.id, "m", startup=10, tps=500)
    await db.set_provider_capabilities(
        speechless.id, {"endpoints": [{"path": "/v1/audio/speech", "state": "missing"}]}
    )
    rt = Router(db, pm)
    out = await rt.explain("m", "/v1/audio/speech")
    rows = _by_name(out["steps"][0])
    assert out["chosen"]["provider_id"] == full.id
    assert rows["speechless"]["excluded"].startswith(
        "last probe found /v1/audio/speech missing on its llamacpp API"
    )


async def test_explains_fallback_and_pins(db: Database, pm: ProviderManager):
    a = await _provider(db, "a", ["small"])
    b = await _provider(db, "b", ["small"])
    await _bench(db, a.id, "small", startup=10, tps=100)
    await db.set_model_fallback("big", "small")
    rt = Router(db, pm)

    out = await rt.explain("big", "/v1/chat/completions")
    assert out["fallback_chain"] == ["big", "small"]
    assert out["steps"][0]["outcome"] == "no provider can serve this model"
    assert out["chosen"]["model"] == "small" and out["chosen"]["fell_back"]
    assert "falling back" in out["summary"]

    no_fb = await rt.explain(
        "big",
        "/v1/chat/completions",
        preferences=RoutingPreferences(allow_fallback=False),
    )
    assert no_fb["chosen"] is None and len(no_fb["steps"]) == 1

    pinned = await rt.explain(
        "small",
        "/v1/chat/completions",
        preferences=RoutingPreferences(pinned_providers={"small": b.id}),
    )
    rows = _by_name(pinned["steps"][0])
    assert pinned["chosen"]["provider_id"] == b.id
    assert pinned["steps"][0]["outcome"] == "pinned to b"
    assert rows["a"]["excluded"] == "another provider is pinned for this model"


async def test_explains_chaos(db: Database, pm: ProviderManager):
    await _provider(db, "a", ["m1", "m2"])
    rt = Router(db, pm)
    out = await rt.explain(
        "anything", "/v1/chat/completions", preferences=RoutingPreferences(mode="chaos")
    )
    assert out["chaos_pool"][0]["models"] == ["m1", "m2"]
    assert "random" in out["summary"] and "this time: a" in out["summary"]


async def test_explains_empty_chaos_pool(db: Database, pm: ProviderManager):
    await _provider(db, "down", ["m"], status=ProviderStatus.OFFLINE)
    rt = Router(db, pm)
    out = await rt.explain(
        "m", "/v1/chat/completions", preferences=RoutingPreferences(mode="chaos")
    )
    assert out["chosen"] is None and out["chaos_pool"] == []
    assert out["summary"].startswith("No provider can serve this request")


@pytest.fixture
async def client(db: Database, pm: ProviderManager):
    rt = Router(db, pm)
    await rt.rebuild()
    dash_deps.init(db, pm)
    with TestClient(dash_app) as c:
        yield c


async def test_explain_api(db: Database, pm: ProviderManager, client):
    a = await _provider(db, "a", ["m"])
    b = await _provider(db, "b", ["m"])
    await _bench(db, a.id, "m", startup=100, tps=10)
    await _bench(db, b.id, "m", startup=900, tps=500)

    data = client.get(
        "/api/routing/explain", params={"model": "m", "mode": "latency"}
    ).json()
    assert data["chosen"]["provider"] == "a"
    assert data["request"]["endpoint"] == "/v1/chat/completions"

    data = client.get(
        "/api/routing/explain",
        params={"model": "m", "mode": "throughput", "endpoint": "/api/chat"},
    ).json()
    assert data["chosen"]["provider"] == "b"
    assert data["request"]["protocol"] == "ollama"

    key_id = await db.create_api_key("lrk_x", key_hash("lrk_x"), "latency", True)
    await db.set_api_key_model_pin(key_id, "m", b.id)
    data = client.get(
        "/api/routing/explain", params={"model": "m", "key_id": key_id}
    ).json()
    assert data["chosen"]["provider"] == "b"
    assert data["api_key"]["id"] == key_id
    assert data["request"]["pins"] == {"m": b.id}


async def test_explain_api_validation(client):
    assert client.get("/api/routing/explain").status_code == 400
    assert (
        client.get(
            "/api/routing/explain", params={"model": "m", "endpoint": "/api/tags"}
        ).status_code
        == 400
    )
    assert (
        client.get(
            "/api/routing/explain", params={"model": "m", "mode": "fastest"}
        ).status_code
        == 400
    )
    assert (
        client.get(
            "/api/routing/explain", params={"model": "m", "key_id": "999"}
        ).status_code
        == 404
    )


async def test_dashboard_renders_explorer(db: Database, pm: ProviderManager, client):
    await _provider(db, "a", ["m"])
    page = client.get("/").text
    assert "Route Explorer" in page and 'id="rx-models"' in page
