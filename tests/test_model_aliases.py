"""Model aliases: one model under different names at different providers."""

from __future__ import annotations

from llama_router.dashboard.routes import suggest_aliases
from llama_router.models import (
    BenchmarkResult,
    ProviderModel,
    ProviderProtocol,
    ProviderStatus,
)
from llama_router.provider_manager import ProviderManager
from llama_router.router import Router, RoutingPreferences
from llama_router.routing_table import alias_groups
from tests.test_dashboard_auth import dash_client  # noqa: F401  (fixture)

CHAT = "/v1/chat/completions"


async def _provider(db, name, models, startup=100.0, locality="local"):
    p = await db.add_provider(name, ProviderProtocol.OPENAI)
    await db.update_provider_settings(p.id, locality=locality)
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
    return p


def test_alias_groups_follow_chains_and_stop_at_loops() -> None:
    g = alias_groups({"a": "b", "b": "c", "x": "c", "p": "q", "q": "p"})
    assert g["a"] == g["c"] == frozenset({"a", "b", "c", "x"})
    assert g["p"] == frozenset({"p", "q"})
    assert "zzz" not in g


async def test_either_name_reaches_both_providers(db) -> None:
    pm = ProviderManager(db)
    direct = await _provider(db, "anthropic", ["claude-x"], startup=5000)
    relay = await _provider(db, "openrouter", ["anthropic/claude-x"], startup=60)
    await db.set_model_alias("anthropic/claude-x", "claude-x")
    rt = Router(db, pm)
    for asked in ("claude-x", "anthropic/claude-x"):
        r = await rt.route(asked, CHAT)
        assert r.provider.id == relay.id
        assert r.backend_model == "anthropic/claude-x"
    # Each provider gets its own name for the model.
    r = await rt.route(
        "claude-x", CHAT, RoutingPreferences(pinned_providers={"claude-x": direct.id})
    )
    assert (r.provider.id, r.backend_model) == (direct.id, "claude-x")
    await db.remove_model_alias("anthropic/claude-x")
    await rt.rebuild()
    assert (await rt.route("claude-x", CHAT)).provider.id == direct.id


async def test_alias_lets_a_local_key_keep_the_model_name(db) -> None:
    pm = ProviderManager(db)
    await _provider(db, "cloud", ["vendor/m"], locality="cloud")
    lan = await _provider(db, "lan", ["m"])
    await db.set_model_alias("vendor/m", "m")
    rt = Router(db, pm)
    r = await rt.route("vendor/m", CHAT, RoutingPreferences(locality="local"))
    assert (r.provider.id, r.backend_model) == (lan.id, "m")


def test_suggestions() -> None:
    names = {
        1: {"claude-x", "llama3.2"},
        2: {"anthropic/claude-x", "openai/gpt-y", "llama3.2:latest"},
    }
    got = suggest_aliases(names, {})
    assert {(s["alias"], s["canonical"]) for s in got} == {
        ("anthropic/claude-x", "claude-x"),
        ("llama3.2:latest", "llama3.2"),
    }
    assert suggest_aliases(names, {"anthropic/claude-x": "claude-x"}) == [
        {"alias": "llama3.2:latest", "canonical": "llama3.2"}
    ]


async def test_alias_api(dash_client) -> None:
    client, db = dash_client
    r = await client.post("/api/model-aliases", json={"alias": "a/m", "canonical": "m"})
    assert r.status_code == 200
    assert await db.get_model_aliases() == {"a/m": "m"}
    r = await client.post("/api/model-aliases", json={"alias": "m", "canonical": "a/m"})
    assert r.status_code == 400  # loop
    r = await client.post("/api/model-aliases", json={"alias": "m", "canonical": "m"})
    assert r.status_code == 400
    page = await client.get("/")
    assert 'data-remove-alias="a/m"' in page.text
    data = (await client.get("/api/model-aliases")).json()
    assert data["aliases"] == {"a/m": "m"}
    assert (await client.delete("/api/model-aliases/a/m")).status_code == 200
    assert (await client.delete("/api/model-aliases/a/m")).status_code == 404
