"""Model groups: one name that routes to the best of several models."""

from __future__ import annotations

from typing import Any

import pytest

from llama_router.affinity import Binding, SessionAffinity
from llama_router.database import Database
from llama_router.models import (
    BenchmarkResult,
    Provider,
    ProviderModel,
    ProviderProtocol,
    ProviderStatus,
)
from llama_router.provider_manager import ProviderManager
from llama_router.router import Router, RoutingPreferences

CHAT = "/v1/chat/completions"


async def _provider(
    db: Database,
    name: str,
    models: dict[str, float],
    *,
    locality: str = "local",
) -> Provider:
    """A provider serving *models*, each benchmarked with its cold start."""
    p = await db.add_provider(name, ProviderProtocol.OPENAI)
    pid = p.require_id()
    await db.update_provider_settings(pid, locality=locality)
    await db.update_provider_status(pid, ProviderStatus.IDLE)
    await db.set_provider_models(
        pid, [ProviderModel(provider_id=pid, name=m) for m in models]
    )
    for m, startup in models.items():
        await _bench(db, pid, m, startup)
    return p


async def _bench(db: Database, pid: int, model: str, startup: float) -> None:
    await db.save_benchmark(
        BenchmarkResult(
            provider_id=pid,
            model_name=model,
            protocol="openai",
            startup_time_ms=startup,
            tokens_per_second=50.0,
        )
    )


async def _group(db: Database, name: str, *members: str | tuple[str, Provider]) -> int:
    g = await db.create_model_group(name)
    gid = g.id
    assert gid is not None
    for m in members:
        if isinstance(m, tuple):
            await db.add_model_group_member(gid, m[0], m[1].require_id())
        else:
            await db.add_model_group_member(gid, m)
    return gid


def _router(db: Database, pm: ProviderManager, **kw: Any) -> Router:
    return Router(
        db,
        pm,
        affinity=SessionAffinity(1800, 1000, kw.pop("clock")),
        session_affinity=True,
    )


def _chat(model: str = "coding") -> dict[str, Any]:
    return {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a coding agent. " * 200},
            {"role": "user", "content": "fix the bug"},
        ],
    }


# --- Storage ---


async def test_groups_are_stored_with_ordered_unique_members(db: Database) -> None:
    p = await _provider(db, "gpu", {"m": 100.0})
    gid = await _group(db, "coding", "a", "b")
    again = await db.add_model_group_member(gid, "a")
    assert again.position == 0  # the existing member, not a new one
    scoped = await db.add_model_group_member(gid, "a", p.require_id())
    assert scoped.position == 2

    g = await db.get_model_group("coding")
    assert g is not None
    assert [(m.model_name, m.provider_id) for m in g.members] == [
        ("a", None),
        ("b", None),
        ("a", p.id),
    ]
    assert scoped.id is not None
    assert await db.remove_model_group_member(gid, scoped.id)
    assert not await db.remove_model_group_member(gid, scoped.id)

    assert await db.update_model_group("coding", description="for code")
    g = await db.get_model_group("coding")
    assert g is not None
    assert g.description == "for code"

    assert await db.delete_model_group("coding")
    assert not await db.delete_model_group("coding")
    assert await db.list_model_groups() == []


async def test_removing_a_provider_drops_its_scoped_members(db: Database) -> None:
    p = await _provider(db, "gpu", {"m": 100.0})
    await _group(db, "coding", ("m", p), "other")
    await db.remove_provider(p.require_id())
    g = await db.get_model_group("coding")
    assert g is not None
    assert [m.model_name for m in g.members] == ["other"]


async def test_group_writes_rebuild_the_routing_table(
    db: Database, pm: ProviderManager
) -> None:
    await _provider(db, "gpu", {"m": 100.0})
    rt = Router(db, pm)
    assert await rt.route("coding", CHAT) is None
    await _group(db, "coding", "m")
    r = await rt.route("coding", CHAT)
    assert r is not None
    assert r.model_key == "m"


# --- Routing ---


async def test_group_routes_to_the_best_member_model(
    db: Database, pm: ProviderManager
) -> None:
    await _provider(db, "slow", {"big-coder": 5000.0})
    fast = await _provider(db, "fast", {"small-coder": 50.0})
    await _group(db, "coding", "big-coder", "small-coder")
    rt = Router(db, pm)

    r = await rt.route("coding", CHAT, RoutingPreferences(mode="latency"))
    assert r is not None
    assert r.provider.id == fast.id
    assert (r.resolved_model, r.model_key, r.backend_model, r.group) == (
        "coding",
        "small-coder",
        "small-coder",
        "coding",
    )
    # Asking for a member by name is unaffected.
    direct = await rt.route("big-coder", CHAT)
    assert direct is not None
    assert direct.group is None
    assert direct.model_key == "big-coder"


async def test_member_scoped_to_a_provider_uses_only_that_provider(
    db: Database, pm: ProviderManager
) -> None:
    slow = await _provider(db, "slow", {"coder": 5000.0})
    await _provider(db, "fast", {"coder": 50.0})
    await _group(db, "coding", ("coder", slow))
    rt = Router(db, pm)
    r = await rt.route("coding", CHAT)
    assert r is not None
    assert r.provider.id == slow.id


async def test_one_provider_can_offer_several_members(
    db: Database, pm: ProviderManager
) -> None:
    gpu = await _provider(db, "gpu", {"big": 5000.0, "small": 50.0})
    await _group(db, "coding", "big", "small")
    rt = Router(db, pm)
    await rt.rebuild()
    table = rt.table
    assert table is not None
    cands = table.all_candidates[("coding", CHAT)]
    assert sorted(c.model for c in cands) == ["big", "small"]
    r = await rt.route("coding", CHAT, RoutingPreferences(mode="latency"))
    assert r is not None
    assert (r.provider.id, r.model_key) == (gpu.id, "small")


async def test_member_reaches_providers_through_aliases(
    db: Database, pm: ProviderManager
) -> None:
    lan = await _provider(db, "lan", {"m": 100.0})
    await db.set_model_alias("vendor/m", "m")
    await _group(db, "vision", "vendor/m")
    rt = Router(db, pm)
    r = await rt.route("vision", CHAT)
    assert r is not None
    assert (r.provider.id, r.backend_model) == (lan.id, "m")


async def test_key_locality_limits_group_members(
    db: Database, pm: ProviderManager
) -> None:
    await _provider(db, "cloud", {"gpt-v": 10.0}, locality="cloud")
    lan = await _provider(db, "lan", {"llava": 3000.0})
    await _group(db, "vision", "gpt-v", "llava")
    rt = Router(db, pm)
    r = await rt.route("vision", CHAT, RoutingPreferences(locality="local"))
    assert r is not None
    assert (r.provider.id, r.model_key) == (lan.id, "llava")


async def test_hub_sees_only_shared_members(db: Database, pm: ProviderManager) -> None:
    shared = await _provider(db, "shared", {"slow": 5000.0})
    private = await _provider(db, "private", {"fast": 50.0})
    await db.set_hub_shared_providers([shared.require_id()])
    await _group(db, "coding", "slow", "fast")
    await _group(db, "secret", "fast")
    rt = Router(db, pm)
    r = await rt.route("coding", CHAT, RoutingPreferences(hub_only=True))
    assert r is not None
    assert r.provider.id == shared.id
    r = await rt.route("coding", CHAT)
    assert r is not None
    assert r.provider.id == private.id
    assert await rt.route("secret", CHAT, RoutingPreferences(hub_only=True)) is None

    table = rt.table
    assert table is not None
    assert table.listed_groups() == ["coding", "secret"]
    assert table.listed_groups(hub_only=True) == ["coding"]


async def test_fallback_into_and_out_of_a_group(
    db: Database, pm: ProviderManager
) -> None:
    await _provider(db, "gpu", {"m": 100.0})
    await _group(db, "coding", "m")
    await _group(db, "empty", "nobody-serves-this")
    await db.set_model_fallback("retired-model", "coding")
    await db.set_model_fallback("empty", "m")
    rt = Router(db, pm)
    r = await rt.route("retired-model", CHAT)
    assert r is not None
    assert (r.resolved_model, r.model_key) == ("coding", "m")
    assert r.group == "coding"
    r = await rt.route("empty", CHAT)
    assert r is not None
    assert (r.resolved_model, r.group) == ("m", None)


async def test_pin_on_a_group_picks_the_best_member_on_that_provider(
    db: Database, pm: ProviderManager
) -> None:
    await _provider(db, "fast", {"small": 60.0})
    gpu = await _provider(db, "gpu", {"big": 5000.0, "mid": 500.0})
    await _group(db, "coding", "big", "mid", "small")
    rt = Router(db, pm)
    prefs = RoutingPreferences(
        mode="latency", pinned_providers={"coding": gpu.require_id()}
    )
    r = await rt.route("coding", CHAT, prefs)
    assert r is not None
    assert (r.provider.id, r.model_key) == (gpu.id, "mid")


async def test_group_with_no_served_member_routes_nowhere(
    db: Database, pm: ProviderManager
) -> None:
    await _provider(db, "gpu", {"m": 100.0})
    await _group(db, "empty", "nobody-serves-this")
    rt = Router(db, pm)
    assert await rt.route("empty", CHAT) is None
    table = rt.table
    assert table is not None
    assert table.groups["empty"] == ("nobody-serves-this",)
    assert table.listed_groups() == []


async def test_groups_do_not_nest(db: Database, pm: ProviderManager) -> None:
    await _provider(db, "gpu", {"m": 100.0})
    await _group(db, "inner", "m")
    await _group(db, "outer", "inner")
    rt = Router(db, pm)
    assert await rt.route("outer", CHAT) is None


async def test_group_shadowing_a_model_is_reported(
    db: Database, pm: ProviderManager
) -> None:
    await _provider(db, "a", {"coding": 100.0})
    b = await _provider(db, "b", {"m": 100.0})
    await _group(db, "coding", "m")
    rt = Router(db, pm)
    r = await rt.route("coding", CHAT)
    assert r is not None
    assert (r.provider.id, r.model_key) == (b.id, "m")
    table = rt.table
    assert table is not None
    assert any("hides a model" in w for w in table.warnings)
    assert table.to_dict()["groups"] == {"coding": ["m"]}


async def test_provider_scoped_member_is_labelled_with_the_provider(
    db: Database, pm: ProviderManager
) -> None:
    gpu = await _provider(db, "gpu", {"m": 100.0})
    await _group(db, "coding", ("m", gpu), "n")
    rt = Router(db, pm)
    await rt.rebuild()
    table = rt.table
    assert table is not None
    assert table.groups["coding"] == ("m@gpu", "n")


# --- Session affinity ---


async def test_session_keeps_its_member_model_when_another_gets_faster(
    db: Database, pm: ProviderManager, clock: Any
) -> None:
    gpu = await _provider(db, "gpu", {"big": 500.0, "small": 5000.0})
    gid = gpu.require_id()
    await _group(db, "coding", "big", "small")
    rt = _router(db, pm, clock=clock)
    prefs = RoutingPreferences(mode="latency")
    session = rt.session_for({}, _chat(), prefs)
    assert session is not None

    first = await rt.route("coding", CHAT, prefs, session=session)
    assert first is not None
    assert first.model_key == "big"
    assert rt.affinity.peek(session) == Binding(gid, "big")

    # "small" is now much faster on the same provider. Without a session the
    # request goes there; the conversation stays on "big".
    await _bench(db, gid, "small", 60.0)
    fresh = await rt.route("coding", CHAT, prefs)
    assert fresh is not None
    assert fresh.model_key == "small"
    again = await rt.route("coding", CHAT, prefs, session=session)
    assert again is not None
    assert again.model_key == "big"
    assert rt.affinity.hits == 1


async def test_session_moves_when_its_member_leaves_the_group(
    db: Database, pm: ProviderManager, clock: Any
) -> None:
    a = await _provider(db, "a", {"big": 50.0})
    b = await _provider(db, "b", {"small": 5000.0})
    gid = await _group(db, "coding", "big", "small")
    rt = _router(db, pm, clock=clock)
    session = rt.session_for({}, _chat(), None)
    first = await rt.route("coding", CHAT, session=session)
    assert first is not None
    assert first.provider.id == a.id

    g = await db.get_model_group("coding")
    assert g is not None
    big = next(m for m in g.members if m.model_name == "big")
    assert big.id is not None
    await db.remove_model_group_member(gid, big.id)
    moved = await rt.route("coding", CHAT, session=session)
    assert moved is not None
    assert (moved.provider.id, moved.model_key) == (b.id, "small")
    assert rt.affinity.rebinds == 1
    assert rt.affinity.peek(session) == Binding(b.require_id(), "small")


async def test_group_and_direct_sessions_are_separate(
    db: Database, pm: ProviderManager, clock: Any
) -> None:
    await _provider(db, "gpu", {"m": 100.0})
    await _group(db, "coding", "m")
    rt = _router(db, pm, clock=clock)
    assert rt.session_for({}, _chat("coding"), None) != rt.session_for(
        {}, _chat("m"), None
    )


async def test_explain_names_the_group_member_and_binding(
    db: Database, pm: ProviderManager, clock: Any
) -> None:
    gpu = await _provider(db, "gpu", {"big": 500.0, "small": 5000.0})
    await _group(db, "coding", "big", "small")
    rt = _router(db, pm, clock=clock)
    prefs = RoutingPreferences(mode="latency")
    session = rt.session_for({}, _chat(), prefs)
    await rt.route("coding", CHAT, prefs, session=session)

    out = await rt.explain("coding", CHAT, prefs, session=session)
    step = out["steps"][0]
    assert step["group"] == {"name": "coding", "members": ["big", "small"]}
    assert step["outcome"] == "kept on big on gpu for this session"
    assert out["session"]["bound_model"] == "big"
    assert out["session"]["honored"] is True
    bound = [c for c in step["candidates"] if c["session_bound"]]
    assert [(c["provider_id"], c["model"]) for c in bound] == [(gpu.id, "big")]


@pytest.mark.parametrize("mode", ["latency", "balanced", "throughput", "cheapest"])
async def test_every_mode_ranks_group_buckets(
    db: Database, pm: ProviderManager, mode: str
) -> None:
    await _provider(db, "a", {"x": 100.0})
    await _provider(db, "b", {"y": 100.0})
    await _group(db, "g", "x", "y")
    rt = Router(db, pm)
    await rt.rebuild()
    table = rt.table
    assert table is not None
    assert len(table.ranked_for("g", CHAT, mode)) == 2
