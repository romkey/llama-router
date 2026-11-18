"""Private model groups: groups a dashboard user defines for their own keys."""

from __future__ import annotations

import httpx

from llama_router.dashboard.auth_core import AUTH_COOKIE, hash_password, sign_session
from llama_router.dashboard.middleware import invalidate_dashboard_user_count_cache
from llama_router.database import Database
from llama_router.model_listing import listed_models
from llama_router.models import (
    BenchmarkResult,
    Provider,
    ProviderModel,
    ProviderProtocol,
    ProviderStatus,
)
from llama_router.provider_manager import ProviderManager
from llama_router.provider_secrets import session_signing_key
from llama_router.router import Router, RoutingPreferences
from llama_router.routing_table import UNROUTABLE, ModelAccess, user_group_key

CHAT = "/v1/chat/completions"
DashClient = tuple[httpx.AsyncClient, Database]


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


async def _people(db: Database) -> tuple[int, int]:
    ann = await db.create_dashboard_user("ann", hash_password("pw-123456789"), False)
    bob = await db.create_dashboard_user("bob", hash_password("pw-123456789"), False)
    invalidate_dashboard_user_count_cache()
    return ann, bob


def _as(user_id: int) -> dict[str, str]:
    token = sign_session(user_id, session_signing_key())
    return {"Cookie": f"{AUTH_COOKIE}={token}"}


def _prefs(owner: int | None, **kw: object) -> RoutingPreferences:
    return RoutingPreferences(owner_user_id=owner, **kw)  # type: ignore[arg-type]


# --- Routing ---


async def test_private_group_routes_for_its_owner_only(
    db: Database, pm: ProviderManager
) -> None:
    await _provider(db, "a", {"llama": 100.0, "qwen": 900.0})
    await _provider(db, "b", {"qwen": 50.0})
    ann, bob = await _people(db)
    await db.create_user_model_group(ann, "mine", members=[("qwen", None)])
    rt = Router(db, pm)

    got = await rt.route("mine", CHAT, _prefs(ann))
    assert got is not None
    assert (got.provider.name, got.model_key) == ("b", "qwen")
    # Shown and logged by the name the owner gave it.
    assert (got.resolved_model, got.group) == ("mine", "mine")
    # Nobody else has a group of that name.
    assert await rt.route("mine", CHAT, _prefs(bob)) is None
    assert await rt.route("mine", CHAT, None) is None
    # The table key is not a back door to someone's group.
    key = user_group_key(ann, "mine")
    assert await rt.route(key, CHAT, _prefs(bob)) is None
    assert await rt.route(key, CHAT, _prefs(ann)) is None
    # Not counted in the shared group metrics.
    assert rt.group_routes == {}


async def test_private_group_hides_a_shared_name_for_its_owner_only(
    db: Database, pm: ProviderManager
) -> None:
    await _provider(db, "a", {"llama": 100.0, "qwen": 100.0})
    ann, bob = await _people(db)
    await db.create_user_model_group(ann, "llama", members=[("qwen", None)])
    rt = Router(db, pm)

    mine = await rt.route("llama", CHAT, _prefs(ann))
    theirs = await rt.route("llama", CHAT, _prefs(bob))
    assert mine is not None
    assert theirs is not None
    assert mine.model_key == "qwen"
    assert theirs.model_key == "llama"


async def test_group_can_include_the_model_it_shadows(
    db: Database, pm: ProviderManager
) -> None:
    await _provider(db, "a", {"llama": 100.0, "qwen": 900.0})
    ann, _ = await _people(db)
    # "llama" for ann: the real llama, or qwen when llama can't serve.
    await db.create_user_model_group(
        ann, "llama", members=[("llama", None), ("qwen", None)]
    )
    rt = Router(db, pm)
    table = await rt.current_table()
    got = await rt.route("llama", CHAT, _prefs(ann))
    assert got is not None
    assert (got.model_key, got.group) == ("llama", "llama")
    key = user_group_key(ann, "llama")
    assert {c.model for c in table.all_candidates[(key, CHAT)]} == {"llama", "qwen"}
    access = ModelAccess(models=frozenset({"llama"}))
    assert not table.allows(access, "llama", ann)
    assert table.allows(ModelAccess(models=frozenset({"llama", "qwen"})), "llama", ann)


async def test_member_tied_to_a_provider_and_no_fallbacks(
    db: Database, pm: ProviderManager
) -> None:
    await _provider(db, "fast", {"qwen": 10.0})
    slow = await _provider(db, "slow", {"qwen": 900.0})
    ann, _ = await _people(db)
    await db.create_user_model_group(
        ann, "pinned", members=[("qwen", slow.require_id())]
    )
    await db.create_user_model_group(ann, "nothing", members=[("absent", None)])
    await db.set_model_fallback("nothing", "qwen")
    rt = Router(db, pm)

    got = await rt.route("pinned", CHAT, _prefs(ann))
    assert got is not None
    assert got.provider.name == "slow"
    # A fallback set for a shared name does not apply to the private group.
    assert await rt.route("nothing", CHAT, _prefs(ann)) is None


async def test_limited_key_needs_every_member(
    db: Database, pm: ProviderManager
) -> None:
    await _provider(db, "a", {"llama": 100.0, "qwen": 100.0})
    ann, _ = await _people(db)
    await db.create_user_model_group(
        ann, "both", members=[("llama", None), ("qwen", None)]
    )
    await db.create_user_model_group(ann, "one", members=[("llama", None)])
    rt = Router(db, pm)
    table = await rt.current_table()
    access = ModelAccess(models=frozenset({"llama"}))

    assert table.allows(access, "one", ann)
    assert not table.allows(access, "both", ann)
    assert await rt.route("both", CHAT, _prefs(ann, access=access)) is None
    got = await rt.route("one", CHAT, _prefs(ann, access=access))
    assert got is not None
    assert got.model_key == "llama"


async def test_resolve_costs_nothing_without_private_groups(
    db: Database, pm: ProviderManager
) -> None:
    await _provider(db, "a", {"llama": 100.0})
    rt = Router(db, pm)
    table = await rt.current_table()
    assert table.private_models == {}
    assert table.resolve(7, "llama") == "llama"
    assert table.resolve(None, "~7/x") == "~7/x"


async def test_hub_requests_never_reach_private_groups(
    db: Database, pm: ProviderManager
) -> None:
    await _provider(db, "a", {"qwen": 100.0})
    ann, _ = await _people(db)
    await db.create_user_model_group(ann, "mine", members=[("qwen", None)])
    rt = Router(db, pm)
    assert await rt.route("mine", CHAT, _prefs(ann, hub_only=True)) is None


async def test_changes_rebuild_the_table(db: Database, pm: ProviderManager) -> None:
    await _provider(db, "a", {"llama": 100.0, "qwen": 100.0})
    ann, _ = await _people(db)
    rt = Router(db, pm)
    assert (await rt.current_table()).resolve(ann, "mine") == "mine"

    g = await db.create_user_model_group(ann, "mine", members=[("llama", None)])
    assert g.id is not None
    got = await rt.route("mine", CHAT, _prefs(ann))
    assert got is not None
    assert got.model_key == "llama"

    await db.update_user_model_group(
        ann, g.id, name="mine", description=None, members=[("qwen", None)]
    )
    got = await rt.route("mine", CHAT, _prefs(ann))
    assert got is not None
    assert got.model_key == "qwen"

    await db.delete_dashboard_user(ann)
    assert await db.list_user_model_groups() == []
    assert (await rt.current_table()).resolve(ann, "mine") == "mine"


async def test_explain_shows_the_private_group(
    db: Database, pm: ProviderManager
) -> None:
    await _provider(db, "a", {"qwen": 100.0})
    ann, _ = await _people(db)
    await db.create_user_model_group(ann, "mine", members=[("qwen", None)])
    rt = Router(db, pm)
    out = await rt.explain("mine", CHAT, _prefs(ann))
    assert out["chosen"]["provider"] == "a"
    assert out["chosen"]["fell_back"] is False
    assert out["fallback_chain"] == ["mine"]
    step = out["steps"][0]
    assert step["group"] == {"name": "mine", "members": ["qwen"], "private": True}
    assert out["summary"].startswith("Routes to a")


async def test_listing_shows_the_owners_groups(
    db: Database, pm: ProviderManager
) -> None:
    await _provider(db, "a", {"llama": 100.0, "qwen": 100.0})
    ann, bob = await _people(db)
    await db.create_user_model_group(ann, "llama", members=[("qwen", None)])
    await db.create_user_model_group(ann, "ghost", members=[("absent", None)])
    # Nothing serves it, yet routing sends "qwen" to it for ann.
    await db.create_user_model_group(ann, "qwen", members=[("absent", None)])
    rt = Router(db, pm)

    mine = await listed_models(db, rt, owner_user_id=ann)
    by_name = {m["name"]: m for m in mine}
    # Replaces the model of the same name; a group nothing serves is hidden,
    # and so is the model it shadows, which ann can't reach by that name.
    assert sorted(by_name) == ["llama"]
    assert by_name["llama"]["details"]["private"] is True
    assert by_name["llama"]["details"]["members"] == ["qwen"]
    theirs = await listed_models(db, rt, owner_user_id=bob)
    assert sorted(m["name"] for m in theirs) == ["llama", "qwen"]
    assert "private" not in {m["name"]: m for m in theirs}["llama"]["details"]
    assert await listed_models(
        db, rt, owner_user_id=ann, hub_only=True
    ) == await listed_models(db, rt, hub_only=True)


async def test_internal_key_unroutable_constant() -> None:
    assert UNROUTABLE == ""


# --- Dashboard API ---


async def test_user_manages_own_groups(dash_client: DashClient) -> None:
    c, db = dash_client
    p = await _provider(db, "a", {"llama": 100.0, "qwen": 100.0})
    ann, bob = await _people(db)

    page = await c.get("/my-groups", headers=_as(ann))
    assert page.status_code == 200
    assert "groups-body" in page.text

    resp = await c.post(
        "/api/my-groups",
        json={
            "name": "coder",
            "description": "for coding",
            "members": [
                {"model_name": "qwen"},
                {"model_name": "llama", "provider_id": p.id},
            ],
        },
        headers=_as(ann),
    )
    assert resp.status_code == 201, resp.text
    gid = resp.json()["id"]

    listed = (await c.get("/api/my-groups", headers=_as(ann))).json()
    assert [g["name"] for g in listed["groups"]] == ["coder"]
    assert [m["label"] for m in listed["groups"][0]["members"]] == [
        "qwen",
        "llama@a",
    ]
    assert "qwen" in listed["models"]
    assert listed["providers"] == [{"id": p.id, "name": "a"}]
    # Bob sees none of it and can't change it.
    assert (await c.get("/api/my-groups", headers=_as(bob))).json()["groups"] == []
    resp = await c.patch(f"/api/my-groups/{gid}", json={"name": "x"}, headers=_as(bob))
    assert resp.status_code == 404
    assert (
        await c.delete(f"/api/my-groups/{gid}", headers=_as(bob))
    ).status_code == 404

    # A member may be the model the group's name shadows.
    resp = await c.patch(
        f"/api/my-groups/{gid}",
        json={"name": "llama", "members": [{"model_name": "llama"}]},
        headers=_as(ann),
    )
    assert resp.status_code == 200, resp.text
    resp = await c.patch(
        f"/api/my-groups/{gid}",
        json={"name": "coding", "members": [{"model_name": "llama"}]},
        headers=_as(ann),
    )
    assert resp.status_code == 200, resp.text
    group = await db.get_user_model_group(ann, gid)
    assert group is not None
    assert (group.name, group.description) == ("coding", "for coding")
    assert [m.model_name for m in group.members] == ["llama"]

    resp = await c.patch(
        f"/api/my-groups/{gid}", json={"description": ""}, headers=_as(ann)
    )
    assert resp.status_code == 200
    group = await db.get_user_model_group(ann, gid)
    assert group is not None
    assert group.description is None
    assert [m.model_name for m in group.members] == ["llama"]

    assert (
        await c.delete(f"/api/my-groups/{gid}", headers=_as(ann))
    ).status_code == 200
    assert await db.list_user_model_groups(ann) == []
    actions = [e["action"] for e in await db.list_audit_log()]
    assert {"my_model_group.create", "my_model_group.update"} <= set(actions)
    assert "my_model_group.delete" in actions


async def test_group_validation(dash_client: DashClient) -> None:
    c, db = dash_client
    await _provider(db, "a", {"llama": 100.0})
    ann, bob = await _people(db)
    shared = await db.create_model_group("shared")
    assert shared.id is not None
    ok = await c.post(
        "/api/my-groups", json={"name": "g", "members": []}, headers=_as(ann)
    )
    assert ok.status_code == 201

    async def post(body: dict[str, object]) -> int:
        return (await c.post("/api/my-groups", json=body, headers=_as(ann))).status_code

    assert await post({"name": ""}) == 400
    assert await post({"name": "x" * 256}) == 400
    assert await post({"name": "g"}) == 409
    assert await post({"name": "h", "members": "llama"}) == 400
    assert await post({"name": "h", "members": [{"model_name": ""}]}) == 400
    assert await post({"name": "h", "members": [{"model_name": "shared"}]}) == 400
    assert await post({"name": "h", "members": [{"model_name": "g"}]}) == 400
    assert (
        await post(
            {"name": "h", "members": [{"model_name": "llama", "provider_id": "x"}]}
        )
        == 400
    )
    assert (
        await post(
            {"name": "h", "members": [{"model_name": "llama", "provider_id": 999}]}
        )
        == 404
    )
    assert (
        await post(
            {"name": "h", "members": [{"model_name": f"m{i}"} for i in range(51)]}
        )
        == 400
    )
    assert await post({"name": "h", "description": "d" * 1001}) == 400
    # The same name is fine for someone else.
    resp = await c.post("/api/my-groups", json={"name": "g"}, headers=_as(bob))
    assert resp.status_code == 201
    # Renaming onto another of your groups is refused.
    h = (await c.post("/api/my-groups", json={"name": "h"}, headers=_as(ann))).json()
    resp = await c.patch(
        f"/api/my-groups/{h['id']}", json={"name": "g"}, headers=_as(ann)
    )
    assert resp.status_code == 409


async def test_group_limit(dash_client: DashClient, monkeypatch: object) -> None:
    from llama_router.config import settings

    c, db = dash_client
    ann, _ = await _people(db)
    old = settings.max_model_groups_per_user
    settings.max_model_groups_per_user = 1
    try:
        first = await c.post("/api/my-groups", json={"name": "a"}, headers=_as(ann))
        assert first.status_code == 201
        second = await c.post("/api/my-groups", json={"name": "b"}, headers=_as(ann))
        assert second.status_code == 409
    finally:
        settings.max_model_groups_per_user = old


async def test_management_keys_cannot_use_my_groups(dash_client: DashClient) -> None:
    c, db = dash_client
    admin = await db.create_dashboard_user("adm", hash_password("pw-123456789"), True)
    invalidate_dashboard_user_count_cache()
    r = await c.post(
        "/api/management-keys",
        json={"name": "bot", "privileges": ["keys:write", "models:write"]},
        headers=_as(admin),
    )
    assert r.status_code == 200, r.text
    bearer = {"Authorization": f"Bearer {r.json()['management_key']}"}
    assert (await c.get("/api/my-groups", headers=bearer)).status_code == 403
    resp = await c.post("/api/my-groups", json={"name": "g"}, headers=bearer)
    assert resp.status_code == 403


# --- Wiring: the OpenAI API and dashboard chat ---


async def test_v1_models_lists_the_key_owners_groups(
    db: Database, pm: ProviderManager
) -> None:
    from llama_router.auth import generate_api_key, key_prefix
    from llama_router.llamacpp_api import deps as v1_deps
    from llama_router.llamacpp_api.app import app as v1_app
    from llama_router.provider_secrets import key_hash
    from tests.conftest import asgi_client

    await _provider(db, "a", {"qwen": 100.0})
    ann, bob = await _people(db)
    await db.create_user_model_group(ann, "mine", members=[("qwen", None)])
    keys = {}
    for who in (ann, bob):
        plaintext = generate_api_key()
        await db.create_api_key(
            key_prefix(plaintext),
            key_hash(plaintext),
            "balanced",
            True,
            owner_user_id=who,
        )
        keys[who] = {"Authorization": f"Bearer {plaintext}"}
    v1_deps.init(db, pm, Router(db, pm))
    try:
        async with asgi_client(v1_app) as client:
            ids = {
                who: [
                    m["id"]
                    for m in (await client.get("/v1/models", headers=h)).json()["data"]
                ]
                for who, h in keys.items()
            }
            one = await client.get("/v1/models/mine", headers=keys[ann])
            none = await client.get("/v1/models/mine", headers=keys[bob])
    finally:
        v1_deps._db, v1_deps._pm, v1_deps._router = None, None, None
    assert ids[ann] == ["mine", "qwen"]
    assert ids[bob] == ["qwen"]
    assert (one.status_code, none.status_code) == (200, 404)


async def test_chat_targets_include_the_users_groups(dash_client: DashClient) -> None:
    c, db = dash_client
    await _provider(db, "a", {"qwen": 100.0})
    ann, bob = await _people(db)
    await db.create_user_model_group(ann, "mine", members=[("qwen", None)])
    pm = ProviderManager(db)
    rt = Router(db, pm)
    from llama_router.dashboard import deps as dash_deps

    dash_deps.init(db, pm)
    pm.attach_router(rt)
    await rt.current_table()
    mine = (await c.get("/api/chat/targets", headers=_as(ann))).json()
    theirs = (await c.get("/api/chat/targets", headers=_as(bob))).json()
    assert mine["any"]["groups"] == ["mine"]
    assert theirs["any"]["groups"] == []
