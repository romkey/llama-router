"""Shared private model groups: an owner shares a group, others opt in."""

from __future__ import annotations

import pytest
from sqlalchemy.exc import IntegrityError

from llama_router.database import Database
from llama_router.model_listing import listed_models
from llama_router.provider_manager import ProviderManager
from llama_router.router import Router
from tests.test_user_model_groups import (
    CHAT,
    DashClient,
    _as,
    _people,
    _prefs,
    _provider,
)


async def _cat(db: Database) -> int:
    from llama_router.dashboard.auth_core import hash_password
    from llama_router.dashboard.middleware import (
        invalidate_dashboard_user_count_cache,
    )

    cat = await db.create_dashboard_user("cat", hash_password("pw-123456789"), False)
    invalidate_dashboard_user_count_cache()
    return cat


async def _router_group(db: Database, name: str, model: str) -> None:
    g = await db.create_model_group(name)
    assert g.id is not None
    await db.add_model_group_member(g.id, model)


# --- Routing ---


async def test_only_users_who_opt_in_can_use_a_shared_group(
    db: Database, pm: ProviderManager
) -> None:
    await _provider(db, "a", {"llama": 100.0, "qwen": 100.0})
    ann, bob = await _people(db)
    cat = await _cat(db)
    g = await db.create_user_model_group(
        ann, "coder", members=[("qwen", None)], shared=True
    )
    assert g.id is not None
    rt = Router(db, pm)

    # Shared, but nobody opted in yet.
    assert await rt.route("coder", CHAT, _prefs(bob)) is None
    assert await db.subscribe_user_model_group(bob, g.id)
    got = await rt.route("coder", CHAT, _prefs(bob))
    assert got is not None
    assert (got.model_key, got.group) == ("qwen", "coder")
    # Cat didn't opt in.
    assert await rt.route("coder", CHAT, _prefs(cat)) is None
    # The owner still uses it.
    assert await rt.route("coder", CHAT, _prefs(ann)) is not None

    # Members change for everyone who opted in.
    await db.update_user_model_group(
        ann, g.id, name="coder", description=None, members=[("llama", None)]
    )
    got = await rt.route("coder", CHAT, _prefs(bob))
    assert got is not None
    assert got.model_key == "llama"

    await db.unsubscribe_user_model_group(bob, g.id)
    assert await rt.route("coder", CHAT, _prefs(bob)) is None


async def test_unsharing_drops_every_opt_in(db: Database, pm: ProviderManager) -> None:
    await _provider(db, "a", {"qwen": 100.0})
    ann, bob = await _people(db)
    g = await db.create_user_model_group(
        ann, "coder", members=[("qwen", None)], shared=True
    )
    assert g.id is not None
    assert await db.subscribe_user_model_group(bob, g.id)
    rt = Router(db, pm)
    assert await rt.route("coder", CHAT, _prefs(bob)) is not None

    await db.update_user_model_group(
        ann, g.id, name="coder", description=None, shared=False
    )
    assert await db.list_user_model_group_subscriptions() == []
    assert await rt.route("coder", CHAT, _prefs(bob)) is None
    # Sharing again doesn't bring the opt-in back.
    await db.update_user_model_group(
        ann, g.id, name="coder", description=None, shared=True
    )
    assert await rt.route("coder", CHAT, _prefs(bob)) is None


async def test_cannot_opt_in_to_unshared_or_own_groups(db: Database) -> None:
    ann, bob = await _people(db)
    private = await db.create_user_model_group(ann, "private")
    shared = await db.create_user_model_group(ann, "shared", shared=True)
    assert private.id is not None
    assert shared.id is not None
    assert not await db.subscribe_user_model_group(bob, private.id)
    assert not await db.subscribe_user_model_group(ann, shared.id)
    assert not await db.subscribe_user_model_group(bob, 999_999)
    # Twice is the same as once.
    assert await db.subscribe_user_model_group(bob, shared.id)
    assert await db.subscribe_user_model_group(bob, shared.id)
    assert await db.list_user_model_group_subscriptions(bob) == [(bob, shared.id)]


async def test_opted_in_group_wins_over_a_router_wide_group(
    db: Database, pm: ProviderManager
) -> None:
    await _provider(db, "a", {"llama": 100.0, "qwen": 100.0})
    ann, bob = await _people(db)
    cat = await _cat(db)
    await _router_group(db, "chat", "llama")
    g = await db.create_user_model_group(
        ann, "chat", members=[("qwen", None)], shared=True
    )
    assert g.id is not None
    assert await db.subscribe_user_model_group(bob, g.id)
    rt = Router(db, pm)

    mine = await rt.route("chat", CHAT, _prefs(bob))
    theirs = await rt.route("chat", CHAT, _prefs(cat))
    assert mine is not None
    assert theirs is not None
    assert mine.model_key == "qwen"
    assert theirs.model_key == "llama"


async def test_own_group_wins_then_the_earliest_opt_in(
    db: Database, pm: ProviderManager
) -> None:
    await _provider(db, "a", {"llama": 100.0, "qwen": 100.0, "phi": 100.0})
    ann, bob = await _people(db)
    cat = await _cat(db)
    first = await db.create_user_model_group(
        ann, "coder", members=[("qwen", None)], shared=True
    )
    later = await db.create_user_model_group(
        bob, "coder", members=[("llama", None)], shared=True
    )
    assert first.id is not None
    assert later.id is not None
    assert await db.subscribe_user_model_group(cat, first.id)
    assert await db.subscribe_user_model_group(cat, later.id)
    rt = Router(db, pm)
    got = await rt.route("coder", CHAT, _prefs(cat))
    assert got is not None
    assert got.model_key == "qwen"

    # Cat's own group of that name wins over both.
    await db.create_user_model_group(cat, "coder", members=[("phi", None)])
    got = await rt.route("coder", CHAT, _prefs(cat))
    assert got is not None
    assert got.model_key == "phi"


async def test_deleting_a_group_or_user_drops_opt_ins(
    db: Database, pm: ProviderManager
) -> None:
    await _provider(db, "a", {"qwen": 100.0})
    ann, bob = await _people(db)
    one = await db.create_user_model_group(
        ann, "one", members=[("qwen", None)], shared=True
    )
    two = await db.create_user_model_group(
        ann, "two", members=[("qwen", None)], shared=True
    )
    assert one.id is not None
    assert two.id is not None
    await db.subscribe_user_model_group(bob, one.id)
    await db.subscribe_user_model_group(bob, two.id)
    assert await db.delete_user_model_group(ann, one.id)
    assert await db.list_user_model_group_subscriptions() == [(bob, two.id)]
    await db.delete_dashboard_user(ann)
    assert await db.list_user_model_group_subscriptions() == []
    rt = Router(db, pm)
    assert await rt.route("two", CHAT, _prefs(bob)) is None


async def test_listing_shows_opted_in_groups(db: Database, pm: ProviderManager) -> None:
    await _provider(db, "a", {"qwen": 100.0})
    ann, bob = await _people(db)
    g = await db.create_user_model_group(
        ann, "coder", members=[("qwen", None)], shared=True
    )
    assert g.id is not None
    rt = Router(db, pm)
    assert [m["name"] for m in await listed_models(db, rt, owner_user_id=bob)] == [
        "qwen"
    ]
    await db.subscribe_user_model_group(bob, g.id)
    rows = {m["name"]: m for m in await listed_models(db, rt, owner_user_id=bob)}
    assert sorted(rows) == ["coder", "qwen"]
    assert rows["coder"]["details"]["shared"] is True
    owner = {m["name"]: m for m in await listed_models(db, rt, owner_user_id=ann)}
    assert owner["coder"]["details"]["shared"] is False


# --- Dashboard API ---


async def test_share_browse_and_opt_in(dash_client: DashClient) -> None:
    c, db = dash_client
    await _provider(db, "a", {"llama": 100.0, "qwen": 100.0})
    await _router_group(db, "chat", "llama")
    ann, bob = await _people(db)
    pm = ProviderManager(db)
    rt = Router(db, pm)
    from llama_router.dashboard import deps as dash_deps

    dash_deps.init(db, pm)
    pm.attach_router(rt)

    # Off by default.
    resp = await c.post(
        "/api/my-groups",
        json={"name": "private", "members": [{"model_name": "qwen"}]},
        headers=_as(ann),
    )
    assert resp.status_code == 201
    resp = await c.post(
        "/api/my-groups",
        json={"name": "chat", "shared": True, "members": [{"model_name": "qwen"}]},
        headers=_as(ann),
    )
    assert resp.status_code == 201, resp.text
    gid = resp.json()["id"]
    assert (
        await c.post(
            "/api/my-groups", json={"name": "x", "shared": "yes"}, headers=_as(ann)
        )
    ).status_code == 400

    await rt.current_table()
    # Bob sees only the shared one, with its shadowing called out.
    shared = (await c.get("/api/shared-groups", headers=_as(bob))).json()["groups"]
    assert [(g["name"], g["owner"]) for g in shared] == [("chat", "ann")]
    assert shared[0]["opted_in"] is False
    assert shared[0]["shadows"] == "group"
    assert shared[0]["choice"] is None
    # Ann doesn't see her own in the shared list.
    assert (await c.get("/api/shared-groups", headers=_as(ann))).json()["groups"] == []

    resp = await c.post(f"/api/shared-groups/{gid}/opt-in", headers=_as(bob))
    assert resp.status_code == 200, resp.text
    await rt.current_table()
    shared = (await c.get("/api/shared-groups", headers=_as(bob))).json()["groups"]
    assert shared[0]["opted_in"] is True
    assert shared[0]["choice"] == "qwen on a"
    got = await rt.route("chat", CHAT, _prefs(bob))
    assert got is not None
    assert got.model_key == "qwen"
    mine = (await c.get("/api/my-groups", headers=_as(ann))).json()["groups"]
    by_name = {g["name"]: g for g in mine}
    assert (by_name["chat"]["shared"], by_name["chat"]["opted_in"]) == (True, 1)
    assert (by_name["private"]["shared"], by_name["private"]["opted_in"]) == (
        False,
        0,
    )
    targets = (await c.get("/api/chat/targets", headers=_as(bob))).json()
    assert "chat" in targets["any"]["groups"]

    # Not someone's unshared group, nor one's own.
    private_id = by_name["private"]["id"]
    assert (
        await c.post(f"/api/shared-groups/{private_id}/opt-in", headers=_as(bob))
    ).status_code == 404
    assert (
        await c.post(f"/api/shared-groups/{gid}/opt-in", headers=_as(ann))
    ).status_code == 404

    resp = await c.delete(f"/api/shared-groups/{gid}/opt-in", headers=_as(bob))
    assert resp.status_code == 200
    assert await db.list_user_model_group_subscriptions() == []
    actions = {e["action"] for e in await db.list_audit_log()}
    assert {"my_model_group.opt_in", "my_model_group.opt_out"} <= actions


async def test_opt_in_refuses_a_name_already_in_use(dash_client: DashClient) -> None:
    c, db = dash_client
    ann, bob = await _people(db)
    cat = await _cat(db)
    a = await db.create_user_model_group(ann, "coder", shared=True)
    b = await db.create_user_model_group(bob, "coder", shared=True)
    assert a.id is not None
    assert b.id is not None
    # Bob has his own "coder".
    resp = await c.post(f"/api/shared-groups/{a.id}/opt-in", headers=_as(bob))
    assert resp.status_code == 409
    shared = (await c.get("/api/shared-groups", headers=_as(bob))).json()["groups"]
    assert shared[0]["hidden_by"] == "own"
    # Cat already uses another "coder"; the list says so before the click.
    assert (
        await c.post(f"/api/shared-groups/{a.id}/opt-in", headers=_as(cat))
    ).status_code == 200
    shared = (await c.get("/api/shared-groups", headers=_as(cat))).json()["groups"]
    state = {g["id"]: (g["opted_in"], g["hidden_by"]) for g in shared}
    assert state == {a.id: (True, None), b.id: (False, "shared")}
    assert (
        await c.post(f"/api/shared-groups/{b.id}/opt-in", headers=_as(cat))
    ).status_code == 409
    # Opting in again to the same one is fine.
    assert (
        await c.post(f"/api/shared-groups/{a.id}/opt-in", headers=_as(cat))
    ).status_code == 200


async def test_unsharing_through_the_api(dash_client: DashClient) -> None:
    c, db = dash_client
    ann, bob = await _people(db)
    g = await db.create_user_model_group(ann, "coder", shared=True)
    assert g.id is not None
    await db.subscribe_user_model_group(bob, g.id)
    resp = await c.patch(
        f"/api/my-groups/{g.id}", json={"shared": False}, headers=_as(ann)
    )
    assert resp.status_code == 200
    assert resp.json()["shared"] is False
    assert await db.list_user_model_group_subscriptions() == []
    assert (await c.get("/api/shared-groups", headers=_as(bob))).json()["groups"] == []
    # Leaving "shared" out keeps it as it is.
    await c.patch(f"/api/my-groups/{g.id}", json={"shared": True}, headers=_as(ann))
    await c.patch(f"/api/my-groups/{g.id}", json={"name": "code"}, headers=_as(ann))
    group = await db.get_user_model_group(ann, g.id)
    assert group is not None
    assert group.shared is True


async def test_concurrent_opt_in_is_not_an_error(
    db: Database, monkeypatch: pytest.MonkeyPatch
) -> None:
    ann, bob = await _people(db)
    g = await db.create_user_model_group(ann, "coder", shared=True)
    assert g.id is not None
    real = db._subscribe

    async def racing(user_id: int, group_id: int) -> bool:
        # Another request opts in between this one's check and its insert.
        await real(user_id, group_id)
        raise IntegrityError("INSERT", {}, Exception("duplicate"))

    monkeypatch.setattr(db, "_subscribe", racing)
    assert await db.subscribe_user_model_group(bob, g.id)
    assert await db.list_user_model_group_subscriptions(bob) == [(bob, g.id)]
