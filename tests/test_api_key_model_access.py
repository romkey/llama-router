"""API keys limited to chosen models and model groups."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import httpx
import pytest
from fastapi import FastAPI
from pytest_httpx import HTTPXMock
from starlette.requests import Request

from llama_router import proxy
from llama_router.api import deps as api_deps
from llama_router.api.app import app as ollama_app
from llama_router.auth import (
    generate_api_key,
    key_prefix,
    routing_preferences_from_request,
)
from llama_router.dashboard import deps as dash_deps
from llama_router.dashboard.auth_core import AUTH_COOKIE, hash_password, sign_session
from llama_router.dashboard.middleware import invalidate_dashboard_user_count_cache
from llama_router.database import Database
from llama_router.llamacpp_api import deps as v1_deps
from llama_router.llamacpp_api.app import app as v1_app
from llama_router.models import (
    BenchmarkResult,
    Provider,
    ProviderModel,
    ProviderProtocol,
    ProviderStatus,
)
from llama_router.provider_manager import ProviderManager
from llama_router.provider_secrets import key_hash, session_signing_key
from llama_router.router import Router, RoutingPreferences
from llama_router.routing_table import ModelAccess

CHAT = "/v1/chat/completions"
FAST = "http://fast.test:8080"
SLOW = "http://slow.test:8080"

DashClient = tuple[httpx.AsyncClient, Database]


@pytest.fixture
async def pm(db: Database) -> AsyncIterator[ProviderManager]:
    manager = ProviderManager(db)
    yield manager
    await manager.stop()


async def _provider(
    db: Database,
    pm: ProviderManager,
    name: str,
    url: str,
    models: list[str],
    startup: float = 100.0,
) -> Provider:
    added = await db.add_provider(name, ProviderProtocol.OPENAI)
    pid = added.require_id()
    await db.add_address(pid, url, is_preferred=True)
    await db.update_provider_status(pid, ProviderStatus.IDLE)
    await db.set_provider_models(
        pid, [ProviderModel(provider_id=pid, name=m) for m in models]
    )
    for m in models:
        await db.save_benchmark(
            BenchmarkResult(
                provider_id=pid,
                model_name=m,
                protocol="openai",
                startup_time_ms=startup,
                tokens_per_second=50.0,
            )
        )
    p = await db.get_provider(pid)
    assert p is not None
    await pm._rebuild_clients(p)
    return p


async def _group(db: Database, name: str, *members: str) -> int:
    g = await db.create_model_group(name)
    assert g.id is not None
    for m in members:
        await db.add_model_group_member(g.id, m)
    return g.id


def _prefs(models: tuple[str, ...] = (), groups: tuple[str, ...] = ()) -> Any:
    return RoutingPreferences(
        access=ModelAccess(models=frozenset(models), groups=frozenset(groups))
    )


async def _key(db: Database, owner: int | None = None) -> tuple[int, str]:
    plain = generate_api_key()
    key_id = await db.create_api_key(
        key_prefix(plain), key_hash(plain), "balanced", True, owner_user_id=owner
    )
    return key_id, plain


def _key_request(key: str) -> Request:
    return Request(
        {
            "type": "http",
            "method": "POST",
            "path": "/api/chat",
            "headers": [(b"api-key", key.encode())],
        }
    )


# --- Routing ---


@pytest.fixture
async def rt(db: Database, pm: ProviderManager) -> Router:
    await _provider(db, pm, "fast", FAST, ["small-coder", "chat"])
    await _provider(db, pm, "slow", SLOW, ["big-coder"], startup=5000.0)
    await _group(db, "coding", "big-coder", "small-coder")
    await db.set_model_alias("vendor/chat", "chat")
    router = Router(db, pm)
    await router.rebuild()
    return router


async def test_unrestricted_key_routes_anything(rt: Router) -> None:
    for model in ("chat", "coding", "small-coder", "vendor/chat"):
        assert await rt.route(model, CHAT, RoutingPreferences()) is not None


async def test_allowed_model_routes_and_others_do_not(rt: Router) -> None:
    prefs = _prefs(models=("chat",))
    assert await rt.route("chat", CHAT, prefs) is not None
    assert await rt.route("small-coder", CHAT, prefs) is None
    assert await rt.route("coding", CHAT, prefs) is None


async def test_model_is_allowed_under_any_alias(rt: Router) -> None:
    by_canonical = _prefs(models=("chat",))
    r = await rt.route("vendor/chat", CHAT, by_canonical)
    assert r is not None
    assert r.backend_model == "chat"
    by_alias = _prefs(models=("vendor/chat",))
    assert await rt.route("chat", CHAT, by_alias) is not None


async def test_group_allows_its_name_not_its_members(rt: Router) -> None:
    prefs = _prefs(groups=("coding",))
    r = await rt.route("coding", CHAT, prefs)
    assert r is not None
    assert r.group == "coding"
    assert await rt.route("small-coder", CHAT, prefs) is None
    assert await rt.route("big-coder", CHAT, prefs) is None


async def test_allowing_a_model_does_not_allow_a_group_of_it(rt: Router) -> None:
    prefs = _prefs(models=("small-coder", "big-coder"))
    assert await rt.route("coding", CHAT, prefs) is None


async def test_fallback_skips_models_the_key_may_not_use(
    db: Database, rt: Router
) -> None:
    await db.set_model_fallback("gone", "chat")
    await rt.rebuild()
    assert await rt.route("gone", CHAT, _prefs(models=("gone", "chat"))) is not None
    assert await rt.route("gone", CHAT, _prefs(models=("gone",))) is None


async def test_restricted_key_with_nothing_allowed_routes_nowhere(rt: Router) -> None:
    for model in ("chat", "coding", "small-coder"):
        assert await rt.route(model, CHAT, _prefs()) is None


async def test_chaos_mode_picks_only_allowed_models(rt: Router) -> None:
    prefs = RoutingPreferences(
        mode="chaos", access=ModelAccess(models=frozenset({"chat"}))
    )
    for _ in range(20):
        r = await rt.route("chat", CHAT, prefs)
        assert r is not None
        assert r.model_key == "chat"


# --- Storage ---


async def test_access_is_stored_replaced_and_cleared(db: Database) -> None:
    key_id, _ = await _key(db)
    gid = await _group(db, "coding", "m")
    assert await db.get_api_key_access(key_id) == {
        "restricted": False,
        "models": [],
        "groups": [],
    }
    assert await db.set_api_key_access(key_id, True, ["b", "a", "a"], [gid])
    assert await db.get_api_key_access(key_id) == {
        "restricted": True,
        "models": ["a", "b"],
        "groups": [{"id": gid, "name": "coding"}],
    }
    [row] = await db.list_api_keys()
    assert (row["restrict_models"], row["allowed_model_count"]) == (True, 2)
    assert row["allowed_group_count"] == 1

    await db.set_api_key_access(key_id, True, ["c"], [])
    access = await db.get_api_key_access(key_id)
    assert access is not None
    assert (access["models"], access["groups"]) == (["c"], [])

    # Unrestricted keeps no list.
    await db.set_api_key_access(key_id, False, ["ignored"], [gid])
    assert await db.get_api_key_access(key_id) == {
        "restricted": False,
        "models": [],
        "groups": [],
    }
    assert not await db.set_api_key_access(key_id + 99, True, [], [])
    assert await db.get_api_key_access(key_id + 99) is None


async def test_deleting_a_group_narrows_never_widens(db: Database) -> None:
    key_id, _ = await _key(db)
    gid = await _group(db, "coding", "m")
    await db.set_api_key_access(key_id, True, [], [gid])
    await db.delete_model_group("coding")
    # Still restricted, now to nothing.
    assert await db.get_api_key_access(key_id) == {
        "restricted": True,
        "models": [],
        "groups": [],
    }


async def test_deleting_a_key_deletes_its_list(db: Database) -> None:
    key_id, _ = await _key(db)
    await db.set_api_key_access(key_id, True, ["m"], [])
    await db.delete_api_key(key_id)
    rows = await db._select_all("SELECT * FROM api_key_allowed_models")
    assert rows == []


# --- Resolving a key ---


async def test_key_carries_its_access_and_follows_group_renames(
    db: Database,
) -> None:
    await db.set_allow_unauthenticated(False)
    key_id, plain = await _key(db)
    prefs = await routing_preferences_from_request(db, _key_request(plain))
    assert prefs is not None
    assert prefs.access is None

    gid = await _group(db, "coding", "m")
    await db.set_api_key_access(key_id, True, ["chat"], [gid])
    prefs = await routing_preferences_from_request(db, _key_request(plain))
    assert prefs is not None
    assert prefs.access == ModelAccess(
        models=frozenset({"chat"}), groups=frozenset({"coding"})
    )

    await db.update_model_group("coding", new_name="code")
    prefs = await routing_preferences_from_request(db, _key_request(plain))
    assert prefs is not None
    assert prefs.access is not None
    assert prefs.access.groups == frozenset({"code"})


# --- The APIs ---


@pytest.fixture
async def api(db: Database, pm: ProviderManager) -> tuple[Router, str]:
    """A router with a key limited to the group ``coding`` and model ``chat``."""
    await db.set_allow_unauthenticated(False)
    await _provider(db, pm, "fast", FAST, ["small-coder", "chat"])
    await _provider(db, pm, "slow", SLOW, ["big-coder", "other"], startup=5000.0)
    gid = await _group(db, "coding", "big-coder", "small-coder")
    key_id, plain = await _key(db)
    await db.set_api_key_access(key_id, True, ["chat"], [gid])
    rt = Router(db, pm)
    await rt.rebuild()
    api_deps.init(db, pm, rt)
    v1_deps.init(db, pm, rt)
    return rt, plain


def _client(app: FastAPI) -> httpx.AsyncClient:
    transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
    return httpx.AsyncClient(transport=transport, base_url="http://router")


async def test_listings_show_only_what_the_key_may_use(
    api: tuple[Router, str],
) -> None:
    _, key = api
    headers = {"API-KEY": key}
    async with _client(ollama_app) as c:
        tags = (await c.get("/api/tags", headers=headers)).json()["models"]
    assert [m["name"] for m in tags] == ["chat", "coding"]
    async with _client(v1_app) as c:
        listing = (await c.get("/v1/models", headers=headers)).json()["data"]
        hidden = await c.get("/v1/models/other", headers=headers)
        shown = await c.get("/v1/models/coding", headers=headers)
    assert [m["id"] for m in listing] == ["chat", "coding"]
    assert hidden.status_code == 404
    assert shown.status_code == 200


async def test_request_for_a_disallowed_model_is_403(
    api: tuple[Router, str], httpx_mock: HTTPXMock
) -> None:
    _, key = api
    async with _client(v1_app) as c:
        for model in ("other", "small-coder"):
            resp = await c.post(
                CHAT,
                json={"model": model, "messages": [], "stream": False},
                headers={"API-KEY": key},
            )
            assert resp.status_code == 403
            assert "may not use model" in resp.json()["detail"]
    assert httpx_mock.get_requests() == []


async def test_request_for_an_allowed_group_is_served(
    api: tuple[Router, str], httpx_mock: HTTPXMock
) -> None:
    _, key = api
    httpx_mock.add_response(url=FAST + CHAT, json={"choices": []})
    async with _client(v1_app) as c:
        resp = await c.post(
            CHAT,
            json={"model": "coding", "messages": [], "stream": False},
            headers={"API-KEY": key},
        )
    assert resp.status_code == 200


async def test_decision_policy_choice_outside_the_list_is_ignored(
    api: tuple[Router, str],
    httpx_mock: HTTPXMock,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, key = api

    async def choose_other(*_args: Any, **_kw: Any) -> str:
        return "other"

    monkeypatch.setattr(proxy, "apply_decision_policy", choose_other)
    httpx_mock.add_response(url=FAST + CHAT, json={"choices": []})
    async with _client(v1_app) as c:
        resp = await c.post(
            CHAT,
            json={"model": "chat", "messages": [], "stream": False},
            headers={"API-KEY": key},
        )
    assert resp.status_code == 200
    sent = httpx_mock.get_requests()[0]
    assert b'"model":"chat"' in sent.content.replace(b" ", b"")


# --- The dashboard API ---


def _as(user_id: int) -> dict[str, str]:
    token = sign_session(user_id, session_signing_key())
    return {"Cookie": f"{AUTH_COOKIE}={token}"}


async def _users(db: Database) -> tuple[int, int, int]:
    admin = await db.create_dashboard_user("adm", hash_password("pw-123456789"), True)
    ann = await db.create_dashboard_user("ann", hash_password("pw-123456789"), False)
    bob = await db.create_dashboard_user("bob", hash_password("pw-123456789"), False)
    invalidate_dashboard_user_count_cache()
    return admin, ann, bob


async def test_admin_sets_access_on_create_and_edit(dash_client: DashClient) -> None:
    c, db = dash_client
    admin, _, _ = await _users(db)
    gid = await _group(db, "coding", "m")

    resp = await c.post(
        "/api/keys/generate",
        json={"allowed": {"models": ["chat"], "groups": [gid]}},
        headers=_as(admin),
    )
    assert resp.status_code == 200, resp.text
    key_id = resp.json()["id"]
    got = (await c.get(f"/api/keys/{key_id}", headers=_as(admin))).json()
    assert got["allowed"] == {
        "models": ["chat"],
        "groups": [{"id": gid, "name": "coding"}],
    }

    resp = await c.patch(
        f"/api/keys/{key_id}",
        json={"allowed": {"models": [], "groups": []}},
        headers=_as(admin),
    )
    assert resp.json()["allowed"] == {"models": [], "groups": []}
    assert resp.json()["restrict_models"] is True

    # Other edits leave the list alone.
    resp = await c.patch(f"/api/keys/{key_id}", json={"label": "x"}, headers=_as(admin))
    assert resp.json()["allowed"] == {"models": [], "groups": []}

    resp = await c.patch(
        f"/api/keys/{key_id}", json={"allowed": None}, headers=_as(admin)
    )
    assert resp.json()["allowed"] is None
    assert resp.json()["restrict_models"] is False

    # A key created without a list is unrestricted.
    resp = await c.post("/api/keys/generate", json={}, headers=_as(admin))
    other = (await c.get(f"/api/keys/{resp.json()['id']}", headers=_as(admin))).json()
    assert other["allowed"] is None


@pytest.mark.parametrize(
    "allowed",
    [
        "chat",
        {"models": "chat"},
        {"models": [""]},
        {"models": ["x" * 256]},
        {"models": ["bad\nname"]},
        {"models": [3]},
        {"groups": [999]},
        {"groups": ["coding"]},
        {"groups": [True]},
        {"models": [f"m{i}" for i in range(501)]},
    ],
)
async def test_bad_access_is_400(dash_client: DashClient, allowed: object) -> None:
    c, db = dash_client
    admin, _, _ = await _users(db)
    key_id, _ = await _key(db)
    resp = await c.patch(
        f"/api/keys/{key_id}", json={"allowed": allowed}, headers=_as(admin)
    )
    assert resp.status_code == 400
    resp = await c.post(
        "/api/keys/generate", json={"allowed": allowed}, headers=_as(admin)
    )
    assert resp.status_code == 400
    assert len(await db.list_api_keys()) == 1


async def test_users_limit_their_own_keys(dash_client: DashClient) -> None:
    c, db = dash_client
    _, ann, bob = await _users(db)
    gid = await _group(db, "coding", "m")

    resp = await c.post(
        "/api/my-keys", json={"allowed": {"groups": [gid]}}, headers=_as(ann)
    )
    assert resp.status_code == 200, resp.text
    key_id = resp.json()["id"]
    mine = (await c.get(f"/api/my-keys/{key_id}", headers=_as(ann))).json()
    assert mine["allowed"] == {
        "models": [],
        "groups": [{"id": gid, "name": "coding"}],
    }

    resp = await c.patch(
        f"/api/my-keys/{key_id}",
        json={"allowed": {"models": ["chat"]}},
        headers=_as(ann),
    )
    assert resp.status_code == 200
    assert resp.json()["allowed"] == {"models": ["chat"], "groups": []}

    resp = await c.patch(
        f"/api/my-keys/{key_id}", json={"allowed": None}, headers=_as(bob)
    )
    assert resp.status_code == 404
    access = await db.get_api_key_access(key_id)
    assert access is not None
    assert access["restricted"] is True


async def test_allowable_lists_groups_and_models(
    dash_client: DashClient,
) -> None:
    c, db = dash_client
    admin, ann, _ = await _users(db)
    p = await db.add_provider("p")
    pid = p.require_id()
    await db.set_provider_models(
        pid,
        [ProviderModel(provider_id=pid, name=n) for n in ("b", "a", "coding")],
    )
    gid = await _group(db, "coding", "a")
    for user in (ann, admin):
        resp = await c.get("/api/my-keys/allowable", headers=_as(user))
        assert resp.status_code == 200
        assert resp.json() == {
            "groups": [{"id": gid, "name": "coding", "members": ["a"]}],
            # A model hidden by a group of its name isn't offered.
            "models": ["a", "b"],
        }


async def test_pages_have_the_picker(dash_client: DashClient) -> None:
    c, db = dash_client
    admin, ann, _ = await _users(db)
    key_id, _ = await _key(db)
    await db.set_api_key_access(key_id, True, ["a", "b"], [])
    page = await c.get("/", headers=_as(admin))
    assert "new-key-access-picker" in page.text
    assert "edit-key-access-picker" in page.text
    assert "2 models" in page.text
    assert "member models by their own names" in page.text
    page = await c.get("/my-keys", headers=_as(ann))
    assert "my-key-access-picker" in page.text
    assert "my-edit-access-picker" in page.text


async def test_route_explorer_applies_the_key_list(rt: Router, db: Database) -> None:
    await db.set_model_fallback("gone", "small-coder")
    await db.set_model_fallback("small-coder", "chat")
    await rt.rebuild()
    prefs = _prefs(models=("gone", "chat"))
    refused = await rt.explain("coding", CHAT, prefs)
    assert refused["chosen"] is None
    assert "may not use 'coding'" in refused["summary"]
    assert refused["request"]["allowed"] == {"models": ["chat", "gone"], "groups": []}

    out = await rt.explain("gone", CHAT, prefs)
    assert out["chosen"]["model"] == "chat"
    skipped = [s["model"] for s in out["steps"] if s["outcome"].startswith("skipped")]
    assert skipped == ["small-coder"]

    chaos = await rt.explain(
        "chat",
        CHAT,
        RoutingPreferences(
            mode="chaos", access=ModelAccess(models=frozenset({"chat"}))
        ),
    )
    assert [p["models"] for p in chaos["chaos_pool"]] == [["chat"]]


async def test_route_explorer_uses_a_limited_keys_list(dash_client: DashClient) -> None:
    c, db = dash_client
    admin, _, _ = await _users(db)
    key_id, _ = await _key(db)
    await db.set_api_key_access(key_id, True, ["chat"], [])
    pm = dash_deps.get_pm()
    pm.attach_router(Router(db, pm))
    resp = await c.get(
        "/api/routing/explain",
        params={"model": "other", "endpoint": CHAT, "key_id": key_id},
        headers=_as(admin),
    )
    assert resp.status_code == 200, resp.text
    assert "may not use 'other'" in resp.json()["summary"]


async def test_chaos_mode_serves_members_of_an_allowed_group(rt: Router) -> None:
    prefs = RoutingPreferences(
        mode="chaos", access=ModelAccess(groups=frozenset({"coding"}))
    )
    for _ in range(20):
        r = await rt.route("coding", CHAT, prefs)
        assert r is not None
        assert r.model_key in ("small-coder", "big-coder")
    out = await rt.explain("coding", CHAT, prefs)
    pool = sorted(m for p in out["chaos_pool"] for m in p["models"])
    assert pool == ["big-coder", "small-coder"]


async def test_chaos_mode_keeps_a_provider_scoped_member_on_its_provider(
    db: Database, pm: ProviderManager
) -> None:
    fast = await _provider(db, pm, "fast", FAST, ["coder"])
    await _provider(db, pm, "slow", SLOW, ["coder"])
    g = await db.create_model_group("coding")
    assert g.id is not None
    await db.add_model_group_member(g.id, "coder", fast.require_id())
    rt = Router(db, pm)
    await rt.rebuild()
    prefs = RoutingPreferences(
        mode="chaos", access=ModelAccess(groups=frozenset({"coding"}))
    )
    for _ in range(20):
        r = await rt.route("coding", CHAT, prefs)
        assert r is not None
        assert r.provider.id == fast.id
