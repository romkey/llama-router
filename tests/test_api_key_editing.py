"""Editing API keys and regenerating their secrets."""

from __future__ import annotations

import httpx
import pytest
from starlette.requests import Request

from llama_router.auth import (
    generate_api_key,
    key_prefix,
    routing_preferences_from_request,
)
from llama_router.dashboard.auth_core import AUTH_COOKIE, hash_password, sign_session
from llama_router.dashboard.middleware import invalidate_dashboard_user_count_cache
from llama_router.database import Database
from llama_router.provider_secrets import key_hash, session_signing_key

DashClient = tuple[httpx.AsyncClient, Database]


def _as(user_id: int) -> dict[str, str]:
    token = sign_session(user_id, session_signing_key())
    return {"Cookie": f"{AUTH_COOKIE}={token}"}


async def _users(db: Database) -> tuple[int, int, int]:
    admin = await db.create_dashboard_user("adm", hash_password("pw-123456789"), True)
    ann = await db.create_dashboard_user("ann", hash_password("pw-123456789"), False)
    bob = await db.create_dashboard_user("bob", hash_password("pw-123456789"), False)
    invalidate_dashboard_user_count_cache()
    return admin, ann, bob


async def _key(db: Database, owner: int | None = None) -> tuple[int, str]:
    plain = generate_api_key()
    key_id = await db.create_api_key(
        key_prefix(plain),
        key_hash(plain),
        "balanced",
        True,
        locality="local",
        daily_spend_cap_usd=1.0,
        owner_user_id=owner,
        label="old",
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


async def test_update_changes_only_given_fields(db: Database) -> None:
    key_id, _ = await _key(db)
    assert await db.update_api_key(key_id, {"routing_mode": "latency"})
    key = await db.get_api_key(key_id)
    assert key is not None
    assert (key["routing_mode"], key["label"], key["locality"]) == (
        "latency",
        "old",
        "local",
    )
    # Setting a value to what it already is still finds the key (MySQL
    # reports no changed rows).
    assert await db.update_api_key(key_id, {"routing_mode": "latency"})
    assert await db.update_api_key(key_id, {})
    assert not await db.update_api_key(key_id + 99, {"label": "x"})
    assert not await db.update_api_key(key_id + 99, {})
    with pytest.raises(ValueError, match="not editable"):
        await db.update_api_key(key_id, {"key_hash": "x"})


async def test_update_with_owner_only_matches_own_key(db: Database) -> None:
    _, ann, bob = await _users(db)
    key_id, _ = await _key(db, owner=ann)
    assert not await db.update_api_key(key_id, {"label": "x"}, owner_user_id=bob)
    assert await db.get_api_key(key_id, owner_user_id=bob) is None
    assert await db.update_api_key(key_id, {"label": "x"}, owner_user_id=ann)
    key = await db.get_api_key(key_id)
    assert key is not None
    assert key["label"] == "x"


async def test_edit_takes_effect_on_next_request(db: Database) -> None:
    await db.set_allow_unauthenticated(False)
    key_id, plain = await _key(db)
    prefs = await routing_preferences_from_request(db, _key_request(plain))
    assert prefs is not None
    assert prefs.mode == "balanced"
    await db.update_api_key(key_id, {"routing_mode": "throughput"})
    prefs = await routing_preferences_from_request(db, _key_request(plain))
    assert prefs is not None
    assert prefs.mode == "throughput"


async def test_admin_edits_any_key(dash_client: DashClient) -> None:
    c, db = dash_client
    admin, ann, _ = await _users(db)
    key_id, _ = await _key(db, owner=ann)

    got = await c.get(f"/api/keys/{key_id}", headers=_as(admin))
    assert got.status_code == 200
    assert got.json()["label"] == "old"
    assert "key_hash" not in got.json()

    resp = await c.patch(
        f"/api/keys/{key_id}",
        json={
            "label": "  new  ",
            "routing_mode": "cheapest",
            "allow_fallback": False,
            "session_affinity": "on",
            "locality": "any",
            "daily_spend_cap_usd": "",
        },
        headers=_as(admin),
    )
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["label"] == "new"
    assert data["routing_mode"] == "cheapest"
    assert data["allow_fallback"] is False
    assert data["session_affinity"] is True
    assert data["locality"] is None
    assert data["daily_spend_cap_usd"] is None
    assert data["owner_user_id"] == ann


@pytest.mark.parametrize(
    "body",
    [
        {"routing_mode": "fastest"},
        {"allow_fallback": "yes"},
        {"locality": "moon"},
        {"daily_spend_cap_usd": -1},
        {"label": "x" * 101},
        {"session_affinity": "maybe"},
    ],
)
async def test_admin_edit_rejects_bad_values(
    dash_client: DashClient, body: dict[str, object]
) -> None:
    c, db = dash_client
    admin, _, _ = await _users(db)
    key_id, _ = await _key(db)
    resp = await c.patch(f"/api/keys/{key_id}", json=body, headers=_as(admin))
    assert resp.status_code == 400


async def test_admin_edit_unknown_key_is_404(dash_client: DashClient) -> None:
    c, db = dash_client
    admin, _, _ = await _users(db)
    assert (await c.get("/api/keys/999", headers=_as(admin))).status_code == 404
    resp = await c.patch("/api/keys/999", json={"label": "x"}, headers=_as(admin))
    assert resp.status_code == 404
    resp = await c.post("/api/keys/999/regenerate", headers=_as(admin))
    assert resp.status_code == 404


async def test_viewer_cannot_use_admin_edit(dash_client: DashClient) -> None:
    c, db = dash_client
    _, ann, _ = await _users(db)
    key_id, _ = await _key(db, owner=ann)
    resp = await c.patch(f"/api/keys/{key_id}", json={"label": "x"}, headers=_as(ann))
    assert resp.status_code == 403
    resp = await c.post(f"/api/keys/{key_id}/regenerate", headers=_as(ann))
    assert resp.status_code == 403


async def test_self_service_route_still_reachable(dash_client: DashClient) -> None:
    """/api/keys/{id} must not swallow /api/keys/self-service or /usage."""
    c, db = dash_client
    admin, _, _ = await _users(db)
    assert (await c.get("/api/keys/self-service", headers=_as(admin))).is_success
    assert (await c.get("/api/keys/usage", headers=_as(admin))).is_success


async def test_owner_edits_own_key(dash_client: DashClient) -> None:
    c, db = dash_client
    _, ann, bob = await _users(db)
    key_id, _ = await _key(db, owner=ann)

    assert (await c.get(f"/api/my-keys/{key_id}", headers=_as(ann))).is_success
    resp = await c.patch(
        f"/api/my-keys/{key_id}",
        json={"label": "laptop", "routing_mode": "latency", "allow_fallback": False},
        headers=_as(ann),
    )
    assert resp.status_code == 200, resp.text
    assert (resp.json()["label"], resp.json()["routing_mode"]) == ("laptop", "latency")

    # Not someone else's.
    assert (await c.get(f"/api/my-keys/{key_id}", headers=_as(bob))).status_code == 404
    resp = await c.patch(
        f"/api/my-keys/{key_id}", json={"label": "mine"}, headers=_as(bob)
    )
    assert resp.status_code == 404


@pytest.mark.parametrize("body", [{"locality": "any"}, {"daily_spend_cap_usd": None}])
async def test_owner_cannot_change_admin_limits(
    dash_client: DashClient, body: dict[str, object]
) -> None:
    c, db = dash_client
    _, ann, _ = await _users(db)
    key_id, _ = await _key(db, owner=ann)
    resp = await c.patch(f"/api/my-keys/{key_id}", json=body, headers=_as(ann))
    assert resp.status_code == 400
    key = await db.get_api_key(key_id)
    assert key is not None
    assert (key["locality"], key["daily_spend_cap_usd"]) == ("local", 1.0)


async def test_owner_cannot_edit_admin_issued_key(dash_client: DashClient) -> None:
    c, db = dash_client
    admin, _, _ = await _users(db)
    key_id, _ = await _key(db)
    # Not even an admin goes through /api/my-keys for a key they don't own.
    resp = await c.patch(
        f"/api/my-keys/{key_id}", json={"label": "x"}, headers=_as(admin)
    )
    assert resp.status_code == 404


async def test_regenerate_replaces_secret_keeps_settings(
    dash_client: DashClient,
) -> None:
    c, db = dash_client
    admin, ann, bob = await _users(db)
    await db.set_allow_unauthenticated(False)
    key_id, old = await _key(db, owner=ann)
    provider_id = (await db.add_provider("p")).require_id()
    await db.set_api_key_model_pin(key_id, "m", provider_id)
    # Resolve once so the old secret sits in the auth cache.
    assert await routing_preferences_from_request(db, _key_request(old))

    resp = await c.post(f"/api/my-keys/{key_id}/regenerate", headers=_as(bob))
    assert resp.status_code == 404

    resp = await c.post(f"/api/my-keys/{key_id}/regenerate", headers=_as(ann))
    assert resp.status_code == 200, resp.text
    new = resp.json()["api_key"]
    assert new != old
    assert resp.json()["id"] == key_id
    assert resp.json()["key_prefix"] == key_prefix(new)

    assert await db.lookup_api_key(key_hash(old)) is None
    found = await db.lookup_api_key(key_hash(new))
    assert found is not None
    assert found["id"] == key_id
    key = await db.get_api_key(key_id)
    assert key is not None
    assert (key["label"], key["locality"], key["pin_count"]) == ("old", "local", 1)

    with pytest.raises(Exception, match="Invalid API-KEY"):
        await routing_preferences_from_request(db, _key_request(old))
    prefs = await routing_preferences_from_request(db, _key_request(new))
    assert prefs is not None
    assert prefs.key_id == key_id

    # Admins regenerate any key.
    resp = await c.post(f"/api/keys/{key_id}/regenerate", headers=_as(admin))
    assert resp.status_code == 200
    assert await db.lookup_api_key(key_hash(new)) is None


async def test_deprecated_endpoints_still_work(dash_client: DashClient) -> None:
    c, db = dash_client
    admin, _, _ = await _users(db)
    key_id, _ = await _key(db)
    resp = await c.post(
        f"/api/keys/{key_id}/spend-cap",
        json={"daily_spend_cap_usd": 3},
        headers=_as(admin),
    )
    assert resp.status_code == 200
    resp = await c.post(
        f"/api/keys/{key_id}/locality", json={"locality": "cloud"}, headers=_as(admin)
    )
    assert resp.status_code == 200
    key = await db.get_api_key(key_id)
    assert key is not None
    assert (key["daily_spend_cap_usd"], key["locality"]) == (3.0, "cloud")


async def test_pages_have_edit_controls(dash_client: DashClient) -> None:
    c, db = dash_client
    admin, ann, _ = await _users(db)
    await _key(db)
    page = await c.get("/my-keys", headers=_as(ann))
    assert "myEditKeyModal" in page.text
    page = await c.get("/", headers=_as(admin))
    assert "apiKeyEditModal" in page.text
    assert "regenerateApiKey(" in page.text
