"""Strict pins and model pins on the keys dashboard users issue themselves."""

from __future__ import annotations

from typing import Any

import httpx
import pytest

from llama_router.config import settings
from llama_router.dashboard.auth_core import AUTH_COOKIE, hash_password, sign_session
from llama_router.dashboard.middleware import invalidate_dashboard_user_count_cache
from llama_router.database import Database
from llama_router.models import Provider, ProviderModel, ProviderProtocol
from llama_router.provider_secrets import session_signing_key

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


async def _provider(
    db: Database, name: str, locality: str, models: tuple[str, ...] = ("m",)
) -> Provider:
    p = await db.add_provider(name, ProviderProtocol.OPENAI)
    await db.update_provider_settings(p.require_id(), locality=locality)
    await db.set_provider_models(
        p.require_id(),
        [ProviderModel(provider_id=p.require_id(), name=m) for m in models],
    )
    return p


async def test_owner_turns_strict_pins_on_and_off(dash_client: DashClient) -> None:
    c, db = dash_client
    _, ann, bob = await _users(db)
    key = (
        await c.post("/api/my-keys", json={"strict_pins": True}, headers=_as(ann))
    ).json()
    assert key["strict_pins"] is True
    assert (await db.get_api_key(key["id"]) or {})["strict_pins"] is True

    resp = await c.patch(
        f"/api/my-keys/{key['id']}", json={"strict_pins": False}, headers=_as(ann)
    )
    assert resp.status_code == 200, resp.text
    assert resp.json()["strict_pins"] is False
    listed = (await c.get("/api/my-keys", headers=_as(ann))).json()["keys"]
    assert listed[0]["strict_pins"] is False

    resp = await c.patch(
        f"/api/my-keys/{key['id']}", json={"strict_pins": "yes"}, headers=_as(ann)
    )
    assert resp.status_code == 400
    resp = await c.patch(
        f"/api/my-keys/{key['id']}", json={"strict_pins": True}, headers=_as(bob)
    )
    assert resp.status_code == 404
    # New keys default to lenient pins.
    other = (await c.post("/api/my-keys", json={}, headers=_as(ann))).json()
    assert other["strict_pins"] is False


async def test_admin_sets_strict_pins(dash_client: DashClient) -> None:
    c, db = dash_client
    admin, _, _ = await _users(db)
    key = (
        await c.post(
            "/api/keys/generate", json={"strict_pins": True}, headers=_as(admin)
        )
    ).json()
    assert key["strict_pins"] is True
    resp = await c.patch(
        f"/api/keys/{key['id']}", json={"strict_pins": False}, headers=_as(admin)
    )
    assert resp.status_code == 200, resp.text
    assert resp.json()["strict_pins"] is False
    assert (await db.get_api_key(key["id"]) or {})["strict_pins"] is False


async def test_owner_pin_crud_on_own_key(dash_client: DashClient) -> None:
    c, db = dash_client
    _, ann, bob = await _users(db)
    lan = await _provider(db, "lan", "local")
    key = (await c.post("/api/my-keys", json={}, headers=_as(ann))).json()
    base = f"/api/my-keys/{key['id']}/pins"

    resp = await c.post(
        base,
        json={"model_name": "m", "provider_id": lan.require_id()},
        headers=_as(ann),
    )
    assert resp.status_code == 200, resp.text
    pins = (await c.get(base, headers=_as(ann))).json()["pins"]
    assert pins == [
        {"model_name": "m", "provider_id": lan.require_id(), "provider_name": "lan"}
    ]
    listed = (await c.get("/api/my-keys", headers=_as(ann))).json()["keys"]
    assert listed[0]["pin_count"] == 1

    # Another user's key is not theirs to see or change.
    assert (await c.get(base, headers=_as(bob))).status_code == 404
    resp = await c.post(
        base,
        json={"model_name": "m", "provider_id": lan.require_id()},
        headers=_as(bob),
    )
    assert resp.status_code == 404
    assert (await c.delete(f"{base}/m", headers=_as(bob))).status_code == 404
    assert len(await db.list_api_key_model_pins(key["id"])) == 1

    assert (await c.delete(f"{base}/m", headers=_as(ann))).status_code == 200
    assert await db.list_api_key_model_pins(key["id"]) == []


async def test_owner_sees_and_removes_admin_pins(dash_client: DashClient) -> None:
    c, db = dash_client
    admin, ann, _ = await _users(db)
    lan = await _provider(db, "lan", "local", ("org/model:7b",))
    key = (await c.post("/api/my-keys", json={}, headers=_as(ann))).json()
    resp = await c.post(
        f"/api/keys/{key['id']}/pins",
        json={"model_name": "org/model:7b", "provider_id": lan.require_id()},
        headers=_as(admin),
    )
    assert resp.status_code == 200, resp.text
    base = f"/api/my-keys/{key['id']}/pins"
    pins = (await c.get(base, headers=_as(ann))).json()["pins"]
    assert [p["model_name"] for p in pins] == ["org/model:7b"]
    assert (await c.delete(f"{base}/org/model:7b", headers=_as(ann))).status_code == 200
    assert await db.list_api_key_model_pins(key["id"]) == []


async def test_owner_pin_validation(dash_client: DashClient) -> None:
    c, db = dash_client
    _, ann, _ = await _users(db)
    lan = await _provider(db, "lan", "local", ("m", "other"))
    cloud = await _provider(db, "cloud", "cloud")
    key = (
        await c.post(
            "/api/my-keys",
            json={"allowed": {"models": ["m"], "groups": []}},
            headers=_as(ann),
        )
    ).json()
    base = f"/api/my-keys/{key['id']}/pins"

    async def pin(body: dict[str, Any]) -> httpx.Response:
        return await c.post(base, json=body, headers=_as(ann))

    # Self-issued keys are local-only until an admin says otherwise.
    resp = await pin({"model_name": "m", "provider_id": cloud.require_id()})
    assert resp.status_code == 400
    assert "local" in resp.json()["detail"]
    resp = await pin({"model_name": "other", "provider_id": lan.require_id()})
    assert resp.status_code == 400
    assert "may not use" in resp.json()["detail"]
    assert (await pin({"model_name": "m", "provider_id": 9999})).status_code == 400
    assert (await pin({"model_name": "m"})).status_code == 400
    assert (await pin({"model_name": "m", "provider_id": "x"})).status_code == 400
    assert (
        await pin({"model_name": "", "provider_id": lan.require_id()})
    ).status_code == 400
    assert (await pin({"model_name": "m", "provider_id": True})).status_code == 400
    assert await db.list_api_key_model_pins(key["id"]) == []
    assert (
        await pin({"model_name": "m", "provider_id": lan.require_id()})
    ).status_code == 200


async def test_owner_pin_limit(
    dash_client: DashClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    c, db = dash_client
    _, ann, _ = await _users(db)
    lan = await _provider(db, "lan", "local", ("a", "b"))
    monkeypatch.setattr(settings, "max_pins_per_key", 1)
    key = (await c.post("/api/my-keys", json={}, headers=_as(ann))).json()
    base = f"/api/my-keys/{key['id']}/pins"
    first = {"model_name": "a", "provider_id": lan.require_id()}
    assert (await c.post(base, json=first, headers=_as(ann))).status_code == 200
    # Re-pinning the same model replaces it; a new one is over the limit.
    assert (await c.post(base, json=first, headers=_as(ann))).status_code == 200
    resp = await c.post(
        base,
        json={"model_name": "b", "provider_id": lan.require_id()},
        headers=_as(ann),
    )
    assert resp.status_code == 409


async def test_allowable_lists_providers_per_model_without_secrets(
    dash_client: DashClient,
) -> None:
    c, db = dash_client
    _, ann, _ = await _users(db)
    lan = await _provider(db, "lan", "local", ("m",))
    cloud = await _provider(db, "cloud", "cloud", ("m", "n"))
    await db.add_address(lan.require_id(), "http://10.0.0.5:11434")
    data = (await c.get("/api/my-keys/allowable", headers=_as(ann))).json()
    assert data["providers"]["m"] == [
        {"id": cloud.require_id(), "name": "cloud", "locality": "cloud"},
        {"id": lan.require_id(), "name": "lan", "locality": "local"},
    ]
    assert data["providers"]["n"] == [
        {"id": cloud.require_id(), "name": "cloud", "locality": "cloud"}
    ]
    assert "10.0.0.5" not in str(data)


async def test_explain_takes_strict_pins_from_key_or_query(
    dash_client: DashClient,
) -> None:
    from llama_router.dashboard import deps as dash_deps
    from llama_router.models import ProviderStatus
    from llama_router.router import Router

    c, db = dash_client
    admin, _, _ = await _users(db)
    pinned = await _provider(db, "pinned", "local")
    other = await _provider(db, "other", "local")
    await db.update_provider_status(pinned.require_id(), ProviderStatus.OFFLINE)
    await db.update_provider_status(other.require_id(), ProviderStatus.IDLE)
    Router(db, dash_deps.get_pm())
    key = (
        await c.post(
            "/api/keys/generate", json={"strict_pins": True}, headers=_as(admin)
        )
    ).json()
    await db.set_api_key_model_pin(key["id"], "m", pinned.require_id())

    url = f"/api/routing/explain?model=m&key_id={key['id']}"
    out = (await c.get(url, headers=_as(admin))).json()
    assert out["request"]["strict_pins"] is True
    assert out["chosen"] is None
    assert out["refused"]["status"] == 503

    out = (await c.get(url + "&strict_pins=false", headers=_as(admin))).json()
    assert out["request"]["strict_pins"] is False
    assert out["chosen"]["provider_id"] == other.require_id()
    bad = await c.get(url + "&strict_pins=maybe", headers=_as(admin))
    assert bad.status_code == 400
