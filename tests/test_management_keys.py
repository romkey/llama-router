"""Management keys: issuing them and what they may do on the dashboard API."""

from __future__ import annotations

import time
from typing import Any

import httpx
import pytest

from llama_router.config import settings
from llama_router.dashboard import management_keys as mk
from llama_router.dashboard.auth_core import AUTH_COOKIE, hash_password, sign_session
from llama_router.dashboard.middleware import invalidate_dashboard_user_count_cache
from llama_router.database import Database

DashClient = tuple[httpx.AsyncClient, Database]


def _cookie(user_id: int) -> dict[str, str]:
    token = sign_session(user_id, settings.session_secret.strip())
    return {"Cookie": f"{AUTH_COOKIE}={token}"}


def _bearer(key: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {key}"}


async def _admin(db: Database) -> dict[str, str]:
    uid = await db.create_dashboard_user("adm", hash_password("password-12345"), True)
    invalidate_dashboard_user_count_cache()
    return _cookie(uid)


async def _issue(
    client: httpx.AsyncClient,
    admin: dict[str, str],
    privileges: list[str],
    **extra: Any,
) -> str:
    r = await client.post(
        "/api/management-keys",
        json={"name": "bot", "privileges": privileges, **extra},
        headers=admin,
    )
    assert r.status_code == 200, r.text
    key = str(r.json()["management_key"])
    assert key.startswith("lrm_")
    return key


@pytest.mark.parametrize(
    ("method", "path", "needed"),
    [
        ("GET", "/api/keys", "keys:read"),
        ("POST", "/api/keys/generate", "keys:write"),
        ("DELETE", "/api/keys/3", "keys:write"),
        ("PUT", "/api/keys/3/decision-policy", "keys:write"),
        ("GET", "/api/keys/3/pins", "keys:read"),
        ("GET", "/api/model-aliases", "models:read"),
        ("POST", "/api/model-groups/1/members", "models:write"),
        ("POST", "/api/routing-table/rebuild", "models:write"),
        ("GET", "/api/routing/explain", "models:read"),
        ("GET", "/api/status", "status:read"),
        ("POST", "/api/status", None),
        ("GET", "/api/keysets", None),
        ("GET", "/api/management-keys", None),
        ("POST", "/api/management-keys", None),
        ("GET", "/users", None),
        ("GET", "/", None),
        ("POST", "/api/auth/allow-unauthenticated", None),
        ("GET", "/api/workers", None),
        ("OPTIONS", "/api/keys", None),
    ],
)
def test_required_privilege(method: str, path: str, needed: str | None) -> None:
    assert mk.required_privilege(method, path) == needed


def test_write_includes_read() -> None:
    assert mk.has_privilege(frozenset({"keys:write"}), "keys:read")
    assert not mk.has_privilege(frozenset({"keys:read"}), "keys:write")
    assert not mk.has_privilege(frozenset({"models:write"}), "keys:read")


async def test_admin_issues_lists_and_revokes(dash_client: DashClient) -> None:
    client, db = dash_client
    admin = await _admin(db)
    key = await _issue(client, admin, ["keys:write", "keys:read"])

    listed = (await client.get("/api/management-keys", headers=admin)).json()
    [row] = listed["keys"]
    assert row["name"] == "bot"
    assert row["privileges"] == ["keys:read", "keys:write"]
    assert row["created_by"] == "adm"
    assert key not in str(listed)
    assert "keys:write" in listed["privileges"]

    page = await client.get("/users", headers=admin)
    assert page.status_code == 200
    assert "Management keys" in page.text
    assert row["key_prefix"] in page.text

    r = await client.delete(f"/api/management-keys/{row['id']}", headers=admin)
    assert r.status_code == 200
    r = await client.get("/api/keys", headers=_bearer(key))
    assert r.status_code == 401


async def test_issue_validates_body(dash_client: DashClient) -> None:
    client, db = dash_client
    admin = await _admin(db)
    for body in (
        {"name": "", "privileges": ["keys:read"]},
        {"name": "x" * 65, "privileges": ["keys:read"]},
        {"name": "bot", "privileges": []},
        {"name": "bot", "privileges": ["root"]},
        {"name": "bot", "privileges": ["keys:read"], "expires_in_days": 0},
        {"name": "bot", "privileges": ["keys:read"], "expires_in_days": "soon"},
    ):
        r = await client.post("/api/management-keys", json=body, headers=admin)
        assert r.status_code == 400, body


async def test_viewer_cannot_issue(dash_client: DashClient) -> None:
    client, db = dash_client
    await _admin(db)
    viewer = await db.create_dashboard_user("view", hash_password("pw-12345678"), False)
    r = await client.post(
        "/api/management-keys",
        json={"name": "bot", "privileges": ["keys:read"]},
        headers=_cookie(viewer),
    )
    assert r.status_code == 403
    r = await client.get("/api/management-keys", headers=_cookie(viewer))
    assert r.status_code == 403


async def test_key_issues_and_revokes_api_keys(dash_client: DashClient) -> None:
    client, db = dash_client
    admin = await _admin(db)
    key = await _issue(client, admin, ["keys:write"])
    auth = _bearer(key)

    r = await client.post("/api/keys/generate", json={}, headers=auth)
    assert r.status_code == 200
    assert r.json()["api_key"].startswith("lrk_")

    r = await client.get("/api/keys", headers=auth)
    assert r.status_code == 200
    [api_key] = r.json()["keys"]

    r = await client.delete(f"/api/keys/{api_key['id']}", headers=auth)
    assert r.status_code == 200
    assert await db.list_api_keys() == []

    [row] = await db.list_management_keys()
    assert row["last_used_at"] is not None


async def test_read_only_key_cannot_write(dash_client: DashClient) -> None:
    client, db = dash_client
    admin = await _admin(db)
    auth = _bearer(await _issue(client, admin, ["keys:read"]))

    assert (await client.get("/api/keys", headers=auth)).status_code == 200
    r = await client.post("/api/keys/generate", json={}, headers=auth)
    assert r.status_code == 403
    assert "keys:write" in r.json()["detail"]
    assert await db.list_api_keys() == []


async def test_key_cannot_escape_its_privileges(dash_client: DashClient) -> None:
    client, db = dash_client
    admin = await _admin(db)
    auth = _bearer(
        await _issue(client, admin, sorted(mk.PRIVILEGES))  # every privilege
    )
    for method, path in (
        ("GET", "/"),
        ("GET", "/users"),
        ("POST", "/users/add"),
        ("GET", "/api/management-keys"),
        ("POST", "/api/management-keys"),
        ("POST", "/api/auth/allow-unauthenticated"),
        ("GET", "/api/workers"),
        ("GET", "/api/chat/conversations"),
    ):
        r = await client.request(method, path, headers=auth, json={})
        assert r.status_code == 403, (method, path)
        assert r.json()["detail"] == "Not available to management keys"
    assert len(await db.list_management_keys()) == 1


async def test_status_privilege(dash_client: DashClient) -> None:
    client, db = dash_client
    admin = await _admin(db)
    keys_only = _bearer(await _issue(client, admin, ["keys:read"]))
    status = _bearer(await _issue(client, admin, ["status:read"]))
    assert (await client.get("/api/status", headers=keys_only)).status_code == 403
    assert (await client.get("/api/status", headers=status)).status_code == 200


async def test_unknown_and_expired_keys_are_refused(dash_client: DashClient) -> None:
    client, db = dash_client
    admin = await _admin(db)
    r = await client.get("/api/keys", headers=_bearer("lrm_nope"))
    assert r.status_code == 401

    expired = mk.generate_management_key()
    await db.create_management_key(
        name="old",
        key_prefix="lrm_old",
        key_hash=mk.management_key_hash(expired),
        privileges=["keys:read"],
        created_by="adm",
        expires_at=int(time.time()) - 1,
    )
    r = await client.get("/api/keys", headers=_bearer(expired))
    assert r.status_code == 401

    page = await client.get("/users", headers=admin)
    assert "expired" in page.text


async def test_expiring_key_works_until_it_expires(dash_client: DashClient) -> None:
    client, db = dash_client
    admin = await _admin(db)
    key = await _issue(client, admin, ["keys:read"], expires_in_days=30)
    [row] = await db.list_management_keys()
    assert row["expires_at"] > time.time() + 29 * 86400
    assert (await client.get("/api/keys", headers=_bearer(key))).status_code == 200


async def test_other_bearer_tokens_use_the_session(dash_client: DashClient) -> None:
    """Only lrm_ tokens are management keys; others fall through to sign-in."""
    client, db = dash_client
    await _admin(db)
    r = await client.get("/api/keys", headers=_bearer("something-else"))
    assert r.status_code == 401
    assert r.json()["detail"] == "Not authenticated"
