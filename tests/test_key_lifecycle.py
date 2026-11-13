"""API key expiry, renewal, pausing and holds when an owner loses access."""

from __future__ import annotations

import contextlib
import sqlite3
import time
from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import httpx
import pytest
from alembic import command

from llama_router import proxy
from llama_router.auth import generate_api_key, key_prefix
from llama_router.config import settings
from llama_router.dashboard import oidc
from llama_router.dashboard.auth_core import AUTH_COOKIE, hash_password, sign_session
from llama_router.dashboard.middleware import invalidate_dashboard_user_count_cache
from llama_router.database import Database
from llama_router.key_lifecycle import (
    REASON_INACTIVE,
    key_status,
    sweep_idle_sso_users,
)
from llama_router.llamacpp_api import deps as v1_deps
from llama_router.llamacpp_api.app import app as v1_app
from llama_router.models import (
    BenchmarkResult,
    ProviderModel,
    ProviderProtocol,
    ProviderStatus,
)
from llama_router.provider_manager import ProviderManager
from llama_router.provider_secrets import key_hash, session_signing_key
from llama_router.router import Router
from tests.conftest import asgi_client
from tests.test_alembic_roundtrip import _cfg

CHAT = "/v1/chat/completions"
BODY = {"model": "m", "messages": [{"role": "user", "content": "hi"}]}
DashClient = tuple[httpx.AsyncClient, Database]


@pytest.fixture
async def v1(
    db: Database, monkeypatch: pytest.MonkeyPatch
) -> AsyncIterator[httpx.AsyncClient]:
    pm = ProviderManager(db)
    rt = Router(db, pm)
    v1_deps.init(db, pm, rt)

    class Client:
        async def chat_completions(self, body: dict[str, Any]) -> dict[str, Any]:
            return {"choices": [{"message": {"content": "ok"}}]}

    monkeypatch.setattr(proxy, "get_v1_client", lambda pm, pid: Client())
    monkeypatch.setattr(proxy, "log_request_background", lambda *a, **k: None)
    p = await db.add_provider("local", ProviderProtocol.OPENAI)
    pid = p.require_id()
    await db.update_provider_status(pid, ProviderStatus.IDLE)
    await db.set_provider_models(pid, [ProviderModel(provider_id=pid, name="m")])
    await db.save_benchmark(
        BenchmarkResult(
            provider_id=pid, model_name="m", protocol="openai", startup_time_ms=100.0
        )
    )
    async with asgi_client(v1_app) as client:
        yield client
    v1_deps._db, v1_deps._pm, v1_deps._router = None, None, None


async def _key(db: Database, owner: int | None = None) -> tuple[str, int]:
    plaintext = generate_api_key()
    key_id = await db.create_api_key(
        key_prefix(plaintext),
        key_hash(plaintext),
        "balanced",
        True,
        owner_user_id=owner,
    )
    return plaintext, key_id


async def _post(v1: httpx.AsyncClient, key: str) -> httpx.Response:
    return await v1.post(CHAT, json=BODY, headers={"Authorization": f"Bearer {key}"})


def _as(user_id: int) -> dict[str, str]:
    return {"Cookie": f"{AUTH_COOKIE}={sign_session(user_id, session_signing_key())}"}


async def _users(db: Database) -> tuple[int, int]:
    admin = await db.create_dashboard_user("adm", hash_password("pw-123456789"), True)
    ann = await db.create_dashboard_user("ann", hash_password("pw-123456789"), False)
    invalidate_dashboard_user_count_cache()
    return admin, ann


# --- expiry ---


async def test_expiry_applies_before_and_after_including_from_cache(
    db: Database, v1: httpx.AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    key, key_id = await _key(db)
    soon = datetime.now(UTC) + timedelta(hours=1)
    await db.set_api_key_expiry(key_id, soon)
    assert (await _post(v1, key)).status_code == 200  # cached now
    later = soon.timestamp() + 1
    monkeypatch.setattr(time, "time", lambda: later)
    resp = await _post(v1, key)
    assert resp.status_code == 401
    assert resp.json()["detail"] == "API key expired"
    # Kept, not deleted.
    assert await db.get_api_key(key_id) is not None


async def test_status_labels() -> None:
    now = datetime(2026, 1, 10, tzinfo=UTC)
    assert key_status({}, now=now)["status"] == "active"
    soon = {"expires_at": now + timedelta(days=3)}
    assert key_status(soon, now=now) == {
        "status": "expiring",
        "label": "expires in 3 days",
        "days_left": 3,
    }
    assert (
        key_status({"expires_at": now - timedelta(seconds=1)}, now=now)["status"]
        == "expired"
    )
    asked = {"expires_at": now - timedelta(days=1), "renewal_requested_at": now}
    assert key_status(asked, now=now)["label"] == "expired, renewal requested"
    assert key_status({"paused_at": now}, now=now)["status"] == "paused"
    held = {"paused_at": now, "pause_locked": True, "pause_reason": "owner inactive"}
    assert key_status(held, now=now) == {
        "status": "held",
        "label": "held (owner inactive)",
    }


async def test_max_lifetime_on_create_edit_and_renew(
    dash_client: DashClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    c, db = dash_client
    monkeypatch.setattr(settings, "max_key_lifetime_days", 30)
    monkeypatch.setattr(settings, "key_renewal_days", 365)
    admin, ann = await _users(db)
    far = (datetime.now(UTC) + timedelta(days=60)).date().isoformat()
    refused = await c.post("/api/my-keys", json={"expires_at": far}, headers=_as(ann))
    assert refused.status_code == 400
    assert "within 30 days" in refused.json()["detail"]
    made = (await c.post("/api/my-keys", json={}, headers=_as(ann))).json()
    expires = datetime.fromisoformat(made["expires_at"])
    assert timedelta(days=29) < expires - datetime.now(UTC) <= timedelta(days=30)
    edit = await c.patch(
        f"/api/my-keys/{made['id']}", json={"expires_at": None}, headers=_as(ann)
    )
    assert edit.status_code == 200  # blank means the longest allowed
    edit = await c.patch(
        f"/api/my-keys/{made['id']}", json={"expires_at": far}, headers=_as(ann)
    )
    assert edit.status_code == 400
    renewed = (
        await c.post(f"/api/my-keys/{made['id']}/renew", headers=_as(ann))
    ).json()
    left = datetime.fromisoformat(renewed["expires_at"]) - datetime.now(UTC)
    assert left <= timedelta(days=30)  # renewal days capped by the lifetime
    # Admins may issue keys that never expire.
    admin_key = (await c.post("/api/keys/generate", json={}, headers=_as(admin))).json()
    assert admin_key["expires_at"] is None


async def test_renewal_options_for_owner_and_admin(dash_client: DashClient) -> None:
    c, db = dash_client
    admin, ann = await _users(db)
    made = (await c.post("/api/my-keys", json={}, headers=_as(ann))).json()
    kid = made["id"]
    past = datetime.now(UTC) - timedelta(days=1)
    await db.set_api_key_expiry(kid, past)

    async def setmode(mode: str) -> None:
        resp = await c.put(
            "/api/keys/lifecycle-settings",
            json={"renewal_mode": mode},
            headers=_as(admin),
        )
        assert resp.status_code == 200, resp.text

    # allowed (the default): the owner renews at once, even after expiry.
    renewed = await c.post(f"/api/my-keys/{kid}/renew", headers=_as(ann))
    assert renewed.status_code == 200
    assert renewed.json()["key_status"] == "active"

    await setmode("off")
    assert (
        await c.post(f"/api/my-keys/{kid}/renew", headers=_as(ann))
    ).status_code == 403
    assert (
        await c.post(f"/api/keys/{kid}/renew", headers=_as(admin))
    ).status_code == 200

    await setmode("approval")
    await db.set_api_key_expiry(kid, past)
    asked = (await c.post(f"/api/my-keys/{kid}/renew", headers=_as(ann))).json()
    assert asked["key_status"] == "renewal_requested"
    key = await db.get_api_key(kid)
    assert key is not None
    assert key["expires_at"] is not None
    assert key["expires_at"] < datetime.now(UTC)  # a request revives nothing
    pending = (await c.get("/api/keys/renewal-requests", headers=_as(admin))).json()
    assert [k["id"] for k in pending["keys"]] == [kid]
    denied = await c.post(
        f"/api/keys/{kid}/renewal/deny",
        json={"reason": "no longer needed"},
        headers=_as(admin),
    )
    assert denied.json()["renewal_denied_reason"] == "no longer needed"
    assert denied.json()["renewal_requested_at"] is None
    await c.post(f"/api/my-keys/{kid}/renew", headers=_as(ann))
    approved = await c.post(f"/api/keys/{kid}/renewal/approve", headers=_as(admin))
    assert approved.json()["key_status"] == "active"
    assert approved.json()["renewal_denied_reason"] is None
    assert (
        await c.post(f"/api/keys/{kid}/renewal/approve", headers=_as(admin))
    ).status_code == 409


async def test_pending_request_does_not_revive_an_expired_key(
    db: Database, v1: httpx.AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    uid = await db.create_dashboard_user("ann", "x", False)
    key, kid = await _key(db, uid)
    await db.set_api_key_expiry(kid, datetime.now(UTC) - timedelta(minutes=1))
    await db.request_api_key_renewal(kid, uid, datetime.now(UTC))
    assert (await _post(v1, key)).status_code == 401


# --- pausing ---


async def test_owner_pauses_and_resumes(dash_client: DashClient, db: Database) -> None:
    c, _ = dash_client
    _, ann = await _users(db)
    made = (await c.post("/api/my-keys", json={}, headers=_as(ann))).json()
    paused = (await c.post(f"/api/my-keys/{made['id']}/pause", headers=_as(ann))).json()
    assert paused["key_status"] == "paused"
    resumed = (
        await c.post(f"/api/my-keys/{made['id']}/unpause", headers=_as(ann))
    ).json()
    assert resumed["key_status"] == "active"


async def test_owner_cannot_lift_or_overwrite_a_hold(dash_client: DashClient) -> None:
    c, db = dash_client
    admin, ann = await _users(db)
    made = (await c.post("/api/my-keys", json={}, headers=_as(ann))).json()
    kid = made["id"]
    # An owner pause, then an admin hold replaces it.
    await c.post(f"/api/my-keys/{kid}/pause", headers=_as(ann))
    held = await c.post(
        f"/api/keys/{kid}/pause", json={"reason": "audit"}, headers=_as(admin)
    )
    assert held.json()["key_status"] == "held"  # locked by default
    assert (
        await c.post(f"/api/my-keys/{kid}/unpause", headers=_as(ann))
    ).status_code == 403
    assert (
        await c.post(f"/api/my-keys/{kid}/pause", headers=_as(ann))
    ).status_code == 403
    key = await db.get_api_key(kid)
    assert key is not None
    assert key["pause_reason"] == "audit"
    lifted = await c.post(f"/api/keys/{kid}/unpause", headers=_as(admin))
    assert lifted.json()["key_status"] == "active"
    # Unlocked admin pause: the owner may lift it.
    await c.post(f"/api/keys/{kid}/pause", json={"locked": False}, headers=_as(admin))
    assert (
        await c.post(f"/api/my-keys/{kid}/unpause", headers=_as(ann))
    ).status_code == 200


async def test_hold_applies_on_the_next_request(
    db: Database, v1: httpx.AsyncClient
) -> None:
    key, kid = await _key(db)
    assert (await _post(v1, key)).status_code == 200  # cached
    await db.pause_api_key(
        kid, by_user_id=None, locked=True, reason="x", when=datetime.now(UTC)
    )
    resp = await _post(v1, key)
    assert resp.status_code == 403
    assert resp.json()["detail"] == "API key paused"
    await db.unpause_api_key(kid)
    assert (await _post(v1, key)).status_code == 200


async def test_regenerate_keeps_expiry_and_pause(dash_client: DashClient) -> None:
    c, db = dash_client
    admin, _ = await _users(db)
    made = (
        await c.post(
            "/api/keys/generate", json={"expires_at": "2099-01-01"}, headers=_as(admin)
        )
    ).json()
    kid = made["id"]
    await c.post(f"/api/keys/{kid}/pause", json={}, headers=_as(admin))
    await c.post(f"/api/keys/{kid}/regenerate", headers=_as(admin))
    key = await db.get_api_key(kid)
    assert key is not None
    assert key["expires_at"] == datetime(2099, 1, 2, tzinfo=UTC)  # end of 1 Jan
    assert key["paused_at"] is not None


# --- owner lifecycle ---


async def test_unapprove_and_pause_all_hold_keys(dash_client: DashClient) -> None:
    c, db = dash_client
    admin, ann = await _users(db)
    a = (await c.post("/api/my-keys", json={}, headers=_as(ann))).json()["id"]
    b = (await c.post("/api/my-keys", json={}, headers=_as(ann))).json()["id"]
    resp = await c.post(f"/users/{ann}/unapprove", headers=_as(admin))
    assert resp.status_code == 303
    for kid in (a, b):
        key = await db.get_api_key(kid)
        assert key is not None
        assert key["pause_locked"]
        assert key["pause_reason"] == "owner not approved"
    # Approving again does not lift the hold.
    await c.post(f"/users/{ann}/approve", headers=_as(admin))
    key = await db.get_api_key(a)
    assert key is not None
    assert key["pause_locked"]
    await db.unpause_api_key(a)
    resp = await c.post(f"/api/users/{ann}/pause-keys", headers=_as(admin))
    assert resp.json() == {"user_id": ann, "held": 2}
    key = await db.get_api_key(a)
    assert key is not None
    assert key["pause_locked"]


async def test_oidc_refusal_holds_keys_and_sign_in_does_not_lift(db: Database) -> None:
    issuer = "https://idp.test"
    cfg = oidc.OIDCConfig(
        issuer=issuer,
        client_id="c",
        client_secret="s",  # noqa: S106 - a test value
        scopes="openid",
        groups_claim="groups",
        admin_group="admins",
        allowed_group="staff",
        redirect_base_url="",
        display_name="SSO",
        only=False,
        logout_at_idp=False,
        allow_insecure=False,
    )
    # An account must exist first (the first SSO sign-in must be an admin).
    await db.create_dashboard_user("root", "x", True)
    claims = {
        "iss": issuer,
        "sub": "u1",
        "preferred_username": "una",
        "groups": ["staff"],
    }
    user = await oidc.sign_in(db, cfg, claims)
    _, kid = await _key(db, int(user["id"]))
    with pytest.raises(oidc.OIDCError):
        await oidc.sign_in(db, cfg, {**claims, "groups": []})
    key = await db.get_api_key(kid)
    assert key is not None
    assert (key["pause_locked"], key["pause_reason"]) == (True, "no longer a member")
    await oidc.sign_in(db, cfg, claims)  # back in the group
    key = await db.get_api_key(kid)
    assert key is not None
    assert key["pause_locked"]


async def test_idle_sweep_holds_inactive_sso_users_keys(
    db: Database, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(settings, "oidc_key_max_idle_days", 30)
    issuer = "https://idp.test"
    idle = await db.create_oidc_dashboard_user(
        "idle",
        issuer,
        "s1",
        email=None,
        display_name=None,
        is_admin=False,
    )
    active = await db.create_oidc_dashboard_user(
        "busy",
        issuer,
        "s2",
        email=None,
        display_name=None,
        is_admin=False,
    )
    local = await db.create_dashboard_user("local", "x", False)
    keys = {uid: (await _key(db, uid))[1] for uid in (idle, active, local)}
    await db.touch_dashboard_user_login(active)
    now = datetime.now(UTC) + timedelta(days=31)
    await db._execute(
        "UPDATE dashboard_users SET last_login_at = ? WHERE id = ?",
        (db._bind_time(datetime.now(UTC) + timedelta(days=5)), active),
    )
    held = await sweep_idle_sso_users(db, now=now)
    assert held == 1
    statuses: dict[int, dict[str, Any]] = {}
    for uid, kid in keys.items():
        row = await db.get_api_key(kid)
        assert row is not None
        statuses[uid] = row
    assert statuses[idle]["pause_reason"] == REASON_INACTIVE
    assert statuses[active]["paused_at"] is None
    assert statuses[local]["paused_at"] is None
    monkeypatch.setattr(settings, "oidc_key_max_idle_days", 0)
    assert await sweep_idle_sso_users(db, now=now) == 0


def test_migration_round_trip(tmp_path: Path) -> None:
    dbfile = tmp_path / "life.db"
    cfg = _cfg(dbfile)
    command.upgrade(cfg, "037_api_key_lifecycle")

    def cols() -> set[str]:
        with contextlib.closing(sqlite3.connect(dbfile)) as con:
            return {r[1] for r in con.execute("PRAGMA table_info(api_keys)")}

    wanted = {
        "expires_at",
        "renewal_requested_at",
        "renewal_denied_reason",
        "paused_at",
        "paused_by_user_id",
        "pause_locked",
        "pause_reason",
    }
    assert wanted <= cols()
    command.downgrade(cfg, "036_user_quotas")
    assert not (wanted & cols())
    command.upgrade(cfg, "head")
    assert wanted <= cols()


# --- review regressions ---


async def test_owner_pause_write_never_touches_a_hold(db: Database) -> None:
    uid = await db.create_dashboard_user("ann", "x", False)
    _, kid = await _key(db, uid)
    await db.pause_api_key(
        kid, by_user_id=None, locked=True, reason="audit", when=datetime.now(UTC)
    )
    now = datetime.now(UTC)
    assert not await db.owner_set_pause(kid, uid, paused=False, when=now)
    assert not await db.owner_set_pause(kid, uid, paused=True, when=now)
    key = await db.get_api_key(kid)
    assert key is not None
    assert (key["pause_locked"], key["pause_reason"]) == (True, "audit")


def test_date_only_expiry_lasts_through_that_day() -> None:
    from llama_router.key_lifecycle import parse_expiry

    assert parse_expiry("2030-05-01") == datetime(2030, 5, 2, tzinfo=UTC)
    assert parse_expiry("2030-05-01T12:00:00Z") == datetime(2030, 5, 1, 12, tzinfo=UTC)


async def test_owner_expiry_must_be_in_the_future(dash_client: DashClient) -> None:
    c, db = dash_client
    _, ann = await _users(db)
    resp = await c.post(
        "/api/my-keys", json={"expires_at": "2001-01-01"}, headers=_as(ann)
    )
    assert resp.status_code == 400


async def test_bad_expiry_in_admin_edit_changes_nothing(
    dash_client: DashClient,
) -> None:
    c, db = dash_client
    admin, _ = await _users(db)
    made = (await c.post("/api/keys/generate", json={}, headers=_as(admin))).json()
    resp = await c.patch(
        f"/api/keys/{made['id']}",
        json={"label": "new", "expires_at": "not a date"},
        headers=_as(admin),
    )
    assert resp.status_code == 400
    key = await db.get_api_key(made["id"])
    assert key is not None
    assert key["label"] is None


async def test_refused_key_is_not_marked_used(db: Database) -> None:
    plaintext, kid = await _key(db)
    await db.pause_api_key(
        kid, by_user_id=None, locked=True, reason=None, when=datetime.now(UTC)
    )
    await db.lookup_api_key(key_hash(plaintext))
    key = await db.get_api_key(kid)
    assert key is not None
    assert key["last_used_at"] is None
