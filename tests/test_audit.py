"""The audit trail of state-changing dashboard actions."""

from __future__ import annotations

import contextlib
import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import httpx
import pytest
from alembic import command

from llama_router import audit
from llama_router.config import settings
from llama_router.dashboard.app import app as dash_app
from llama_router.dashboard.auth_core import AUTH_COOKIE, hash_password, sign_session
from llama_router.dashboard.middleware import invalidate_dashboard_user_count_cache
from llama_router.database import Database
from llama_router.provider_manager import ProviderManager
from llama_router.provider_secrets import session_signing_key
from tests.test_alembic_roundtrip import _cfg

DashClient = tuple[httpx.AsyncClient, Database]


def _as(user_id: int) -> dict[str, str]:
    return {"Cookie": f"{AUTH_COOKIE}={sign_session(user_id, session_signing_key())}"}


async def _users(db: Database) -> tuple[int, int]:
    admin = await db.create_dashboard_user("adm", hash_password("pw-123456789"), True)
    ann = await db.create_dashboard_user("ann", hash_password("pw-123456789"), False)
    invalidate_dashboard_user_count_cache()
    return admin, ann


def _routes(routes: Any) -> Any:
    for r in routes:
        if type(r).__name__ == "_IncludedRouter":
            inner = next((v for v in vars(r).values() if hasattr(v, "routes")), None)
            yield from _routes(inner.routes if inner is not None else [])
        else:
            yield r


def test_every_state_changing_route_writes_an_audit_entry() -> None:
    """Walks the dashboard's routes: each non-GET one is in ACTIONS (the
    middleware writes its entry) or EXEMPT (with a reason)."""
    seen = set()
    missing = []
    for route in _routes(dash_app.router.routes):
        methods = getattr(route, "methods", None) or set()
        if not methods - {"GET", "HEAD"}:
            continue
        name = audit.route_name(route.endpoint)
        seen.add(name)
        if name not in audit.ACTIONS and name not in audit.EXEMPT:
            missing.append(f"{sorted(methods)} {route.path} ({name})")
    assert not missing, "routes with no audit action:\n" + "\n".join(missing)
    stale = (set(audit.ACTIONS) | set(audit.EXEMPT)) - seen
    assert not stale, f"audit entries for routes that no longer exist: {stale}"
    assert not set(audit.ACTIONS) & set(audit.EXEMPT)


async def test_one_entry_per_action_with_the_actor(dash_client: DashClient) -> None:
    c, db = dash_client
    admin, ann = await _users(db)
    made = await c.post("/api/keys/generate", json={"label": "ci"}, headers=_as(admin))
    kid = made.json()["id"]
    await c.patch(
        f"/api/keys/{kid}",
        json={"label": "ci-2", "daily_spend_cap_usd": 3},
        headers=_as(admin),
    )
    await c.post(f"/api/keys/{kid}/pause", json={"reason": "check"}, headers=_as(admin))
    mine = await c.post("/api/my-keys", json={}, headers=_as(ann))
    entries = await db.list_audit_log()
    got = [(e["action"], e["actor_name"], e["target_id"]) for e in entries][::-1]
    assert got == [
        ("api_key.create", "adm", None),
        ("api_key.update", "adm", str(kid)),
        ("api_key.pause", "adm", str(kid)),
        ("api_key.create", "ann", None),
    ]
    update = entries[2]
    assert update["actor_kind"] == "user"
    assert "label: ci -> ci-2" in update["summary"]
    assert "daily_spend_cap_usd: none -> 3.0" in update["summary"]
    assert "reason: check" in entries[1]["summary"]
    assert mine.status_code == 200
    # Owners see entries for their own keys only.
    own = (await c.get("/api/my-keys/audit", headers=_as(ann))).json()["entries"]
    assert own == []
    await c.post(f"/api/my-keys/{mine.json()['id']}/pause", headers=_as(ann))
    own = (await c.get("/api/my-keys/audit", headers=_as(ann))).json()["entries"]
    assert [e["action"] for e in own] == ["api_key.pause"]
    assert "client_ip" not in own[0]


async def test_failed_request_writes_nothing(dash_client: DashClient) -> None:
    c, db = dash_client
    admin, _ = await _users(db)
    resp = await c.post("/api/keys/999/pause", json={}, headers=_as(admin))
    assert resp.status_code == 404
    assert await db.list_audit_log() == []


async def test_management_key_is_recorded_as_itself(dash_client: DashClient) -> None:
    c, db = dash_client
    admin, _ = await _users(db)
    r = await c.post(
        "/api/management-keys",
        json={"name": "bot", "privileges": ["keys:write"]},
        headers=_as(admin),
    )
    key = r.json()["management_key"]
    resp = await c.post(
        "/api/keys/generate", json={}, headers={"Authorization": f"Bearer {key}"}
    )
    assert resp.status_code == 200
    entry = (await db.list_audit_log())[0]
    assert entry["actor_kind"] == "management_key"
    assert entry["actor_name"] == "bot"
    assert entry["action"] == "api_key.create"
    # It may not read the trail without audit:read; one with it may.
    denied = await c.get("/api/audit", headers={"Authorization": f"Bearer {key}"})
    assert denied.status_code == 403
    r = await c.post(
        "/api/management-keys",
        json={"name": "auditor", "privileges": ["audit:read"]},
        headers=_as(admin),
    )
    auditor = r.json()["management_key"]
    ok = await c.get(
        "/api/audit?action_prefix=api_key.",
        headers={"Authorization": f"Bearer {auditor}"},
    )
    assert ok.status_code == 200
    assert [e["action"] for e in ok.json()["entries"]] == ["api_key.create"]


async def test_no_secret_in_any_summary(dash_client: DashClient) -> None:
    c, db = dash_client
    admin, ann = await _users(db)
    await c.post(
        "/users/add",
        data={"username": "bob", "password": "hunter2-secret-pw", "is_admin": "0"},
        headers=_as(admin),
    )
    await c.post(
        f"/users/{ann}/password",
        data={"new_password": "another-secret-pw"},
        headers=_as(admin),
    )
    provider = await c.post(
        "/providers/add",
        data={
            "name": "cloudy",
            "url": "https://api.example.test/v1",
            "protocol": "openai",
            "api_key": "sk-live-SECRETVALUE",
        },
        headers=_as(admin),
    )
    assert provider.status_code in (200, 303), provider.text
    p = await db.get_provider_by_name("cloudy")
    assert p is not None
    if p is not None:
        await c.post(
            f"/providers/{p.id}/edit",
            data={
                "name": "cloudy",
                "protocol": "openai",
                "api_key": "sk-live-OTHERSECRET",
                "settings_form": "1",
                "extra_headers": "Authorization: Bearer HEADERSECRET",
            },
            headers=_as(admin),
        )
    made = await c.post("/api/keys/generate", json={}, headers=_as(admin))
    plaintext = made.json()["api_key"]
    await c.post(f"/api/keys/{made.json()['id']}/regenerate", headers=_as(admin))
    entries = await db.list_audit_log()
    assert len(entries) >= 4
    text = "\n".join(str(e["summary"]) for e in entries)
    for secret in (
        "hunter2-secret-pw",
        "another-secret-pw",
        "SECRETVALUE",
        "OTHERSECRET",
        "HEADERSECRET",
        plaintext,
    ):
        assert secret not in text
    assert "password: changed" in text
    if p is not None:
        assert "api_key: changed" in text


def test_changes_masks_secrets_and_lists_old_and_new() -> None:
    old = {"label": "a", "api_key": "x", "cap": None}
    new = {"label": "b", "api_key": "y", "cap": 2, "same": 1}
    assert audit.changes(old, new) == (
        "label: a -> b; api_key: changed; cap: none -> 2; same: 1"
    )
    assert audit.changes(old, {"label": "a"}) == "no change"


async def test_system_entries_and_retention(
    db: Database, monkeypatch: pytest.MonkeyPatch
) -> None:
    await audit.record(None, "user.keys_held", ("user", 3), "held", db=db)
    entry = (await db.list_audit_log())[0]
    assert (entry["actor_kind"], entry["actor_name"]) == ("system", "router")
    await db._execute(
        "UPDATE audit_log SET at = ?",
        (db._bind_time(datetime.now(UTC) - timedelta(days=400)),),
    )
    await audit.record(None, "routing.rebuild", ("system", None), None, db=db)
    monkeypatch.setattr(settings, "audit_retention_days", 365)
    monkeypatch.setattr(settings, "request_log_retention_days", 0)
    await ProviderManager(db).purge_request_logs(force=True)
    assert [e["action"] for e in await db.list_audit_log()] == ["routing.rebuild"]


async def test_audit_page_is_admin_only(dash_client: DashClient) -> None:
    c, db = dash_client
    admin, ann = await _users(db)
    await audit.record(None, "routing.rebuild", ("system", None), "x", db=db)
    page = await c.get("/audit?actor_kind=system", headers=_as(admin))
    assert page.status_code == 200
    assert "routing.rebuild" in page.text
    assert (await c.get("/audit", headers=_as(ann))).status_code in (302, 403)
    assert (await c.get("/api/audit", headers=_as(ann))).status_code in (302, 403)


def test_migration_round_trip(tmp_path: Path) -> None:
    dbfile = tmp_path / "audit.db"
    cfg = _cfg(dbfile)

    def tables() -> set[str]:
        with contextlib.closing(sqlite3.connect(dbfile)) as con:
            return {
                r[0]
                for r in con.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                )
            }

    command.upgrade(cfg, "038_audit_log")
    assert "audit_log" in tables()
    command.downgrade(cfg, "037_api_key_lifecycle")
    assert "audit_log" not in tables()
    command.upgrade(cfg, "head")
    assert "audit_log" in tables()


async def test_refused_form_change_writes_nothing(dash_client: DashClient) -> None:
    c, db = dash_client
    admin, ann = await _users(db)
    resp = await c.post(
        f"/users/{ann}/limits", data={"daily_token_limit": "0"}, headers=_as(admin)
    )
    assert "error=" in resp.headers["location"]
    assert await db.list_audit_log() == []
