"""The first administrator comes from the environment; anonymous access is opt-in."""

from __future__ import annotations

import logging

import pytest

from llama_router.__main__ import _log_oidc_config, _warn_if_anonymous
from llama_router.config import settings
from llama_router.dashboard.auth_core import hash_password, verify_password
from llama_router.dashboard.initial_admin import ensure_initial_admin
from llama_router.database import Database


async def test_creates_the_administrator(db: Database) -> None:
    assert await ensure_initial_admin(db, "Root", "a-long-password")
    row = await db.get_dashboard_user_by_username("root")
    assert row is not None
    assert row["is_admin"] is True
    assert verify_password("a-long-password", row["password_hash"])


async def test_ignored_once_an_account_exists(db: Database) -> None:
    await db.create_dashboard_user("alice", hash_password("pw-123456789"), True)
    assert not await ensure_initial_admin(db, "root", "a-long-password")
    assert await db.get_dashboard_user_by_username("root") is None


async def test_warns_when_unset(db: Database, caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING):
        assert not await ensure_initial_admin(db, "", "")
    assert "LLAMA_ROUTER_ADMIN_USERNAME" in caplog.text
    assert await db.count_dashboard_users() == 0


@pytest.mark.parametrize(
    ("username", "password"),
    [("root", ""), ("root", "short"), ("", "a-long-password"), ("bad name", "x" * 9)],
)
async def test_refuses_unusable_values(
    db: Database, caplog: pytest.LogCaptureFixture, username: str, password: str
) -> None:
    with caplog.at_level(logging.ERROR):
        assert not await ensure_initial_admin(db, username, password)
    assert "LLAMA_ROUTER_ADMIN_" in caplog.text
    assert await db.count_dashboard_users() == 0


async def test_missing_setting_means_no_anonymous_access(db: Database) -> None:
    await db.set_allow_unauthenticated(True)
    await db._execute(
        "DELETE FROM app_settings WHERE app_settings.key = 'allow_unauthenticated'"
    )
    assert await db.get_allow_unauthenticated() is False


async def test_startup_warns_when_anonymous_access_is_on(
    db: Database, caplog: pytest.LogCaptureFixture
) -> None:
    await db.set_allow_unauthenticated(False)
    with caplog.at_level(logging.WARNING):
        await _warn_if_anonymous(db)
    assert "Unauthenticated" not in caplog.text
    await db.set_allow_unauthenticated(True)
    with caplog.at_level(logging.WARNING):
        await _warn_if_anonymous(db)
    assert "Unauthenticated API access is ON" in caplog.text


async def test_lost_race_with_another_replica(
    db: Database, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Two routers sharing a database start together; one creates the admin."""
    real_create = db.create_dashboard_user

    async def other_replica_wins(name: str, password_hash: str, is_admin: bool) -> int:
        await real_create(name, password_hash, is_admin)
        return await real_create(name, password_hash, is_admin)

    monkeypatch.setattr(db, "create_dashboard_user", other_replica_wins)
    assert not await ensure_initial_admin(db, "root", "a-long-password")
    assert await db.count_dashboard_users() == 1


async def test_other_insert_errors_are_raised(
    db: Database, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def broken(name: str, password_hash: str, is_admin: bool) -> int:
        raise RuntimeError("database down")

    monkeypatch.setattr(db, "create_dashboard_user", broken)
    with pytest.raises(RuntimeError):
        await ensure_initial_admin(db, "root", "a-long-password")


@pytest.mark.parametrize(("admin_group", "disabled"), [("admins", True), ("", False)])
def test_oidc_only_startup_log_matches_password_sign_in(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    admin_group: str,
    disabled: bool,
) -> None:
    """OIDC_ONLY without an admin group keeps password sign-in; the log says so."""
    monkeypatch.setattr(settings, "oidc_issuer", "https://idp.example.org")
    monkeypatch.setattr(settings, "oidc_client_id", "router")
    monkeypatch.setattr(settings, "oidc_client_secret", "secret")
    monkeypatch.setattr(settings, "oidc_admin_group", admin_group)
    monkeypatch.setattr(settings, "oidc_only", True)
    with caplog.at_level(logging.INFO):
        _log_oidc_config()
    assert ("password sign-in disabled" in caplog.text) is disabled
    assert ("password sign-in stays enabled" in caplog.text) is not disabled
