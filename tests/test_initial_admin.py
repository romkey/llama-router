"""The first administrator comes from the environment; anonymous access is opt-in."""

from __future__ import annotations

import logging

import pytest

from llama_router.__main__ import _warn_if_anonymous
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
