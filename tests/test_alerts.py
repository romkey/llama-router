"""Webhook alerts."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta
from typing import Any

import httpx
import pytest

from llama_router import alerts as alerts_mod
from llama_router.alerts import LagWatch, alerts, periodic_checks
from llama_router.auth import generate_api_key, key_prefix
from llama_router.config import settings
from llama_router.database import Database
from llama_router.key_lifecycle import hold_user_keys
from llama_router.models import ProviderProtocol, ProviderStatus
from llama_router.provider_secrets import key_hash
from llama_router.quotas import check_rate
from llama_router.single_process import LoopLag


class Hook:
    def __init__(self) -> None:
        self.sent: list[dict[str, Any]] = []
        self.fail = False

    async def post(self, url: str, payload: dict[str, Any]) -> None:
        if self.fail:
            raise httpx.ConnectError("down")
        self.sent.append(payload)


@pytest.fixture
async def hook(monkeypatch: pytest.MonkeyPatch) -> AsyncIterator[Hook]:
    monkeypatch.setattr(settings, "alert_webhook_url", "https://hooks.test/x")
    monkeypatch.setattr(settings, "alert_events", "all")
    monkeypatch.setattr(alerts_mod, "RETRY_DELAY_S", 0.0)
    h = Hook()
    alerts.reset()
    monkeypatch.setattr(alerts, "post", h.post)
    yield h
    await alerts.stop()
    alerts.reset()


async def _events(h: Hook) -> list[str]:
    await alerts.drain()
    return [p["event"] for p in h.sent]


async def test_empty_url_sends_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "alert_webhook_url", "")
    alerts.reset()
    assert not alerts.emit("provider_offline", "1", "x")
    assert not alerts.configured
    assert alerts._queue is None


async def test_provider_state_changes_alert_once_and_debounce(
    db: Database, hook: Hook
) -> None:
    p = await db.add_provider("gpu", ProviderProtocol.OLLAMA)
    pid = p.require_id()
    await db.update_provider_status(pid, ProviderStatus.IDLE)  # baseline
    await db.update_provider_status(pid, ProviderStatus.OFFLINE)
    await db.update_provider_status(pid, ProviderStatus.IDLE)
    await db.update_provider_status(pid, ProviderStatus.OFFLINE)  # debounced
    assert await _events(hook) == ["provider_offline", "provider_online"]
    assert hook.sent[0]["text"] == "Provider gpu is offline"
    assert hook.sent[0]["details"] == {"provider_id": pid, "provider": "gpu"}
    assert set(hook.sent[0]) == {"event", "time", "text", "details"}


async def test_model_unavailable(hook: Hook) -> None:
    alerts.models_served({"a", "b"})
    alerts.models_served({"a"})
    alerts.models_served({"a"})
    assert await _events(hook) == ["model_unavailable"]
    assert hook.sent[0]["details"] == {"model": "b"}


async def test_quota_alert_once_per_day(db: Database, hook: Hook) -> None:
    uid = await db.create_dashboard_user("ann", "x", False)
    check_rate(db, key_id=None, key_rpm=None, user_id=uid, user_rpm=1)
    for _ in range(3):
        with pytest.raises(Exception):  # noqa: B017,PT011 - the 429 itself
            check_rate(db, key_id=None, key_rpm=None, user_id=uid, user_rpm=1)
    assert await _events(hook) == ["quota_reached"]


async def test_held_keys_expiring_and_lag(db: Database, hook: Hook) -> None:
    uid = await db.create_dashboard_user("ann", "x", False)
    plaintext = generate_api_key()
    kid = await db.create_api_key(
        key_prefix(plaintext),
        key_hash(plaintext),
        "balanced",
        True,
        owner_user_id=uid,
        label="laptop",
    )
    await hold_user_keys(db, uid, "owner inactive")
    # An admin's hold is not the router's: no alert.
    await db.unpause_api_key(kid)
    await hold_user_keys(db, uid, "paused by an administrator", by_user_id=1)
    await db.set_api_key_expiry(kid, datetime.now(UTC) + timedelta(days=2))
    lag = LoopLag()
    watch = LagWatch()
    await periodic_checks(db, lag, watch)  # baseline for lag
    for _ in range(100):
        lag.observe(0.2)
    await periodic_checks(db, lag, watch)
    events = await _events(hook)
    assert events == ["keys_held", "key_expiring", "event_loop_lag"]
    text = json.dumps(hook.sent)
    assert plaintext not in text
    assert key_prefix(plaintext).rstrip("…") not in text
    assert hook.sent[1]["details"]["label"] == "laptop"


async def test_worker_token_expiring(db: Database, hook: Hook) -> None:
    soon = int((datetime.now(UTC) + timedelta(days=3)).timestamp())
    p = await db.add_provider("w1", ProviderProtocol.OLLAMA)
    await db.create_worker(
        "w1", "hash", "lrw_abc", p.require_id(), token_expires_at=soon
    )
    await periodic_checks(db, LoopLag(), LagWatch())
    assert await _events(hook) == ["worker_token_expiring"]
    assert "lrw_abc" not in json.dumps(hook.sent)


async def test_delivery_failure_never_reaches_the_caller(hook: Hook) -> None:
    hook.fail = True
    assert alerts.emit("provider_offline", "1", "x")  # queued, returns at once
    await alerts.drain()
    assert alerts.failed == 1
    assert hook.sent == []


async def test_full_queue_drops(hook: Hook, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(alerts_mod, "QUEUE_MAX", 2)
    alerts.reset()
    gate = asyncio.Event()

    async def slow(url: str, payload: dict[str, Any]) -> None:
        await gate.wait()

    monkeypatch.setattr(alerts, "post", slow)
    results = [alerts.emit("model_unavailable", str(i), "x") for i in range(5)]
    assert results.count(False) >= 2
    assert alerts.dropped >= 2
    gate.set()


async def test_events_setting_filters(
    hook: Hook, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(settings, "alert_events", "provider_offline")
    assert not alerts.emit("model_unavailable", "m", "x")
    assert alerts.emit("provider_offline", "1", "x")


async def test_send_test_alert(
    dash_client: tuple[httpx.AsyncClient, Database], hook: Hook
) -> None:
    from llama_router.dashboard.auth_core import AUTH_COOKIE, sign_session
    from llama_router.dashboard.middleware import (
        invalidate_dashboard_user_count_cache,
    )
    from llama_router.provider_secrets import session_signing_key

    c, db = dash_client
    admin = await db.create_dashboard_user("adm", "x", True)
    invalidate_dashboard_user_count_cache()
    cookie = {"Cookie": f"{AUTH_COOKIE}={sign_session(admin, session_signing_key())}"}
    resp = await c.post("/api/alerts/test", headers=cookie)
    assert resp.json() == {"delivered": True}
    assert hook.sent[-1]["event"] == "test"
    page = await c.get("/users", headers=cookie)
    assert "hooks.test" in page.text


async def test_lasting_states_repeat_at_most_daily(
    db: Database, hook: Hook, monkeypatch: pytest.MonkeyPatch
) -> None:
    clock = [1000.0]
    monkeypatch.setattr(alerts, "clock", lambda: clock[0])
    uid = await db.create_dashboard_user("ann", "x", False)
    plaintext = generate_api_key()
    kid = await db.create_api_key(
        key_prefix(plaintext), key_hash(plaintext), "balanced", True, owner_user_id=uid
    )
    await db.set_api_key_expiry(kid, datetime.now(UTC) + timedelta(days=2))
    watch = LagWatch()
    await periodic_checks(db, LoopLag(), watch)
    clock[0] += 3600  # an hour later: past the 10-minute debounce
    await periodic_checks(db, LoopLag(), watch)
    alerts.quota_reached("tokens", "User 1", "user:1")
    clock[0] += 3600
    alerts.quota_reached("tokens", "User 1", "user:1")
    assert await _events(hook) == ["key_expiring", "quota_reached"]


async def test_first_change_after_restart_is_reported(db: Database, hook: Hook) -> None:
    p = await db.add_provider("gpu", ProviderProtocol.OLLAMA)
    pid = p.require_id()
    await db.update_provider_status(pid, ProviderStatus.IDLE)
    alerts.reset()  # a restart: nothing in memory
    await db.update_provider_status(pid, ProviderStatus.OFFLINE)
    assert await _events(hook) == ["provider_offline"]


async def test_worker_marked_offline_at_start_up_is_no_outage(
    db: Database, hook: Hook
) -> None:
    p = await db.add_provider("w", ProviderProtocol.OLLAMA)
    pid = p.require_id()
    await db.update_provider_status(pid, ProviderStatus.IDLE)
    alerts.reset()  # restart
    alerts.forget_provider(pid)
    await db.update_provider_status(pid, ProviderStatus.OFFLINE)  # start-up
    await db.update_provider_status(pid, ProviderStatus.IDLE)  # dials in
    await db.update_provider_status(pid, ProviderStatus.OFFLINE)  # real outage
    assert await _events(hook) == ["provider_offline"]
