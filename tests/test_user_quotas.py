"""Per-user and per-key quotas: daily tokens, requests per minute and a
daily spend cap across a user's keys."""

from __future__ import annotations

import asyncio
import contextlib
import sqlite3
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import httpx
import pytest
from alembic import command
from fastapi import HTTPException

from llama_router import metrics, proxy, request_logger
from llama_router.auth import (
    generate_api_key,
    invalidate_auth_cache,
    key_prefix,
)
from llama_router.database import Database
from llama_router.llamacpp_api import deps as v1_deps
from llama_router.llamacpp_api.app import app as v1_app
from llama_router.models import (
    BenchmarkResult,
    ProviderModel,
    ProviderProtocol,
    ProviderStatus,
)
from llama_router.pricing import estimate_tokens
from llama_router.provider_manager import ProviderManager
from llama_router.provider_secrets import key_hash
from llama_router.quotas import UserLimits, set_default_limits
from llama_router.router import Router
from llama_router.spend import SpendTracker, _day_start, _trackers, tracker_for
from tests.conftest import asgi_client
from tests.test_alembic_roundtrip import _cfg

CHAT = "/v1/chat/completions"
# max_tokens keeps the held estimate small: a few prompt tokens and 20.
BODY = {"model": "m", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 20}
USAGE = {"prompt_tokens": 30, "completion_tokens": 20}  # 50 a request


@pytest.fixture
def pm(db: Database) -> ProviderManager:
    return ProviderManager(db)


@pytest.fixture
async def rt(db: Database, pm: ProviderManager) -> AsyncIterator[Router]:
    router = Router(db, pm)
    request_logger.attach_pricer(router.price_for)
    yield router
    request_logger.attach_pricer(None)


@pytest.fixture
async def v1(
    db: Database, pm: ProviderManager, rt: Router
) -> AsyncIterator[httpx.AsyncClient]:
    v1_deps.init(db, pm, rt)
    request_logger.start_log_writer()
    async with asgi_client(v1_app) as client:
        yield client
    await request_logger.stop_log_writer()
    v1_deps._db, v1_deps._pm, v1_deps._router = None, None, None


class Upstream:
    def __init__(self) -> None:
        self.calls = 0
        self.gate: asyncio.Event | None = None

    def client(self, pm: ProviderManager, provider_id: int) -> Any:
        up = self

        class Client:
            async def chat_completions(self, body: dict[str, Any]) -> dict[str, Any]:
                up.calls += 1
                if up.gate is not None:
                    await up.gate.wait()
                return {
                    "choices": [{"message": {"content": "ok"}}],
                    "usage": dict(USAGE),
                }

        return Client()


@pytest.fixture
def up(monkeypatch: pytest.MonkeyPatch) -> Upstream:
    u = Upstream()
    monkeypatch.setattr(proxy, "get_v1_client", u.client)
    return u


async def _provider(db: Database, name: str, *, locality: str = "local") -> int:
    p = await db.add_provider(name, ProviderProtocol.OPENAI)
    pid = p.require_id()
    await db.update_provider_status(pid, ProviderStatus.IDLE)
    await db.update_provider_settings(pid, locality=locality)
    await db.set_provider_models(pid, [ProviderModel(provider_id=pid, name="m")])
    await db.save_benchmark(
        BenchmarkResult(
            provider_id=pid,
            model_name="m",
            protocol="openai",
            startup_time_ms=100.0 if locality == "local" else 50.0,
            tokens_per_second=50.0,
        )
    )
    return pid


async def _user(db: Database, name: str = "alice", **limits: Any) -> int:
    uid = await db.create_dashboard_user(name, "x", False)
    if limits:
        await db.set_dashboard_user_limits(uid, limits)
    return uid


async def _key(db: Database, owner: int | None = None, **kw: Any) -> tuple[str, int]:
    plaintext = generate_api_key()
    limits: dict[str, Any] = {
        k: kw.pop(k) for k in ("daily_token_limit", "requests_per_minute") if k in kw
    }
    key_id = await db.create_api_key(
        key_prefix(plaintext),
        key_hash(plaintext),
        kw.pop("mode", "balanced"),
        True,
        owner_user_id=owner,
        **kw,
    )
    if limits:
        await db.update_api_key(key_id, limits)
    return plaintext, key_id


async def _post(v1: httpx.AsyncClient, key: str) -> httpx.Response:
    resp = await v1.post(CHAT, json=BODY, headers={"Authorization": f"Bearer {key}"})
    await request_logger.flush_request_log()
    return resp


async def test_token_limit_sums_across_a_users_keys(
    db: Database, v1: httpx.AsyncClient, up: Upstream
) -> None:
    await _provider(db, "local")
    uid = await _user(db, daily_token_limit=100)
    a, _ = await _key(db, uid)
    b, _ = await _key(db, uid)
    assert (await _post(v1, a)).status_code == 200
    assert (await _post(v1, b)).status_code == 200
    # 100 tokens used across the two keys: the next is refused.
    refused = await _post(v1, a)
    assert refused.status_code == 429
    detail = refused.json()["detail"]
    assert "daily token limit of 100 reached for this user" in detail
    assert "resets at" in detail
    assert int(refused.headers["retry-after"]) > 0
    assert up.calls == 2
    rows = await db._select_all("SELECT user_id, api_key_id FROM request_log")
    assert {r["user_id"] for r in rows} == {uid}


async def test_revoke_and_reissue_does_not_reset_usage(
    db: Database, v1: httpx.AsyncClient, up: Upstream
) -> None:
    await _provider(db, "local")
    uid = await _user(db, daily_token_limit=60)
    old, old_id = await _key(db, uid)
    assert (await _post(v1, old)).status_code == 200
    await db.delete_api_key(old_id)
    new, _ = await _key(db, uid)
    # The new key's first request is checked against 50 tokens already used
    # today; its estimate does not fit in the 10 left.
    assert (await _post(v1, new)).status_code == 429


async def test_usage_survives_a_restart(
    db: Database, v1: httpx.AsyncClient, up: Upstream
) -> None:
    await _provider(db, "local")
    uid = await _user(db, daily_token_limit=60)
    key, _ = await _key(db, uid)
    assert (await _post(v1, key)).status_code == 200
    # A new process: a fresh tracker, seeded from the request log.
    _trackers.pop(db, None)
    await tracker_for(db).seed_today()
    assert tracker_for(db).usage(user_id=uid).tokens == 50
    invalidate_auth_cache(db)
    assert (await _post(v1, key)).status_code == 429


async def test_local_requests_count_tokens_not_dollars(
    db: Database, v1: httpx.AsyncClient, up: Upstream
) -> None:
    await _provider(db, "local")
    uid = await _user(db, daily_spend_cap_usd=0.01)
    key, _ = await _key(db, uid)
    for _ in range(3):
        assert (await _post(v1, key)).status_code == 200
    usage = await db.usage_since(_day_start(), user_id=uid)
    assert usage == (0.0, 150)


async def test_user_spend_cap_moves_keys_to_local_providers(
    db: Database, v1: httpx.AsyncClient, up: Upstream, rt: Router
) -> None:
    local = await _provider(db, "local")
    cloud = await _provider(db, "cloud", locality="cloud")
    await db.set_model_price(cloud, "m", 1_000_000.0, 1_000_000.0)  # $1/token
    uid = await _user(db, daily_spend_cap_usd=10.0)
    tracker = tracker_for(db)
    await tracker.user_committed_today(uid)
    tracker.add(None, 10.0, user_id=uid)
    key, _ = await _key(db, uid)
    resp = await _post(v1, key)
    assert resp.status_code == 200
    rows = await db._select_all("SELECT provider_id FROM request_log")
    assert [r["provider_id"] for r in rows] == [local]
    # A cloud-only key past its owner's cap is refused.
    cloud_key, _ = await _key(db, uid, locality="cloud")
    refused = await _post(v1, cloud_key)
    assert refused.status_code == 429
    assert (
        "daily spend cap of $10.00 reached for this user" in (refused.json()["detail"])
    )


async def test_concurrent_requests_cannot_jointly_exceed_the_limit(
    db: Database, v1: httpx.AsyncClient, up: Upstream
) -> None:
    await _provider(db, "local")
    # Each request holds its estimate (a few prompt tokens and the default
    # completion estimate) until logged, so only one fits at a time.
    estimate = sum(estimate_tokens(BODY, 0))
    uid = await _user(db, daily_token_limit=estimate + estimate // 2)
    key, _ = await _key(db, uid)
    up.gate = asyncio.Event()
    headers = {"Authorization": f"Bearer {key}"}
    first = asyncio.create_task(v1.post(CHAT, json=BODY, headers=headers))
    for _ in range(100):
        if up.calls:
            break
        await asyncio.sleep(0.01)
    second = await v1.post(CHAT, json=BODY, headers=headers)
    assert second.status_code == 429
    up.gate.set()
    assert (await first).status_code == 200


async def test_requests_per_minute_key_tighter_than_user(
    db: Database, v1: httpx.AsyncClient, up: Upstream
) -> None:
    await _provider(db, "local")
    uid = await _user(db, requests_per_minute=5)
    tight, _ = await _key(db, uid, requests_per_minute=2)
    loose, _ = await _key(db, uid)
    assert (await _post(v1, tight)).status_code == 200
    assert (await _post(v1, tight)).status_code == 200
    refused = await _post(v1, tight)
    assert refused.status_code == 429
    assert (
        "rate limit of 2 requests per minute reached for this API key"
        in (refused.json()["detail"])
    )
    assert 1 <= int(refused.headers["retry-after"]) <= 60
    # The user's own limit counts both keys: 2 + 3 = 5.
    for _ in range(3):
        assert (await _post(v1, loose)).status_code == 200
    assert (await _post(v1, loose)).status_code == 429
    text = metrics.render(
        metrics.collect(
            [],
            _NoStats(),
            quotas=(
                tracker_for(db),
                [
                    {
                        "id": uid,
                        "username": "alice",
                        "limits": {"requests_per_minute": 5},
                    }
                ],
            ),
        )
    )
    assert 'llama_router_quota_refusals_total{limit="key_rpm"} 1' in text
    assert 'llama_router_quota_refusals_total{limit="user_rpm"} 1' in text
    assert 'llama_router_user_requests_per_minute_limit{user="alice"' in text


class _NoStats:
    def model_entries(self) -> list[object]:
        return []


async def test_ownerless_key_has_key_limits_only(
    db: Database, v1: httpx.AsyncClient, up: Upstream
) -> None:
    await _provider(db, "local")
    await set_default_limits(db, UserLimits(daily_token_limit=1))
    key, _ = await _key(db, None)
    for _ in range(3):
        assert (await _post(v1, key)).status_code == 200
    capped, _ = await _key(db, None, daily_token_limit=50)
    assert (await _post(v1, capped)).status_code == 200
    assert (await _post(v1, capped)).status_code == 429


async def test_defaults_apply_until_a_user_has_their_own(
    db: Database, v1: httpx.AsyncClient, up: Upstream
) -> None:
    await _provider(db, "local")
    uid = await _user(db)
    key, _ = await _key(db, uid)
    await set_default_limits(db, UserLimits(requests_per_minute=1))
    invalidate_auth_cache(db)
    assert (await _post(v1, key)).status_code == 200
    assert (await _post(v1, key)).status_code == 429
    await db.set_dashboard_user_limits(uid, {"requests_per_minute": 10})
    assert (await _post(v1, key)).status_code == 200


async def test_failed_request_releases_its_token_hold(db: Database) -> None:
    tracker = SpendTracker(db)
    uid = await _user(db)
    held = await tracker.reserve_tokens(
        80, key_id=None, key_limit=None, user_id=uid, user_limit=100
    )
    assert not isinstance(held, str)
    assert (
        await tracker.reserve_tokens(
            80, key_id=None, key_limit=None, user_id=uid, user_limit=100
        )
        == "user_tokens"
    )
    held.release()
    again = await tracker.reserve_tokens(
        80, key_id=None, key_limit=None, user_id=uid, user_limit=100
    )
    assert not isinstance(again, str)


def test_migration_round_trip(tmp_path: Path) -> None:
    dbfile = tmp_path / "quotas.db"
    cfg = _cfg(dbfile)
    command.upgrade(cfg, "036_user_quotas")

    def cols(table: str) -> set[str]:
        with contextlib.closing(sqlite3.connect(dbfile)) as con:
            return {r[1] for r in con.execute(f"PRAGMA table_info({table})")}

    assert {"daily_token_limit", "requests_per_minute", "daily_spend_cap_usd"} <= (
        cols("dashboard_users")
    )
    assert {"daily_token_limit", "requests_per_minute"} <= cols("api_keys")
    assert "user_id" in cols("request_log")
    command.downgrade(cfg, "035_provider_max_concurrent")
    assert "user_id" not in cols("request_log")
    assert "daily_token_limit" not in cols("api_keys")
    command.upgrade(cfg, "head")
    assert "user_id" in cols("request_log")


# --- dashboard ---


def _as(user_id: int) -> dict[str, str]:
    from llama_router.dashboard.auth_core import AUTH_COOKIE, sign_session
    from llama_router.provider_secrets import session_signing_key

    return {"Cookie": f"{AUTH_COOKIE}={sign_session(user_id, session_signing_key())}"}


async def _dash_users(db: Database) -> tuple[int, int]:
    from llama_router.dashboard.auth_core import hash_password
    from llama_router.dashboard.middleware import (
        invalidate_dashboard_user_count_cache,
    )

    admin = await db.create_dashboard_user("adm", hash_password("pw-123456789"), True)
    ann = await db.create_dashboard_user("ann", hash_password("pw-123456789"), False)
    invalidate_dashboard_user_count_cache()
    return admin, ann


async def test_admin_sets_resets_and_defaults_user_limits(
    dash_client: tuple[httpx.AsyncClient, Database],
) -> None:
    c, db = dash_client
    admin, ann = await _dash_users(db)
    resp = await c.post(
        f"/users/{ann}/limits",
        data={
            "daily_token_limit": "5000",
            "requests_per_minute": "",
            "daily_spend_cap_usd": "2.5",
        },
        headers=_as(admin),
    )
    assert resp.status_code == 303
    row = await db.get_dashboard_user_by_id(ann)
    assert row is not None
    assert (row["daily_token_limit"], row["requests_per_minute"]) == (5000, None)
    assert row["daily_spend_cap_usd"] == 2.5
    page = await c.get("/users", headers=_as(admin))
    assert f'id="limits-{ann}"' in page.text
    assert "Default limits for users" in page.text

    resp = await c.post(
        "/users/quota-defaults",
        data={"requests_per_minute": "30"},
        headers=_as(admin),
    )
    assert resp.status_code == 303
    usage = (await c.get("/api/users/usage", headers=_as(admin))).json()
    mine = next(u for u in usage["users"] if u["id"] == ann)
    assert mine["limits"] == {
        "daily_token_limit": 5000,
        "requests_per_minute": 30,
        "daily_spend_cap_usd": 2.5,
    }

    resp = await c.post(f"/users/{ann}/limits", data={"reset": "1"}, headers=_as(admin))
    assert resp.status_code == 303
    row = await db.get_dashboard_user_by_id(ann)
    assert row is not None
    assert row["daily_token_limit"] is None

    bad = await c.post(
        f"/users/{ann}/limits", data={"daily_token_limit": "0"}, headers=_as(admin)
    )
    assert bad.headers["location"] == "/users?error=limits"
    # Viewers may not (the dashboard turns them away from admin pages).
    await db.set_dashboard_user_limits(ann, {"daily_token_limit": 7})
    denied = await c.post(f"/users/{ann}/limits", data={"reset": "1"}, headers=_as(ann))
    assert denied.status_code in (302, 403)
    row = await db.get_dashboard_user_by_id(ann)
    assert row is not None
    assert row["daily_token_limit"] == 7
    assert (await c.get("/api/users/usage", headers=_as(ann))).status_code == 403


async def test_owner_sees_usage_and_may_only_tighten_key_limits(
    dash_client: tuple[httpx.AsyncClient, Database],
) -> None:
    c, db = dash_client
    admin, ann = await _dash_users(db)
    await db.set_dashboard_user_limits(ann, {"daily_token_limit": 1000})
    key = (await c.post("/api/my-keys", json={}, headers=_as(ann))).json()
    mine = (await c.get("/api/my-keys", headers=_as(ann))).json()
    assert mine["quota"]["limits"]["daily_token_limit"] == 1000
    assert mine["quota"]["usage"]["tokens"] == 0
    assert mine["keys"][0]["tokens_today"] == 0

    ok = await c.patch(
        f"/api/my-keys/{key['id']}", json={"daily_token_limit": 500}, headers=_as(ann)
    )
    assert ok.status_code == 200, ok.text
    lower = await c.patch(
        f"/api/my-keys/{key['id']}", json={"daily_token_limit": 400}, headers=_as(ann)
    )
    assert lower.status_code == 200
    for raise_or_clear in (900, None):
        resp = await c.patch(
            f"/api/my-keys/{key['id']}",
            json={"daily_token_limit": raise_or_clear},
            headers=_as(ann),
        )
        assert resp.status_code == 403
    # An admin may raise or clear it.
    resp = await c.patch(
        f"/api/keys/{key['id']}",
        json={"daily_token_limit": None, "requests_per_minute": 12},
        headers=_as(admin),
    )
    assert resp.status_code == 200, resp.text
    found = await db.get_api_key(key["id"])
    assert found is not None
    assert (found["daily_token_limit"], found["requests_per_minute"]) == (None, 12)


async def test_dashboard_chat_counts_against_the_signed_in_user(
    dash_client: tuple[httpx.AsyncClient, Database],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from llama_router.dashboard import chat as chat_mod
    from llama_router.dashboard import deps as dash_deps

    c, db = dash_client
    _, ann = await _dash_users(db)
    pid = await _provider(db, "local")
    pm = dash_deps.get_pm()
    rt = Router(db, pm)
    pm.attach_router(rt)

    class Client:
        async def chat_completions(self, body: dict[str, Any]) -> dict[str, Any]:
            return {"choices": [{"message": {"content": "hey"}}], "usage": USAGE}

    monkeypatch.setattr(chat_mod, "get_v1_client", lambda pm, pid: Client())
    request_logger.start_log_writer()
    await db.set_dashboard_user_limits(ann, {"requests_per_minute": 1})
    body = {"model": "m", "content": "hello", "provider_id": pid}
    first = await c.post("/api/chat/send", json=body, headers=_as(ann))
    assert first.status_code == 200, first.text
    await request_logger.flush_request_log()
    rows = await db._select_all("SELECT user_id, api_key_id FROM request_log")
    assert [(r["user_id"], r["api_key_id"]) for r in rows] == [(ann, None)]
    assert await db.usage_since(_day_start(), user_id=ann) == (0.0, 50)
    second = await c.post("/api/chat/send", json=body, headers=_as(ann))
    assert second.status_code == 429
    await request_logger.stop_log_writer()


# --- review regressions ---


async def test_listing_models_is_not_charged(
    db: Database, v1: httpx.AsyncClient, up: Upstream
) -> None:
    await _provider(db, "local")
    uid = await _user(db, requests_per_minute=1, daily_token_limit=1)
    key, _ = await _key(db, uid)
    headers = {"Authorization": f"Bearer {key}"}
    for _ in range(3):
        assert (await v1.get("/v1/models", headers=headers)).status_code == 200


async def test_request_without_output_limit_runs_on_what_is_left(
    db: Database, v1: httpx.AsyncClient, up: Upstream
) -> None:
    await _provider(db, "local")
    uid = await _user(db, daily_token_limit=100)  # far below the 4096 guess
    key, _ = await _key(db, uid)
    body = {"model": "m", "messages": [{"role": "user", "content": "hi"}]}
    resp = await v1.post(CHAT, json=body, headers={"Authorization": f"Bearer {key}"})
    assert resp.status_code == 200, resp.text
    await request_logger.flush_request_log()
    assert tracker_for(db).usage(user_id=uid).tokens == 50


async def test_cancelled_request_counts_its_estimate(db: Database) -> None:
    from llama_router.request_logger import STATUS_CANCELLED, _build_entry

    uid = await _user(db)
    tracker = tracker_for(db)
    await tracker.user_tokens_committed(uid)
    held = await tracker.reserve_tokens(
        70, key_id=None, key_limit=None, user_id=uid, user_limit=1000
    )
    assert not isinstance(held, str)
    _build_entry(
        db,
        provider=None,
        protocol="v1",
        endpoint=CHAT,
        model="m",
        duration_ms=1.0,
        status=STATUS_CANCELLED,
        user_id=uid,
        token_reservation=held,
    )
    assert tracker.usage(user_id=uid).tokens == 70


async def test_decision_requests_hold_tokens(db: Database) -> None:
    from llama_router import decision_service as svc
    from llama_router import decisions as d
    from llama_router.router import RouteResult, RoutingPreferences
    from tests.test_decision_policy import INJECTION

    uid = await _user(db)
    prefs = RoutingPreferences(key_id=1, owner_user_id=uid, user_daily_token_limit=10)
    req = d.DecisionRequest.model_validate(
        {
            "model": "judge",
            "state": "x" * 400,
            "questions": {"a": INJECTION["question"]},
        }
    )

    class Rt:
        def price_for(self, pid: int, model: str) -> None:
            return None

    from llama_router.models import Provider

    result = RouteResult(Provider(id=1, name="p"), "judge")
    with pytest.raises(HTTPException) as err:
        await svc.hold_quota(db, Rt(), prefs, result, req)  # type: ignore[arg-type]
    assert err.value.status_code == 429
    prefs.user_daily_token_limit = 100_000
    held = await svc.hold_quota(db, Rt(), prefs, result, req)  # type: ignore[arg-type]
    assert held["token_reservation"].tokens > 0


async def test_refused_chat_turn_saves_nothing(
    dash_client: tuple[httpx.AsyncClient, Database],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from llama_router.dashboard import deps as dash_deps

    c, db = dash_client
    _, ann = await _dash_users(db)
    pid = await _provider(db, "local")
    pm = dash_deps.get_pm()
    pm.attach_router(Router(db, pm))
    await db.set_dashboard_user_limits(ann, {"daily_token_limit": 3})
    body = {"model": "m", "content": "hello there " * 10, "provider_id": pid}
    resp = await c.post("/api/chat/send", json=body, headers=_as(ann))
    assert resp.status_code == 429
    assert await db.list_chat_conversations(ann) == []


async def test_unlogged_decision_holds_are_released_at_request_end(
    db: Database,
) -> None:
    from llama_router.spend import (
        EXTRA_RESERVATIONS_STATE,
        SpendReservationMiddleware,
        keep_for_release,
    )

    uid = await _user(db)
    tracker = tracker_for(db)
    held = await tracker.reserve_tokens(
        90, key_id=None, key_limit=None, user_id=uid, user_limit=100
    )
    assert not isinstance(held, str)

    async def app(scope: Any, receive: Any, send: Any) -> None:
        from starlette.requests import Request

        keep_for_release(Request(scope), held)

    scope: dict[str, Any] = {"type": "http", "state": {}}
    await SpendReservationMiddleware(app)(scope, None, None)
    assert EXTRA_RESERVATIONS_STATE not in scope["state"]
    again = await tracker.reserve_tokens(
        90, key_id=None, key_limit=None, user_id=uid, user_limit=100
    )
    assert not isinstance(again, str)


async def test_logged_decision_holds_are_left_for_the_writer(db: Database) -> None:
    from starlette.requests import Request

    from llama_router.request_logger import log_request_background
    from llama_router.spend import EXTRA_RESERVATIONS_STATE, keep_for_release

    uid = await _user(db)
    tracker = tracker_for(db)
    held = await tracker.reserve_tokens(
        90, key_id=None, key_limit=None, user_id=uid, user_limit=100
    )
    assert not isinstance(held, str)
    request = Request({"type": "http", "state": {}, "headers": []})
    keep_for_release(request, held)
    request_logger.start_log_writer()
    log_request_background(
        db,
        provider=None,
        protocol="v1",
        endpoint="/v1/decisions",
        request=request,
        model="m",
        duration_ms=1.0,
        token_reservation=held,
    )
    assert getattr(request.state, EXTRA_RESERVATIONS_STATE) == []
    await request_logger.stop_log_writer()


async def test_dropped_log_row_frees_and_counts_its_token_hold(db: Database) -> None:
    from llama_router.request_logger import _charge_estimate

    uid = await _user(db)
    tracker = tracker_for(db)
    await tracker.user_tokens_committed(uid)
    held = await tracker.reserve_tokens(
        60, key_id=None, key_limit=None, user_id=uid, user_limit=100
    )
    assert not isinstance(held, str)
    _charge_estimate(db, {"token_reservation": held, "user_id": uid})
    assert held.released
    assert tracker.usage(user_id=uid).tokens == 60
    assert await tracker.user_tokens_committed(uid) == 60
