"""Session revocation, sign-in throttling, cross-site checks and cookie flags."""

from __future__ import annotations

import time

import httpx
import pytest
from itsdangerous import URLSafeTimedSerializer

from llama_router.config import settings
from llama_router.dashboard import auth_core, login_throttle
from llama_router.dashboard.auth_core import (
    AUTH_COOKIE,
    OIDC_SESSION_MAX_AGE,
    hash_password,
    sign_session,
    unsign_session,
)
from llama_router.dashboard.login_throttle import LoginThrottle
from llama_router.dashboard.middleware import invalidate_dashboard_user_count_cache
from llama_router.database import Database

DashClient = tuple[httpx.AsyncClient, Database]
PASSWORD = "password-12345"  # noqa: S105  # test fixture value


def _secret() -> str:
    return settings.session_secret.strip()


def _cookie(token: str) -> dict[str, str]:
    return {"Cookie": f"{AUTH_COOKIE}={token}"}


async def _user(db: Database, name: str = "alice", admin: bool = True) -> int:
    uid = await db.create_dashboard_user(name, hash_password(PASSWORD), admin)
    invalidate_dashboard_user_count_cache()
    return uid


async def _sign_in(client: httpx.AsyncClient, name: str = "alice") -> str:
    r = await client.post(
        "/login", data={"username": name, "password": PASSWORD, "next": "/"}
    )
    assert r.status_code == 303, r.text
    token = r.cookies[AUTH_COOKIE]
    client.cookies.clear()
    return token


async def _status(client: httpx.AsyncClient, token: str) -> int:
    client.cookies.clear()
    return (await client.get("/api/status", headers=_cookie(token))).status_code


# --- Session tokens ---


def test_token_round_trip() -> None:
    tok = unsign_session(sign_session(7, "s" * 32, version=3, username="bob"), "s" * 32)
    assert tok is not None
    assert (tok.user_id, tok.version, tok.username) == (7, 3, "bob")
    assert abs(tok.issued_at - time.time()) < 5


def test_token_without_version_is_refused() -> None:
    """Cookies from before session versions must sign in again."""
    old = URLSafeTimedSerializer("s" * 32, salt="lr-dash-v1").dumps({"u": 1})
    assert unsign_session(old, "s" * 32) is None


def test_valid_for_checks_version_name_approval_and_age() -> None:
    tok = unsign_session(sign_session(1, "s" * 32, version=2, username="a"), "s" * 32)
    assert tok is not None
    row = {"username": "a", "session_version": 2, "approved": True}
    assert tok.valid_for(row)
    assert not tok.valid_for({**row, "session_version": 3})
    assert not tok.valid_for({**row, "username": "b"})
    assert not tok.valid_for({**row, "approved": False})
    later = tok.issued_at + OIDC_SESSION_MAX_AGE + 1
    assert tok.valid_for({**row, "auth_source": "local"}, now=later)
    assert not tok.valid_for({**row, "auth_source": "oidc"}, now=later)


async def test_login_cookie_carries_version_and_name(dash_client: DashClient) -> None:
    client, db = dash_client
    uid = await _user(db)
    await db.bump_dashboard_user_session_version(uid)
    tok = unsign_session(await _sign_in(client), _secret())
    assert tok is not None
    assert (tok.user_id, tok.version, tok.username) == (uid, 1, "alice")


async def test_password_change_revokes_other_sessions(dash_client: DashClient) -> None:
    client, db = dash_client
    admin = await _user(db, "root")
    victim = await _user(db, "alice", admin=False)
    alice = await _sign_in(client, "alice")
    root = await _sign_in(client, "root")
    assert await _status(client, alice) == 200
    r = await client.post(
        f"/users/{victim}/password",
        data={"new_password": "another-pass-1"},
        headers=_cookie(root),
    )
    assert r.status_code == 303
    assert await _status(client, alice) == 401
    assert await _status(client, root) == 200
    assert admin != victim


async def test_own_password_change_keeps_this_session(dash_client: DashClient) -> None:
    client, db = dash_client
    uid = await _user(db)
    first = await _sign_in(client)
    second = await _sign_in(client)
    r = await client.post(
        f"/users/{uid}/password",
        data={"new_password": "another-pass-1"},
        headers=_cookie(first),
    )
    assert r.status_code == 303
    assert await _status(client, r.cookies[AUTH_COOKIE]) == 200
    assert await _status(client, second) == 401


async def test_sign_out_everywhere(dash_client: DashClient) -> None:
    client, db = dash_client
    await _user(db, "alice", admin=False)
    phone = await _sign_in(client)
    laptop = await _sign_in(client)
    r = await client.post("/logout/everywhere", headers=_cookie(laptop))
    assert r.status_code == 303
    assert r.headers["location"] == "/login"
    assert await _status(client, phone) == 401
    assert await _status(client, laptop) == 401
    assert await _status(client, await _sign_in(client)) == 200


async def test_admin_signs_out_another_account(dash_client: DashClient) -> None:
    client, db = dash_client
    await _user(db, "root")
    bob = await _user(db, "bob", admin=False)
    bob_token = await _sign_in(client, "bob")
    root = await _sign_in(client, "root")
    r = await client.post(f"/users/{bob}/sign-out", headers=_cookie(root))
    assert r.status_code == 303
    assert await _status(client, bob_token) == 401
    r = await client.post(f"/users/{bob}/sign-out", headers=_cookie(bob_token))
    assert r.status_code == 302
    assert r.headers["location"].startswith("/login")
    r = await client.post("/users/999/sign-out", headers=_cookie(root))
    assert r.status_code == 404


async def test_viewer_cannot_sign_out_others(dash_client: DashClient) -> None:
    client, db = dash_client
    root = await _user(db, "root")
    await _user(db, "bob", admin=False)
    bob = await _sign_in(client, "bob")
    r = await client.post(f"/users/{root}/sign-out", headers=_cookie(bob))
    assert r.status_code == 302
    assert "forbidden" in r.headers["location"]


async def test_reused_id_does_not_inherit_session(dash_client: DashClient) -> None:
    client, db = dash_client
    await _user(db, "root")
    uid = await _user(db, "gone", admin=False)
    token = sign_session(uid, _secret(), username="gone")
    await db.delete_dashboard_user(uid)
    # SQLite hands the highest deleted id to the next row.
    new = await _user(db, "newcomer", admin=False)
    assert new == uid
    assert await _status(client, token) == 401


async def test_recreated_account_does_not_inherit_session(
    dash_client: DashClient,
) -> None:
    """Same id and same username: the old cookie still must not work."""
    client, db = dash_client
    await _user(db, "root")
    uid = await _user(db, "gone", admin=False)
    token = await _sign_in(client, "gone")
    await db.delete_dashboard_user(uid)
    again = await _user(db, "gone", admin=False)
    assert again == uid
    assert await _status(client, token) == 401
    assert await _status(client, await _sign_in(client, "gone")) == 200


async def test_pending_admin_is_not_counted_as_an_admin(db: Database) -> None:
    await db.create_dashboard_user("root", hash_password(PASSWORD), True)
    pending = await db.create_oidc_dashboard_user(
        "sso",
        "https://idp",
        "s1",
        email=None,
        display_name=None,
        is_admin=True,
        approved=False,
    )
    assert await db.count_dashboard_admins() == 1
    await db.approve_dashboard_user(pending)
    assert await db.count_dashboard_admins() == 2


# --- Sign-in throttling ---


def test_throttle_backs_off_exponentially() -> None:
    t = LoginThrottle()
    keys = ["user:a"]
    for _ in range(login_throttle.FREE_ATTEMPTS - 1):
        t.failure(keys, now=0.0)
    assert t.retry_after(keys, now=0.0) == 0
    t.failure(keys, now=0.0)
    assert t.retry_after(keys, now=0.0) == pytest.approx(1.0)
    t.failure(keys, now=2.0)
    assert t.retry_after(keys, now=2.0) == pytest.approx(2.0)
    for i in range(20):
        t.failure(keys, now=10.0 + i)
    assert t.retry_after(keys, now=29.0) <= login_throttle.MAX_DELAY_S
    t.success(keys)
    assert t.retry_after(keys, now=29.0) == 0


def test_throttle_survives_thousands_of_failures() -> None:
    t = LoginThrottle()
    for _ in range(2000):
        t.failure(["ip:x"], now=0.0)
    assert t.retry_after(["ip:x"], now=0.0) == login_throttle.MAX_DELAY_S


async def test_invalid_usernames_are_throttled_too(dash_client: DashClient) -> None:
    client, db = dash_client
    await _user(db)
    bad = {"username": "!", "password": "x", "next": "/"}
    for _ in range(login_throttle.FREE_ATTEMPTS):
        assert (await client.post("/login", data=bad)).status_code == 303
    assert (await client.post("/login", data=bad)).status_code == 429


def test_throttle_forgets_old_failures() -> None:
    t = LoginThrottle()
    for _ in range(login_throttle.FREE_ATTEMPTS):
        t.failure(["k"], now=0.0)
    t.failure(["k"], now=login_throttle.FORGET_AFTER_S + 10)
    assert t.retry_after(["k"], now=login_throttle.FORGET_AFTER_S + 10) == 0


def test_throttle_memory_is_bounded(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(login_throttle, "MAX_KEYS", 10)
    t = LoginThrottle()
    for i in range(50):
        t.failure([f"user:{i}"], now=float(i))
    assert len(t._entries) <= 10
    assert "user:49" in t._entries


async def test_repeated_failures_get_429(dash_client: DashClient) -> None:
    client, db = dash_client
    await _user(db)
    bad = {"username": "alice", "password": "wrong-password", "next": "/"}
    for _ in range(login_throttle.FREE_ATTEMPTS):
        r = await client.post("/login", data=bad)
        assert r.status_code == 303
    r = await client.post("/login", data=bad)
    assert r.status_code == 429
    assert int(r.headers["retry-after"]) >= 1
    assert "Too many failed sign-in attempts" in r.text
    # Even the right password waits out the lock.
    good = {**bad, "password": PASSWORD}
    assert (await client.post("/login", data=good)).status_code == 429
    login_throttle.throttle.clear()
    assert (await client.post("/login", data=good)).status_code == 303


async def test_forged_forwarded_for_does_not_dodge_the_address_limit(
    dash_client: DashClient,
) -> None:
    client, db = dash_client
    await _user(db)
    for i in range(login_throttle.FREE_ATTEMPTS):
        r = await client.post(
            "/login",
            data={"username": f"guess{i}", "password": "x-wrong-1", "next": "/"},
            headers={"X-Forwarded-For": f"203.0.113.{i}"},
        )
        assert r.status_code == 303
    r = await client.post(
        "/login",
        data={"username": "alice", "password": PASSWORD, "next": "/"},
        headers={"X-Forwarded-For": "198.51.100.7"},
    )
    assert r.status_code == 429


async def test_unknown_user_costs_a_password_check(
    dash_client: DashClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    client, db = dash_client
    await _user(db)
    burned: list[str] = []
    real = auth_core.burn_password_check

    def spy(password: str) -> None:
        burned.append(password)
        real(password)

    monkeypatch.setattr("llama_router.dashboard.routes.burn_password_check", spy)
    r = await client.post(
        "/login", data={"username": "nobody", "password": "guess-1", "next": "/"}
    )
    assert r.status_code == 303
    assert "error=invalid" in r.headers["location"]
    assert burned == ["guess-1"]


# --- Cross-site requests ---


@pytest.mark.parametrize(
    ("headers", "allowed"),
    [
        ({}, True),
        ({"Sec-Fetch-Site": "same-origin"}, True),
        ({"Sec-Fetch-Site": "cross-site"}, False),
        ({"Sec-Fetch-Site": "same-site"}, False),
        ({"Origin": "http://testserver"}, True),
        ({"Origin": "http://testserver:80"}, True),
        ({"Origin": "https://evil.example"}, False),
        ({"Origin": "http://testserver:3000"}, False),
        ({"Origin": "null"}, False),
        ({"Origin": "https://llm.example", "X-Forwarded-Host": "llm.example"}, True),
    ],
)
async def test_cross_site_login_post(
    dash_client: DashClient, headers: dict[str, str], allowed: bool
) -> None:
    client, db = dash_client
    await _user(db)
    r = await client.post(
        "/login",
        data={"username": "alice", "password": PASSWORD, "next": "/"},
        headers=headers,
    )
    if allowed:
        assert r.status_code == 303
    else:
        assert r.status_code == 403
        assert r.json() == {"detail": "Cross-site request refused"}


async def test_cross_site_refused_on_dashboard_posts(dash_client: DashClient) -> None:
    client, db = dash_client
    uid = await _user(db)
    token = await _sign_in(client)
    r = await client.post(
        f"/users/{uid}/delete",
        headers={**_cookie(token), "Sec-Fetch-Site": "cross-site"},
    )
    assert r.status_code == 403
    assert await db.get_dashboard_user_by_id(uid) is not None
    # Reads are not affected.
    r = await client.get(
        "/", headers={**_cookie(token), "Sec-Fetch-Site": "cross-site"}
    )
    assert r.status_code == 200


async def test_trusted_origin_allowed(
    dash_client: DashClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    client, db = dash_client
    monkeypatch.setattr(
        settings, "dashboard_trusted_origins", "https://a.example, https://llm.example"
    )
    await _user(db)
    r = await client.post(
        "/login",
        data={"username": "alice", "password": PASSWORD, "next": "/"},
        headers={"Origin": "https://llm.example", "Sec-Fetch-Site": "cross-site"},
    )
    assert r.status_code == 303


# --- Cookie flags ---


async def test_cookie_secure_behind_https_proxy(dash_client: DashClient) -> None:
    client, db = dash_client
    await _user(db)
    data = {"username": "alice", "password": PASSWORD, "next": "/"}
    r = await client.post("/login", data=data)
    assert "secure" not in r.headers["set-cookie"].lower()
    r = await client.post("/login", data=data, headers={"X-Forwarded-Proto": "https"})
    assert "secure" in r.headers["set-cookie"].lower()


async def test_cookie_secure_when_configured(
    dash_client: DashClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    client, db = dash_client
    monkeypatch.setattr(settings, "dashboard_cookie_secure", True)
    await _user(db)
    r = await client.post(
        "/login", data={"username": "alice", "password": PASSWORD, "next": "/"}
    )
    assert "secure" in r.headers["set-cookie"].lower()
