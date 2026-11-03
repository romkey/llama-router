"""Real signed WebAuthn ceremonies plus enrollment, recovery and bypass checks."""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field

import cbor2
import httpx
import pytest
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import ec
from fastapi import Request
from fastapi.responses import Response
from webauthn import base64url_to_bytes
from webauthn.helpers import bytes_to_base64url

from llama_router.config import Settings, settings
from llama_router.dashboard import mfa_recovery
from llama_router.dashboard.auth_core import AUTH_COOKIE, hash_password, sign_session
from llama_router.dashboard.mfa import PENDING_COOKIE
from llama_router.dashboard.mfa_store import MfaStore
from llama_router.dashboard.middleware import invalidate_dashboard_user_count_cache
from llama_router.database import Database

ORIGIN = "https://testserver"
HEADERS = {"X-Llama-Router-Security": "1", "Origin": ORIGIN}
PASSWORD = "test-password-12345"  # noqa: S105 - test fixture password


@dataclass
class Authenticator:
    """Minimal virtual authenticator: real COSE key, attestation and signatures."""

    credential_id: bytes = b"credential-one"
    key: ec.EllipticCurvePrivateKey = field(
        default_factory=lambda: ec.generate_private_key(ec.SECP256R1())
    )

    def response(
        self,
        challenge: str,
        *,
        register: bool,
        origin: str = ORIGIN,
        uv: bool = True,
        count: int = 1,
    ) -> str:
        client = json.dumps(
            {
                "type": "webauthn.create" if register else "webauthn.get",
                "challenge": challenge,
                "origin": origin,
                "crossOrigin": False,
            }
        ).encode()
        flags = 1 | (4 if uv else 0) | (64 if register else 0)
        auth_data = (
            hashlib.sha256(b"testserver").digest()
            + bytes([flags])
            + count.to_bytes(4, "big")
        )
        response = {"clientDataJSON": bytes_to_base64url(client)}
        if register:
            numbers = self.key.public_key().public_numbers()
            cose = cbor2.dumps(
                {
                    1: 2,
                    3: -7,
                    -1: 1,
                    -2: numbers.x.to_bytes(32, "big"),
                    -3: numbers.y.to_bytes(32, "big"),
                }
            )
            auth_data += (
                bytes(16)
                + len(self.credential_id).to_bytes(2, "big")
                + self.credential_id
                + cose
            )
            response["attestationObject"] = bytes_to_base64url(
                cbor2.dumps({"fmt": "none", "attStmt": {}, "authData": auth_data})
            )
        else:
            response["authenticatorData"] = bytes_to_base64url(auth_data)
            response["signature"] = bytes_to_base64url(
                self.key.sign(
                    auth_data + hashlib.sha256(client).digest(),
                    ec.ECDSA(hashes.SHA256()),
                )
            )
        return json.dumps(
            {
                "id": bytes_to_base64url(self.credential_id),
                "rawId": bytes_to_base64url(self.credential_id),
                "type": "public-key",
                "response": response,
            }
        )


@pytest.fixture
async def mfa_client(
    dash_client: tuple[httpx.AsyncClient, Database], monkeypatch: pytest.MonkeyPatch
) -> tuple[httpx.AsyncClient, Database]:
    monkeypatch.setattr(settings, "dashboard_webauthn_origin", ORIGIN)
    monkeypatch.setattr(settings, "dashboard_require_second_factor", False)
    client, db = dash_client
    client.base_url = httpx.URL(ORIGIN)
    return client, db


async def create_user(db: Database, name: str = "alice", *, admin: bool = True) -> int:
    uid = await db.create_dashboard_user(name, hash_password(PASSWORD), admin)
    invalidate_dashboard_user_count_cache()
    return uid


async def login(
    client: httpx.AsyncClient, name: str = "alice", next_url: str = "/"
) -> httpx.Response:
    return await client.post(
        "/login",
        data={"username": name, "password": PASSWORD, "next": next_url},
        headers={"Origin": ORIGIN},
    )


async def enroll(
    client: httpx.AsyncClient, authenticator: Authenticator, *, forced: bool = False
) -> httpx.Response:
    prefix = "/login/second-factor/register" if forced else "/account/security/register"
    start = await client.post(
        prefix + "/options", json={"name": "Laptop"}, headers=HEADERS
    )
    assert start.status_code == 200, start.text
    body = start.json()
    return await client.post(
        prefix + "/verify",
        json={
            "token": body["token"],
            "credential": authenticator.response(
                body["options"]["challenge"], register=True
            ),
        },
        headers=HEADERS,
    )


async def authenticate(
    client: httpx.AsyncClient,
    authenticator: Authenticator,
    *,
    origin: str = ORIGIN,
    uv: bool = True,
    count: int = 2,
) -> httpx.Response:
    start = await client.post(
        "/login/second-factor/authenticate/options", json={}, headers=HEADERS
    )
    assert start.status_code == 200, start.text
    body = start.json()
    assert body["options"]["userVerification"] == "required"
    credential = authenticator.response(
        body["options"]["challenge"],
        register=False,
        origin=origin,
        uv=uv,
        count=count,
    )
    return await client.post(
        "/login/second-factor/authenticate/verify",
        json={"token": body["token"], "credential": credential},
        headers=HEADERS,
    )


@pytest.mark.parametrize("admin", [True, False])
async def test_required_enrollment_blocks_dashboard_until_verified(
    mfa_client: tuple[httpx.AsyncClient, Database],
    monkeypatch: pytest.MonkeyPatch,
    admin: bool,
) -> None:
    client, db = mfa_client
    monkeypatch.setattr(settings, "dashboard_require_second_factor", True)
    uid = await create_user(db, admin=admin)
    response = await login(client)
    assert response.headers["location"] == "/login/second-factor"
    assert AUTH_COOKIE not in client.cookies
    assert (await client.get("/api/status")).status_code == 401
    assert (await client.get("/account/security")).status_code == 302
    assert (await client.get("/login/second-factor")).status_code == 200
    response = await enroll(client, Authenticator(), forced=True)
    assert response.status_code == 200, response.text
    codes = response.json()["recovery_codes"]
    assert len(codes) == 10
    assert len(set(codes)) == 10
    assert response.headers["cache-control"] == "no-store"
    assert PENDING_COOKIE not in client.cookies
    assert (await client.get("/api/status")).status_code == 200
    assert (await client.get("/account/security")).status_code == 200
    account = await MfaStore(db).account(uid)
    assert account is not None
    assert account.enabled
    rows = await db._select_all("SELECT code_hash FROM dashboard_recovery_codes")
    assert all(row["code_hash"] not in codes for row in rows)


async def test_optional_enrollment_and_real_authentication(
    mfa_client: tuple[httpx.AsyncClient, Database],
) -> None:
    client, db = mfa_client
    await create_user(db)
    assert (await login(client)).headers["location"] == "/"
    old_session = client.cookies[AUTH_COOKIE]
    authenticator = Authenticator()
    response = await enroll(client, authenticator)
    assert response.status_code == 200
    client.cookies.set(AUTH_COOKIE, old_session, domain="testserver.local", path="/")
    assert (await client.get("/api/status")).status_code == 401
    client.cookies.clear()
    assert (await login(client, next_url="/my-keys")).headers[
        "location"
    ] == "/login/second-factor"
    assert AUTH_COOKIE not in client.cookies
    assert (await client.get("/api/status")).status_code == 401
    response = await authenticate(client, authenticator)
    assert response.status_code == 200, response.text
    assert response.json()["next"] == "/my-keys"
    assert (await client.get("/api/status")).status_code == 200


@pytest.mark.parametrize("failure", ["origin", "uv", "key", "counter"])
async def test_rejects_invalid_signed_assertions(
    mfa_client: tuple[httpx.AsyncClient, Database], failure: str
) -> None:
    client, db = mfa_client
    await create_user(db)
    await login(client)
    authenticator = Authenticator()
    await enroll(client, authenticator)
    client.cookies.clear()
    await login(client)
    if failure == "origin":
        response = await authenticate(
            client, authenticator, origin="https://evil.example"
        )
    elif failure == "uv":
        response = await authenticate(client, authenticator, uv=False)
    elif failure == "counter":
        response = await authenticate(client, authenticator, count=1)
    else:
        response = await authenticate(client, Authenticator())
    assert response.status_code == 400
    assert AUTH_COOKIE not in client.cookies
    assert (await client.get("/api/status")).status_code == 401


async def test_recovery_is_single_use_and_still_requires_password(
    mfa_client: tuple[httpx.AsyncClient, Database],
) -> None:
    client, db = mfa_client
    await create_user(db)
    await login(client)
    codes = (await enroll(client, Authenticator())).json()["recovery_codes"]
    client.cookies.clear()
    response = await client.post(
        "/login/second-factor/recovery", json={"code": codes[0]}, headers=HEADERS
    )
    assert response.status_code == 401
    await login(client)
    response = await client.post(
        "/login/second-factor/recovery", json={"code": codes[0]}, headers=HEADERS
    )
    assert response.status_code == 200
    client.cookies.clear()
    await login(client)
    response = await client.post(
        "/login/second-factor/recovery", json={"code": codes[0]}, headers=HEADERS
    )
    assert response.status_code == 401
    assert AUTH_COOKIE not in client.cookies


async def test_enforcement_rejects_old_password_sessions_but_exempts_sso(
    mfa_client: tuple[httpx.AsyncClient, Database], monkeypatch: pytest.MonkeyPatch
) -> None:
    client, db = mfa_client
    await create_user(db)
    await login(client)
    monkeypatch.setattr(settings, "dashboard_require_second_factor", True)
    assert (await client.get("/api/status")).status_code == 401
    oidc_uid = await db.create_oidc_dashboard_user(
        "sso",
        "https://idp.example",
        "subject",
        email=None,
        display_name=None,
        is_admin=True,
    )
    client.cookies.clear()
    client.cookies.set(AUTH_COOKIE, sign_session(oidc_uid, settings.session_secret))
    assert (await client.get("/api/status")).status_code == 200
    assert (await client.get("/profile")).status_code == 200
    assert "single sign-on provider" in (await client.get("/profile")).text


async def test_challenges_are_bound_to_pending_login_and_consumed(
    mfa_client: tuple[httpx.AsyncClient, Database],
) -> None:
    client, db = mfa_client
    await create_user(db)
    await login(client)
    authenticator = Authenticator()
    await enroll(client, authenticator)
    client.cookies.clear()
    await login(client)
    first_pending = client.cookies[PENDING_COOKIE]
    start = (
        await client.post("/login/second-factor/authenticate/options", headers=HEADERS)
    ).json()
    body = {
        "token": start["token"],
        "credential": authenticator.response(
            start["options"]["challenge"], register=False, count=2
        ),
    }
    stored = await db._select_all("SELECT token_hash, state FROM dashboard_auth_flows")
    assert first_pending not in str(stored)
    assert start["token"] not in str(stored)
    await login(client)
    response = await client.post(
        "/login/second-factor/authenticate/verify", json=body, headers=HEADERS
    )
    assert response.status_code == 403
    client.cookies.clear()
    client.cookies.set(PENDING_COOKIE, first_pending)
    response = await client.post(
        "/login/second-factor/authenticate/verify", json=body, headers=HEADERS
    )
    assert response.status_code == 200
    client.cookies.clear()
    client.cookies.set(PENDING_COOKIE, first_pending)
    response = await client.post(
        "/login/second-factor/authenticate/verify", json=body, headers=HEADERS
    )
    assert response.status_code == 401


async def test_csrf_and_management_keys_cannot_enroll(
    mfa_client: tuple[httpx.AsyncClient, Database],
) -> None:
    client, db = mfa_client
    await create_user(db)
    await login(client)
    for headers in ({}, {**HEADERS, "Origin": "https://evil.example"}):
        response = await client.post(
            "/account/security/register/options", json={}, headers=headers
        )
        assert response.status_code == 403
    client.cookies.clear()
    response = await client.post(
        "/account/security/register/options",
        json={},
        headers={**HEADERS, "Authorization": "Bearer lrm_invalid"},
    )
    assert response.status_code == 401


async def test_older_session_can_manage_factors_until_password_changes(
    mfa_client: tuple[httpx.AsyncClient, Database], monkeypatch: pytest.MonkeyPatch
) -> None:
    client, db = mfa_client
    uid = await create_user(db)
    await login(client)
    original_time = time.time
    with monkeypatch.context() as clock_patch:
        clock_patch.setattr(time, "time", lambda: original_time() + 301)
        response = await client.post(
            "/account/security/register/options", json={}, headers=HEADERS
        )
        assert response.status_code == 200
    await db.update_dashboard_user_password(uid, hash_password("new-password"))
    assert (await client.get("/api/status")).status_code == 401


async def test_reset_revokes_sessions_and_forces_reenrollment(
    mfa_client: tuple[httpx.AsyncClient, Database], monkeypatch: pytest.MonkeyPatch
) -> None:
    client, db = mfa_client
    uid = await create_user(db)
    await login(client)
    await enroll(client, Authenticator())
    monkeypatch.setattr(mfa_recovery, "Database", lambda: Database(db._url))
    assert await mfa_recovery.reset_second_factor("alice") == 0
    assert (await client.get("/api/status")).status_code == 401
    assert not await MfaStore(db).credentials(uid)
    monkeypatch.setattr(settings, "dashboard_require_second_factor", True)
    client.cookies.clear()
    await login(client)
    assert AUTH_COOKIE not in client.cookies
    assert (
        "Set up your second factor" in (await client.get("/login/second-factor")).text
    )
    assert await mfa_recovery.reset_second_factor("unknown") == 1


async def test_expired_pending_and_cross_user_enrollment_rejected(
    mfa_client: tuple[httpx.AsyncClient, Database], monkeypatch: pytest.MonkeyPatch
) -> None:
    client, db = mfa_client
    monkeypatch.setattr(settings, "dashboard_require_second_factor", True)
    await create_user(db)
    await create_user(db, "bob")
    await login(client)
    start = (
        await client.post(
            "/login/second-factor/register/options", json={}, headers=HEADERS
        )
    ).json()
    body = {
        "token": start["token"],
        "credential": Authenticator().response(
            start["options"]["challenge"], register=True
        ),
    }
    await login(client, "bob")
    response = await client.post(
        "/login/second-factor/register/verify", json=body, headers=HEADERS
    )
    assert response.status_code == 403
    await db._execute("UPDATE dashboard_auth_flows SET expires_at = 0")
    response = await client.post(
        "/login/second-factor/register/options", json={}, headers=HEADERS
    )
    assert response.status_code == 401


async def test_can_add_replacement_but_never_remove_last_authenticator(
    mfa_client: tuple[httpx.AsyncClient, Database],
) -> None:
    client, db = mfa_client
    await create_user(db)
    await login(client)
    first = Authenticator()
    await enroll(client, first)
    first_id = bytes_to_base64url(first.credential_id)
    response = await client.delete(
        "/account/security/credentials/" + first_id, headers=HEADERS
    )
    assert response.status_code == 400
    second = Authenticator(credential_id=b"credential-two")
    response = await enroll(client, second)
    assert response.status_code == 200
    assert response.json()["recovery_codes"] == []
    response = await client.delete(
        "/account/security/credentials/" + first_id, headers=HEADERS
    )
    assert response.status_code == 200
    assert (
        base64url_to_bytes((await MfaStore(db).credentials(1))[0].id)
        == second.credential_id
    )


@pytest.mark.parametrize(
    "origin",
    [
        "http://router.example",
        "https://127.0.0.1",
        "https://router.example/path",
        "https://user@router.example",
        "https://router.example?x=1",
    ],
)
def test_rejects_invalid_origins(origin: str) -> None:
    with pytest.raises(ValueError, match="LLAMA_ROUTER_DASHBOARD_WEBAUTHN_ORIGIN"):
        Settings(dashboard_webauthn_origin=origin)


def test_enforcement_requires_origin() -> None:
    with pytest.raises(
        ValueError, match="LLAMA_ROUTER_DASHBOARD_REQUIRE_SECOND_FACTOR"
    ):
        Settings(dashboard_require_second_factor=True)
    assert Settings(
        dashboard_webauthn_origin="http://localhost:8000"
    ).dashboard_webauthn_origin


async def test_new_recovery_codes_invalidate_old_codes(
    mfa_client: tuple[httpx.AsyncClient, Database],
) -> None:
    client, db = mfa_client
    await create_user(db)
    await login(client)
    old_codes = (await enroll(client, Authenticator())).json()["recovery_codes"]
    response = await client.post("/account/security/recovery-codes", headers=HEADERS)
    assert response.status_code == 200
    new_codes = response.json()["recovery_codes"]
    assert not set(new_codes).intersection(old_codes)
    client.cookies.clear()
    await login(client)
    response = await client.post(
        "/login/second-factor/recovery", json={"code": old_codes[0]}, headers=HEADERS
    )
    assert response.status_code == 401
    await login(client)
    response = await client.post(
        "/login/second-factor/recovery", json={"code": new_codes[0]}, headers=HEADERS
    )
    assert response.status_code == 200


async def test_recovery_code_cannot_be_used_concurrently(
    mfa_client: tuple[httpx.AsyncClient, Database],
) -> None:
    import asyncio

    client, db = mfa_client
    await create_user(db)
    await login(client)
    code = (await enroll(client, Authenticator())).json()["recovery_codes"][0]
    client.cookies.clear()
    await login(client)
    responses = await asyncio.gather(
        client.post(
            "/login/second-factor/recovery", json={"code": code}, headers=HEADERS
        ),
        client.post(
            "/login/second-factor/recovery", json={"code": code}, headers=HEADERS
        ),
    )
    assert sorted(r.status_code for r in responses) == [200, 401]


@pytest.mark.parametrize("failure", ["origin", "uv", "malformed"])
async def test_bad_registration_never_enables_mfa(
    mfa_client: tuple[httpx.AsyncClient, Database], failure: str
) -> None:
    client, db = mfa_client
    uid = await create_user(db)
    await login(client)
    start = (
        await client.post(
            "/account/security/register/options", json={}, headers=HEADERS
        )
    ).json()
    credential = Authenticator().response(
        start["options"]["challenge"],
        register=True,
        origin="https://evil.example" if failure == "origin" else ORIGIN,
        uv=failure != "uv",
    )
    response = await client.post(
        "/account/security/register/verify",
        json={
            "token": start["token"],
            "credential": "[]" if failure == "malformed" else credential,
        },
        headers=HEADERS,
    )
    assert response.status_code == 400
    account = await MfaStore(db).account(uid)
    assert account is not None
    assert not account.enabled
    assert not await MfaStore(db).credentials(uid)


def test_origin_normalizes_default_port_and_hostname() -> None:
    assert (
        Settings(
            dashboard_webauthn_origin="https://Router.Example:443"
        ).dashboard_webauthn_origin
        == "https://router.example"
    )


async def test_session_revocation_invalidates_mfa_sessions_and_pending_login(
    mfa_client: tuple[httpx.AsyncClient, Database],
) -> None:
    client, db = mfa_client
    uid = await create_user(db)
    await login(client)
    authenticator = Authenticator()
    assert (await enroll(client, authenticator)).status_code == 200
    await db.bump_dashboard_user_session_version(uid)
    assert (await client.get("/api/status")).status_code == 401
    client.cookies.clear()
    await login(client)
    assert (await authenticate(client, authenticator)).status_code == 200
    assert (await client.get("/api/status")).status_code == 200
    client.cookies.clear()
    await login(client)
    start = await client.post(
        "/login/second-factor/authenticate/options", headers=HEADERS
    )
    assert start.status_code == 200
    body = start.json()
    await db.bump_dashboard_user_session_version(uid)
    response = await client.post(
        "/login/second-factor/authenticate/verify",
        json={
            "token": body["token"],
            "credential": authenticator.response(
                body["options"]["challenge"], register=False, count=3
            ),
        },
        headers=HEADERS,
    )
    assert response.status_code == 401


async def test_own_password_change_keeps_verified_session_and_revokes_old_cookie(
    mfa_client: tuple[httpx.AsyncClient, Database],
) -> None:
    client, db = mfa_client
    uid = await create_user(db)
    await login(client)
    await enroll(client, Authenticator())
    old_cookie = client.cookies[AUTH_COOKIE]
    response = await client.post(
        f"/users/{uid}/password", data={"new_password": PASSWORD}
    )
    assert response.status_code == 303
    assert (await client.get("/api/status")).status_code == 200
    assert (await client.get("/account/security")).status_code == 200
    response = await client.get(
        "/api/status", headers={"Cookie": f"{AUTH_COOKIE}={old_cookie}"}
    )
    assert response.status_code == 401


async def test_password_change_during_login_invalidates_password_proof(
    mfa_client: tuple[httpx.AsyncClient, Database], monkeypatch: pytest.MonkeyPatch
) -> None:
    from llama_router.dashboard import mfa

    client, db = mfa_client
    uid = await create_user(db)
    original = mfa.password_login_response

    async def changed_password(
        request: Request, user_id: int, next_url: str, *, session_version: int
    ) -> Response:
        await db.update_dashboard_user_password(uid, hash_password("new-password"))
        return await original(
            request, user_id, next_url, session_version=session_version
        )

    monkeypatch.setattr(mfa, "password_login_response", changed_password)
    assert (await login(client)).status_code == 401
    assert AUTH_COOKIE not in client.cookies
    assert PENDING_COOKIE not in client.cookies


@pytest.mark.parametrize("admin", [True, False])
async def test_profile_is_available_to_local_admins_and_viewers(
    mfa_client: tuple[httpx.AsyncClient, Database], admin: bool
) -> None:
    client, db = mfa_client
    await create_user(db, admin=admin)
    await login(client)
    response = await client.get("/profile")
    assert response.status_code == 200
    assert 'href="/profile" aria-label="Account for alice"' in response.text
    assert 'href="/account/security"' not in response.text


async def test_password_login_from_another_origin_explains_itself(
    mfa_client: tuple[httpx.AsyncClient, Database],
) -> None:
    client, db = mfa_client
    await create_user(db)
    # Same-origin as far as the browser goes, but not the configured address.
    client.base_url = httpx.URL("http://10.0.0.5:8000")
    resp = await client.post(
        "/login",
        data={"username": "alice", "password": PASSWORD, "next": "/chat"},
        headers={"Origin": "http://10.0.0.5:8000"},
    )
    assert resp.status_code == 303
    assert resp.headers["location"] == "/login?error=origin&next=%2Fchat"
    assert AUTH_COOKIE not in resp.cookies
    page = await client.get(resp.headers["location"])
    assert ORIGIN in page.text
    assert "different one" in page.text
