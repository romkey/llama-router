"""OpenID Connect dashboard sign-in against a fake identity provider."""

from __future__ import annotations

import base64
import json
import time
from typing import Any
from urllib.parse import parse_qs, urlparse

import httpx
import pytest
from joserfc import jwt
from joserfc.jwk import KeySet, RSAKey

from llama_router.config import settings
from llama_router.dashboard import oidc
from llama_router.dashboard.auth_core import AUTH_COOKIE, hash_password
from llama_router.dashboard.middleware import invalidate_dashboard_user_count_cache
from tests.test_dashboard_auth import dash_client  # noqa: F401  (fixture)

ISSUER = "https://idp.example"
CLIENT_ID = "llama-router"
CLIENT_SECRET = "s3cret"


class FakeIdP:
    """Discovery, JWKS, token and userinfo endpoints for one test."""

    def __init__(self) -> None:
        self.key = RSAKey.generate_key(2048, parameters={"kid": "k1"})
        self.published = [self.key]
        self.signing_key = self.key
        self.claims: dict[str, Any] = {}
        self.userinfo: dict[str, Any] | None = None
        self.nonce = ""
        self.challenge = ""
        self.token_status = 200
        self.discovery_status = 200
        self.extra_metadata: dict[str, Any] = {}
        self.requests: list[httpx.Request] = []
        self.id_token_override: str | None = None

    def metadata(self) -> dict[str, Any]:
        return {
            "issuer": ISSUER,
            "authorization_endpoint": ISSUER + "/authorize",
            "token_endpoint": ISSUER + "/token",
            "jwks_uri": ISSUER + "/jwks",
            "userinfo_endpoint": ISSUER + "/userinfo",
            "end_session_endpoint": ISSUER + "/logout",
            "id_token_signing_alg_values_supported": ["RS256"],
            **self.extra_metadata,
        }

    def id_token(self) -> str:
        now = int(time.time())
        claims = {
            "iss": ISSUER,
            "aud": CLIENT_ID,
            "sub": "user-1",
            "iat": now,
            "exp": now + 300,
            "nonce": self.nonce,
            "preferred_username": "Alice",
            "email": "alice@example.org",
            "name": "Alice Example",
        }
        claims.update(self.claims)
        claims = {k: v for k, v in claims.items() if v is not None}
        kid = self.signing_key.kid
        return jwt.encode({"alg": "RS256", "kid": kid}, claims, self.signing_key)

    def handle(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        path = request.url.path
        if path == "/.well-known/openid-configuration":
            return httpx.Response(self.discovery_status, json=self.metadata())
        if path == "/jwks":
            return httpx.Response(
                200, json=KeySet(self.published).as_dict(private=False)
            )
        if path == "/token":
            return self._token(request)
        if path == "/userinfo":
            if request.headers.get("authorization") != "Bearer at-1":
                return httpx.Response(401)
            return httpx.Response(200, json=self.userinfo or {"sub": "user-1"})
        return httpx.Response(404)

    def _token(self, request: httpx.Request) -> httpx.Response:
        form = parse_qs(request.content.decode())
        basic = base64.b64encode(f"{CLIENT_ID}:{CLIENT_SECRET}".encode()).decode()
        if request.headers.get("authorization") != f"Basic {basic}":
            return httpx.Response(401, json={"error": "invalid_client"})
        verifier = form["code_verifier"][0]
        if oidc.pkce_challenge(verifier) != self.challenge:
            return httpx.Response(400, json={"error": "invalid_grant"})
        if form["code"][0] != "code-1":
            return httpx.Response(400, json={"error": "invalid_grant"})
        if not form["redirect_uri"][0].endswith(oidc.CALLBACK_PATH):
            return httpx.Response(400, json={"error": "invalid_grant"})
        if self.token_status != 200:
            return httpx.Response(self.token_status, json={"error": "server_error"})
        return httpx.Response(
            200,
            json={
                "access_token": "at-1",
                "token_type": "Bearer",
                "id_token": self.id_token_override or self.id_token(),
            },
        )


@pytest.fixture
def idp(httpx_mock, monkeypatch) -> FakeIdP:
    fake = FakeIdP()
    httpx_mock.add_callback(fake.handle, is_reusable=True, is_optional=True)
    monkeypatch.setattr(settings, "oidc_issuer", ISSUER)
    monkeypatch.setattr(settings, "oidc_client_id", CLIENT_ID)
    monkeypatch.setattr(settings, "oidc_client_secret", CLIENT_SECRET)
    monkeypatch.setattr(settings, "oidc_admin_group", "router-admins")
    monkeypatch.setattr(settings, "oidc_allowed_group", "")
    monkeypatch.setattr(settings, "oidc_groups_claim", "groups")
    monkeypatch.setattr(settings, "oidc_redirect_base_url", "")
    monkeypatch.setattr(settings, "oidc_only", False)
    monkeypatch.setattr(settings, "oidc_logout_at_idp", False)
    oidc.reset_cache()
    yield fake
    oidc.reset_cache()


async def _start(client, next_url: str = "/") -> tuple[httpx.Response, dict]:
    r = await client.get("/login/oidc", params={"next": next_url})
    assert r.status_code == 302, r.text
    q = {k: v[0] for k, v in parse_qs(urlparse(r.headers["location"]).query).items()}
    return r, q


async def _login(client, idp: FakeIdP, next_url: str = "/", **claims):
    _, q = await _start(client, next_url)
    idp.nonce = q["nonce"]
    idp.challenge = q["code_challenge"]
    idp.claims = claims
    return await client.get(
        oidc.CALLBACK_PATH, params={"code": "code-1", "state": q["state"]}
    )


async def _seed_local_admin(db) -> int:
    uid = await db.create_dashboard_user("root", hash_password("password123"), True)
    invalidate_dashboard_user_count_cache()
    return uid


# --- Pure helpers ---


def _cfg(**kw) -> oidc.OIDCConfig:
    base = dict(
        issuer=ISSUER,
        client_id=CLIENT_ID,
        client_secret=CLIENT_SECRET,
        scopes="openid profile email groups",
        groups_claim="groups",
        admin_group="admins",
        allowed_group="",
        redirect_base_url="",
        display_name="SSO",
        only=False,
        logout_at_idp=False,
        allow_insecure=False,
    )
    base.update(kw)
    return oidc.OIDCConfig(**base)


@pytest.mark.parametrize(
    "groups,allowed_group,expected",
    [
        (["admins"], "", (True, True)),
        (["staff"], "", (True, False)),
        ([], "", (True, False)),
        (None, "", (True, False)),
        (["staff"], "staff", (True, False)),
        (["other"], "staff", (False, False)),
        (["admins"], "staff", (True, True)),
        ("admins", "", (True, True)),
    ],
)
def test_resolve_roles(groups, allowed_group, expected) -> None:
    claims = {} if groups is None else {"groups": groups}
    cfg = _cfg(allowed_group=allowed_group)
    assert oidc.resolve_roles(cfg, claims) == expected


def test_resolve_roles_without_admin_group_never_promotes() -> None:
    assert oidc.resolve_roles(_cfg(admin_group=""), {"groups": [""]}) == (True, False)


def test_nested_and_dotted_group_claims() -> None:
    cfg = _cfg(groups_claim="realm_access.roles")
    claims = {"realm_access": {"roles": ["admins", "x"]}}
    assert oidc.groups_of(cfg, claims) == ["admins", "x"]
    url_claim = "https://example.org/claims.groups"
    cfg = _cfg(groups_claim=url_claim)
    assert oidc.groups_of(cfg, {url_claim: ["a"]}) == ["a"]
    assert oidc.groups_of(_cfg(), {"groups": {"admins": {}, "b": {}}}) == [
        "admins",
        "b",
    ]
    assert oidc.groups_of(_cfg(), {"groups": 7}) == []


@pytest.mark.parametrize(
    "claims,expected",
    [
        ({"preferred_username": "Alice.Smith"}, "alice.smith"),
        ({"preferred_username": "x", "email": "Bob@Example.org"}, "bob"),
        ({"email": "carol+tag@example.org"}, "carol-tag"),
        ({"preferred_username": "Dörte Müller"}, "d-rte-m-ller"),
        ({"sub": "auth0|12345"}, "auth0-12345"),
        ({"preferred_username": "a" * 100}, "a" * 64),
    ],
)
def test_base_username(claims, expected) -> None:
    assert oidc.base_username(claims) == expected


def test_base_username_falls_back_to_hash() -> None:
    assert oidc.base_username({"sub": "!"}).startswith("user-")


def test_with_suffix_keeps_length() -> None:
    assert oidc.with_suffix("a" * 64, 2) == "a" * 62 + "-2"
    assert oidc.with_suffix("bob", 3) == "bob-3"


def test_config_problems() -> None:
    assert _cfg(issuer="").problems() == []
    assert not _cfg(issuer="").enabled
    assert _cfg().problems() == []
    assert any("CLIENT_ID" in p for p in _cfg(client_id="").problems())
    assert any("https" in p for p in _cfg(issuer="http://idp").problems())
    assert _cfg(issuer="http://idp", allow_insecure=True).problems() == []
    assert any("openid" in p for p in _cfg(scopes="profile").problems())
    assert _cfg(only=True).password_login_enabled is False
    assert _cfg(issuer="", only=True).password_login_enabled is True
    # A broken configuration must not lock everyone out.
    assert _cfg(client_secret="", only=True).password_login_enabled is True
    assert _cfg(issuer="http://idp", only=True).password_login_enabled is True


def test_state_cookie_round_trip_and_tampering() -> None:
    url, cookie = oidc.begin_login(
        _cfg(),
        {"authorization_endpoint": ISSUER + "/authorize?tenant=x"},
        callback_url="https://dash/login/oidc/callback",
        next_url="/users",
        secret="k" * 32,
    )
    assert url.startswith(ISSUER + "/authorize?tenant=x&")
    q = parse_qs(urlparse(url).query)
    assert q["code_challenge_method"] == ["S256"]
    state = oidc.load_state(cookie, "k" * 32)
    assert state["s"] == q["state"][0]
    assert state["n"] == q["nonce"][0]
    assert oidc.pkce_challenge(state["v"]) == q["code_challenge"][0]
    assert state["next"] == "/users"
    with pytest.raises(oidc.OIDCError) as e:
        oidc.load_state(cookie, "other-secret-xxxxxxxxxxxxxxxxxxx")
    assert e.value.code == "state"
    with pytest.raises(oidc.OIDCError):
        oidc.load_state(cookie[:-2] + "xx", "k" * 32)
    with pytest.raises(oidc.OIDCError):
        oidc.load_state(None, "k" * 32)


def test_redirect_uri_prefers_configured_base() -> None:
    assert (
        oidc.redirect_uri(_cfg(redirect_base_url="https://llm.example"), "http", "h")
        == "https://llm.example/login/oidc/callback"
    )
    assert oidc.redirect_uri(_cfg(), "https", "h:8") == "https://h:8" + (
        oidc.CALLBACK_PATH
    )


# --- Login page ---


async def test_login_page_shows_sso_button(dash_client, idp) -> None:
    client, db = dash_client
    await _seed_local_admin(db)
    r = await client.get("/login")
    assert r.status_code == 200
    assert "/login/oidc" in r.text
    assert 'name="password"' in r.text


async def test_login_page_without_oidc_has_no_button(dash_client, monkeypatch) -> None:
    client, db = dash_client
    monkeypatch.setattr(settings, "oidc_issuer", "")
    await _seed_local_admin(db)
    r = await client.get("/login")
    assert "/login/oidc" not in r.text
    assert (await client.get("/login/oidc")).status_code == 404


async def test_oidc_only_hides_password_form(dash_client, idp, monkeypatch) -> None:
    client, db = dash_client
    monkeypatch.setattr(settings, "oidc_only", True)
    await _seed_local_admin(db)
    r = await client.get("/login")
    assert 'name="password"' not in r.text
    r = await client.post(
        "/login", data={"username": "root", "password": "password123"}
    )
    assert r.status_code == 404


async def test_oidc_only_with_broken_config_keeps_password_login(
    dash_client, idp, monkeypatch
) -> None:
    client, db = dash_client
    monkeypatch.setattr(settings, "oidc_only", True)
    monkeypatch.setattr(settings, "oidc_client_secret", "")
    await _seed_local_admin(db)
    r = await client.get("/login")
    assert 'name="password"' in r.text
    assert "/login/oidc" not in r.text
    r = await client.post(
        "/login", data={"username": "root", "password": "password123"}
    )
    assert r.status_code == 303 and r.headers["location"] == "/"
    assert AUTH_COOKIE in r.cookies


async def test_login_page_open_in_bootstrap_when_oidc_on(dash_client, idp) -> None:
    client, _ = dash_client
    r = await client.get("/login")
    assert r.status_code == 200
    assert "/login/oidc" in r.text
    assert 'name="password"' not in r.text


# --- Full flow ---


async def test_admin_group_member_signs_in_as_admin(dash_client, idp) -> None:
    client, db = dash_client
    await _seed_local_admin(db)
    r = await _login(client, idp, "/users", groups=["router-admins", "staff"])
    assert r.status_code == 303, r.text
    assert r.headers["location"] == "/users"
    assert AUTH_COOKIE in r.cookies
    row = await db.get_dashboard_user_by_oidc(ISSUER, "user-1")
    assert row["username"] == "alice"
    assert row["is_admin"] is True
    assert row["auth_source"] == "oidc"
    assert row["password_hash"] is None
    assert row["email"] == "alice@example.org"
    assert row["display_name"] == "Alice Example"
    assert (await client.get("/users")).status_code == 200


async def test_non_member_signs_in_as_viewer(dash_client, idp) -> None:
    client, db = dash_client
    await _seed_local_admin(db)
    r = await _login(client, idp, groups=["staff"])
    assert r.status_code == 303
    row = await db.get_dashboard_user_by_oidc(ISSUER, "user-1")
    assert row["is_admin"] is False
    r = await client.get("/users")
    assert r.status_code == 302
    assert "forbidden" in r.headers["location"]
    assert (await client.get("/")).status_code == 200


async def test_admin_flag_follows_the_idp_on_each_login(dash_client, idp) -> None:
    client, db = dash_client
    await _seed_local_admin(db)
    await _login(client, idp, groups=["router-admins"])
    assert (await db.get_dashboard_user_by_oidc(ISSUER, "user-1"))["is_admin"]
    await _login(client, idp, groups=[], name="Alice Renamed")
    row = await db.get_dashboard_user_by_oidc(ISSUER, "user-1")
    assert row["is_admin"] is False
    assert row["display_name"] == "Alice Renamed"
    assert len(await db.list_dashboard_users()) == 2


async def test_allowed_group_refuses_outsiders(dash_client, idp, monkeypatch) -> None:
    client, db = dash_client
    monkeypatch.setattr(settings, "oidc_allowed_group", "router-users")
    await _seed_local_admin(db)
    r = await _login(client, idp, groups=["staff"])
    assert r.headers["location"] == "/login?error=oidc_not_member"
    assert AUTH_COOKIE not in r.cookies
    assert await db.get_dashboard_user_by_oidc(ISSUER, "user-1") is None
    r = await _login(client, idp, groups=["router-users"])
    assert r.status_code == 303 and r.headers["location"] == "/"


async def test_groups_from_userinfo_when_not_in_id_token(dash_client, idp) -> None:
    client, db = dash_client
    await _seed_local_admin(db)
    idp.userinfo = {"sub": "user-1", "groups": ["router-admins"]}
    await _login(client, idp)
    assert (await db.get_dashboard_user_by_oidc(ISSUER, "user-1"))["is_admin"]


async def test_userinfo_with_other_subject_is_rejected(dash_client, idp) -> None:
    client, db = dash_client
    await _seed_local_admin(db)
    idp.userinfo = {"sub": "someone-else", "groups": ["router-admins"]}
    r = await _login(client, idp)
    assert r.headers["location"] == "/login?error=oidc_token"


async def test_username_collision_gets_suffix_not_takeover(dash_client, idp) -> None:
    client, db = dash_client
    await db.create_dashboard_user("alice", hash_password("password123"), True)
    invalidate_dashboard_user_count_cache()
    r = await _login(client, idp, groups=[])
    assert r.status_code == 303
    row = await db.get_dashboard_user_by_oidc(ISSUER, "user-1")
    assert row["username"] == "alice-2"
    local = await db.get_dashboard_user_by_username("alice")
    assert local["auth_source"] == "local" and local["is_admin"]


async def test_password_login_refused_for_sso_account(dash_client, idp) -> None:
    client, db = dash_client
    await _seed_local_admin(db)
    await _login(client, idp, groups=[])
    client.cookies.clear()
    r = await client.post("/login", data={"username": "alice", "password": "x" * 8})
    assert r.headers["location"] == "/login?error=invalid"


async def test_admin_cannot_set_password_on_sso_account(dash_client, idp) -> None:
    client, db = dash_client
    await _seed_local_admin(db)
    await _login(client, idp, groups=["router-admins"])
    row = await db.get_dashboard_user_by_oidc(ISSUER, "user-1")
    r = await client.post(
        f"/users/{row['id']}/password", data={"new_password": "password123"}
    )
    assert r.headers["location"] == "/users?error=sso-password"
    assert (await db.get_dashboard_user_by_id(row["id"]))["password_hash"] is None


# --- Bootstrap ---


async def test_first_login_without_admin_group_is_admin(
    dash_client, idp, monkeypatch
) -> None:
    client, db = dash_client
    monkeypatch.setattr(settings, "oidc_admin_group", "")
    r = await _login(client, idp)
    assert r.status_code == 303
    assert (await db.get_dashboard_user_by_oidc(ISSUER, "user-1"))["is_admin"]
    # The dashboard now requires sign-in.
    client.cookies.clear()
    assert (await client.get("/")).status_code == 302


async def test_first_login_must_be_admin_when_group_configured(
    dash_client, idp
) -> None:
    client, db = dash_client
    r = await _login(client, idp, groups=["staff"])
    assert r.headers["location"] == "/login?error=oidc_first_admin"
    assert await db.count_dashboard_users() == 0
    r = await _login(client, idp, groups=["router-admins"])
    assert r.status_code == 303
    assert await db.count_dashboard_users() == 1


# --- Rejections ---


async def test_wrong_state_is_rejected(dash_client, idp) -> None:
    client, db = dash_client
    await _seed_local_admin(db)
    _, q = await _start(client)
    r = await client.get(oidc.CALLBACK_PATH, params={"code": "code-1", "state": "x"})
    assert r.headers["location"] == "/login?error=oidc_state"
    assert not [req for req in idp.requests if req.url.path == "/token"]


async def test_missing_state_cookie_is_rejected(dash_client, idp) -> None:
    client, db = dash_client
    await _seed_local_admin(db)
    _, q = await _start(client)
    client.cookies.delete(oidc.STATE_COOKIE)
    r = await client.get(
        oidc.CALLBACK_PATH, params={"code": "code-1", "state": q["state"]}
    )
    assert r.headers["location"] == "/login?error=oidc_state"


async def test_provider_error_param_is_denied(dash_client, idp) -> None:
    client, db = dash_client
    await _seed_local_admin(db)
    _, q = await _start(client)
    r = await client.get(
        oidc.CALLBACK_PATH, params={"error": "access_denied", "state": q["state"]}
    )
    assert r.headers["location"] == "/login?error=oidc_denied"


@pytest.mark.parametrize(
    "claims",
    [
        {"nonce": "wrong"},
        {"iss": "https://evil.example"},
        {"aud": "someone-else"},
        {"aud": [CLIENT_ID, "other"]},
        {"exp": int(time.time()) - 3600},
        {"sub": None},
    ],
)
async def test_bad_id_token_claims_are_rejected(dash_client, idp, claims) -> None:
    client, db = dash_client
    await _seed_local_admin(db)
    _, q = await _start(client)
    idp.nonce = q["nonce"]
    idp.challenge = q["code_challenge"]
    idp.claims = claims
    r = await client.get(
        oidc.CALLBACK_PATH, params={"code": "code-1", "state": q["state"]}
    )
    assert r.headers["location"] == "/login?error=oidc_token"
    assert AUTH_COOKIE not in r.cookies
    assert await db.get_dashboard_user_by_oidc(ISSUER, "user-1") is None


async def test_multiple_audiences_with_matching_azp_accepted(dash_client, idp) -> None:
    client, db = dash_client
    await _seed_local_admin(db)
    r = await _login(client, idp, aud=[CLIENT_ID, "other"], azp=CLIENT_ID)
    assert r.status_code == 303 and r.headers["location"] == "/"


async def test_token_signed_by_unknown_key_is_rejected(dash_client, idp) -> None:
    client, db = dash_client
    await _seed_local_admin(db)
    idp.signing_key = RSAKey.generate_key(2048, parameters={"kid": "k1"})
    r = await _login(client, idp)
    assert r.headers["location"] == "/login?error=oidc_token"


async def test_hs256_token_signed_with_client_secret_is_rejected(
    dash_client, idp
) -> None:
    from joserfc.jwk import OctKey

    client, db = dash_client
    await _seed_local_admin(db)
    _, q = await _start(client)
    idp.nonce, idp.challenge = q["nonce"], q["code_challenge"]
    now = int(time.time())
    idp.id_token_override = jwt.encode(
        {"alg": "HS256"},
        {
            "iss": ISSUER,
            "aud": CLIENT_ID,
            "sub": "user-1",
            "iat": now,
            "exp": now + 60,
            "nonce": q["nonce"],
        },
        OctKey.import_key(CLIENT_SECRET.encode() * 4),
    )
    r = await client.get(
        oidc.CALLBACK_PATH, params={"code": "code-1", "state": q["state"]}
    )
    assert r.headers["location"] == "/login?error=oidc_token"


async def test_key_rotation_refetches_jwks(dash_client, idp) -> None:
    client, db = dash_client
    await _seed_local_admin(db)
    await _login(client, idp, groups=[])
    new_key = RSAKey.generate_key(2048, parameters={"kid": "k2"})
    idp.published = [idp.key, new_key]
    idp.signing_key = new_key
    client.cookies.clear()
    r = await _login(client, idp, groups=[])
    assert r.status_code == 303 and r.headers["location"] == "/"
    jwks_fetches = [req for req in idp.requests if req.url.path == "/jwks"]
    assert len(jwks_fetches) == 2


async def test_token_endpoint_failure(dash_client, idp) -> None:
    client, db = dash_client
    await _seed_local_admin(db)
    idp.token_status = 500
    r = await _login(client, idp)
    assert r.headers["location"] == "/login?error=oidc_token"


async def test_unreachable_provider(dash_client, idp) -> None:
    client, db = dash_client
    await _seed_local_admin(db)
    idp.discovery_status = 503
    r = await client.get("/login/oidc")
    assert r.headers["location"] == "/login?error=oidc_provider"


async def test_discovery_issuer_mismatch(dash_client, idp) -> None:
    client, db = dash_client
    await _seed_local_admin(db)
    idp.extra_metadata = {"issuer": "https://other.example"}
    r = await client.get("/login/oidc")
    assert r.headers["location"] == "/login?error=oidc_provider"


async def test_open_redirect_in_next_is_neutralised(dash_client, idp) -> None:
    client, db = dash_client
    await _seed_local_admin(db)
    r = await _login(client, idp, "https://evil.example/x", groups=[])
    assert r.headers["location"] == "/"


async def test_forwarded_headers_shape_the_redirect_uri(dash_client, idp) -> None:
    client, db = dash_client
    await _seed_local_admin(db)
    r = await client.get(
        "/login/oidc",
        headers={"x-forwarded-proto": "https", "x-forwarded-host": "llm.example"},
    )
    q = parse_qs(urlparse(r.headers["location"]).query)
    assert q["redirect_uri"] == ["https://llm.example/login/oidc/callback"]


# --- Logout ---


async def test_logout_at_idp(dash_client, idp, monkeypatch) -> None:
    client, db = dash_client
    monkeypatch.setattr(settings, "oidc_logout_at_idp", True)
    await _seed_local_admin(db)
    await _login(client, idp, groups=[])
    r = await client.post("/logout")
    assert r.status_code == 303
    loc = urlparse(r.headers["location"])
    assert f"{loc.scheme}://{loc.netloc}{loc.path}" == ISSUER + "/logout"
    q = parse_qs(loc.query)
    assert q["post_logout_redirect_uri"] == ["http://testserver/login"]
    assert q["client_id"] == [CLIENT_ID]


async def test_logout_stays_local_by_default(dash_client, idp) -> None:
    client, db = dash_client
    await _seed_local_admin(db)
    await _login(client, idp, groups=[])
    r = await client.post("/logout")
    assert r.headers["location"] == "/login"


def test_id_token_is_well_formed_jwt() -> None:
    """Guard the fake: the token it signs carries the kid header."""
    fake = FakeIdP()
    header = json.loads(
        base64.urlsafe_b64decode(fake.id_token().split(".")[0] + "==").decode()
    )
    assert header["kid"] == "k1"


# --- Users page ---


async def test_users_page_marks_sso_accounts(dash_client, idp) -> None:
    client, db = dash_client
    await _seed_local_admin(db)
    await _login(client, idp, groups=["router-admins"])
    sso = await db.get_dashboard_user_by_oidc(ISSUER, "user-1")
    local = await db.get_dashboard_user_by_username("root")
    r = await client.get("/users")
    assert r.status_code == 200
    html = r.text
    assert "Single sign-on" in html
    assert ISSUER in html
    assert "http://testserver/login/oidc/callback" in html
    assert "router-admins" in html
    assert "alice@example.org" in html
    assert f'data-bs-target="#pw-{local["id"]}"' in html
    assert f'data-bs-target="#pw-{sso["id"]}"' not in html
    assert f'action="/users/{sso["id"]}/password"' not in html


async def test_users_page_reports_broken_config(dash_client, idp, monkeypatch) -> None:
    client, db = dash_client
    monkeypatch.setattr(settings, "oidc_client_secret", "")
    r = await client.get("/users")
    assert "CLIENT_SECRET is required" in r.text
    assert "/login/oidc?next=/users" not in r.text


async def test_users_page_bootstrap_offers_sso(dash_client, idp) -> None:
    client, _ = dash_client
    r = await client.get("/users")
    assert "/login/oidc?next=/users" in r.text


async def test_users_page_without_oidc_has_no_sso_card(
    dash_client, monkeypatch
) -> None:
    client, _ = dash_client
    monkeypatch.setattr(settings, "oidc_issuer", "")
    r = await client.get("/users")
    assert "Single sign-on" not in r.text
