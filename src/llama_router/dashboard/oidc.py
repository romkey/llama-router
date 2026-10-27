"""OpenID Connect sign-in for the dashboard.

Authorization Code flow with PKCE against one identity provider. The flow
state (state, nonce, PKCE verifier, where to go afterwards) travels in a
short-lived signed cookie, so nothing is kept server side between the
redirect to the provider and the callback. ID tokens are verified against the
provider's published keys with ``joserfc``.

Membership in ``LLAMA_ROUTER_OIDC_ADMIN_GROUP`` makes a user an administrator.
The provider is authoritative: the flag is recomputed at every sign-in.
"""

from __future__ import annotations

import base64
import hashlib
import logging
import re
import secrets
import time
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlencode

import httpx
from itsdangerous import BadSignature, SignatureExpired, URLSafeTimedSerializer
from joserfc import jwt
from joserfc.errors import InvalidKeyIdError, JoseError
from joserfc.jwk import KeySet

from ..config import Settings, settings

logger = logging.getLogger(__name__)

STATE_COOKIE = "lr_oidc"
STATE_MAX_AGE = 600
CALLBACK_PATH = "/login/oidc/callback"

_STATE_SALT = "lr-oidc-v1"
_METADATA_TTL_S = 3600.0
_HTTP_TIMEOUT_S = 10.0
_CLOCK_SKEW_S = 60
# Asymmetric algorithms only: HS* would let anyone holding the client secret
# mint tokens, and "none" is never acceptable.
_ASYMMETRIC_ALGS = (
    "RS256",
    "RS384",
    "RS512",
    "PS256",
    "PS384",
    "PS512",
    "ES256",
    "ES384",
    "ES512",
    "EdDSA",
)


class OIDCError(Exception):
    """A sign-in that cannot complete. *code* goes into ``/login?error=oidc_<code>``."""

    def __init__(self, code: str, detail: str = "") -> None:
        super().__init__(f"{code}: {detail}" if detail else code)
        self.code = code
        self.detail = detail


@dataclass(frozen=True)
class OIDCConfig:
    issuer: str
    client_id: str
    client_secret: str
    scopes: str
    groups_claim: str
    admin_group: str
    allowed_group: str
    redirect_base_url: str
    display_name: str
    only: bool
    logout_at_idp: bool
    allow_insecure: bool

    @classmethod
    def from_settings(cls, s: Settings | None = None) -> OIDCConfig:
        s = s or settings
        return cls(
            issuer=s.oidc_issuer.strip().rstrip("/"),
            client_id=s.oidc_client_id.strip(),
            client_secret=s.oidc_client_secret.strip(),
            scopes=" ".join(s.oidc_scopes.split()) or "openid",
            groups_claim=s.oidc_groups_claim.strip() or "groups",
            admin_group=s.oidc_admin_group.strip(),
            allowed_group=s.oidc_allowed_group.strip(),
            redirect_base_url=s.oidc_redirect_base_url.strip().rstrip("/"),
            display_name=s.oidc_display_name.strip() or "single sign-on",
            only=s.oidc_only,
            logout_at_idp=s.oidc_logout_at_idp,
            allow_insecure=s.oidc_allow_insecure,
        )

    @property
    def enabled(self) -> bool:
        return bool(self.issuer)

    @property
    def password_login_enabled(self) -> bool:
        # OIDC_ONLY takes effect only while SSO actually works; a broken
        # configuration must not lock out every account.
        return not (self.usable and self.only)

    def problems(self) -> list[str]:
        """Configuration errors that stop OIDC from working at all."""
        if not self.enabled:
            return []
        out = []
        if not self.client_id:
            out.append("LLAMA_ROUTER_OIDC_CLIENT_ID is required")
        if not self.client_secret:
            out.append("LLAMA_ROUTER_OIDC_CLIENT_SECRET is required")
        if not self._url_ok(self.issuer):
            out.append(
                "LLAMA_ROUTER_OIDC_ISSUER must be https:// "
                "(set LLAMA_ROUTER_OIDC_ALLOW_INSECURE for local testing)"
            )
        if "openid" not in self.scopes.split():
            out.append("LLAMA_ROUTER_OIDC_SCOPES must include openid")
        return out

    @property
    def usable(self) -> bool:
        return self.enabled and not self.problems()

    def _url_ok(self, url: str) -> bool:
        if url.startswith("https://"):
            return True
        return self.allow_insecure and url.startswith("http://")


def current_config() -> OIDCConfig:
    """Read the settings each time, so tests and restarts see changes."""
    return OIDCConfig.from_settings()


# --- Provider metadata and keys ---


@dataclass
class _ProviderCache:
    metadata: dict[str, Any]
    fetched_at: float
    keys: KeySet | None = None
    keys_fetched_at: float = 0.0


_providers: dict[str, _ProviderCache] = {}


def reset_cache() -> None:
    _providers.clear()


def _client() -> httpx.AsyncClient:
    return httpx.AsyncClient(timeout=_HTTP_TIMEOUT_S, follow_redirects=False)


async def _get_json(url: str, **kw: Any) -> dict[str, Any]:
    try:
        async with _client() as client:
            resp = await client.get(url, **kw)
    except httpx.HTTPError as exc:
        raise OIDCError("provider", f"GET {url}: {exc!r}") from exc
    if resp.status_code != 200:
        raise OIDCError("provider", f"GET {url}: HTTP {resp.status_code}")
    try:
        data = resp.json()
    except ValueError as exc:
        raise OIDCError("provider", f"GET {url}: not JSON") from exc
    if not isinstance(data, dict):
        raise OIDCError("provider", f"GET {url}: not a JSON object")
    return data


async def provider_metadata(cfg: OIDCConfig) -> dict[str, Any]:
    """The provider's discovery document, cached for an hour."""
    cached = _providers.get(cfg.issuer)
    now = time.monotonic()
    if cached is not None and now - cached.fetched_at < _METADATA_TTL_S:
        return cached.metadata
    meta = await _get_json(cfg.issuer + "/.well-known/openid-configuration")
    if str(meta.get("issuer", "")).rstrip("/") != cfg.issuer:
        raise OIDCError(
            "provider",
            f"discovery issuer {meta.get('issuer')!r} does not match {cfg.issuer!r}",
        )
    for field in ("authorization_endpoint", "token_endpoint", "jwks_uri"):
        url = meta.get(field)
        if not isinstance(url, str) or not cfg._url_ok(url):
            raise OIDCError("provider", f"discovery {field} missing or not https")
    _providers[cfg.issuer] = _ProviderCache(metadata=meta, fetched_at=now)
    return meta


async def _key_set(cfg: OIDCConfig, meta: dict[str, Any], refresh: bool) -> KeySet:
    cached = _providers.get(cfg.issuer)
    now = time.monotonic()
    if (
        cached is not None
        and cached.keys is not None
        and not refresh
        and now - cached.keys_fetched_at < _METADATA_TTL_S
    ):
        return cached.keys
    data = await _get_json(meta["jwks_uri"])
    try:
        keys = KeySet.import_key_set(data)  # type: ignore[arg-type]
    except (JoseError, ValueError, KeyError, TypeError) as exc:
        raise OIDCError("provider", f"bad JWKS: {exc!r}") from exc
    if cached is not None:
        cached.keys = keys
        cached.keys_fetched_at = now
    return keys


# --- Flow state ---


def _serializer(secret: str) -> URLSafeTimedSerializer:
    return URLSafeTimedSerializer(secret, salt=_STATE_SALT)


def _b64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def pkce_challenge(verifier: str) -> str:
    return _b64url(hashlib.sha256(verifier.encode("ascii")).digest())


def redirect_uri(cfg: OIDCConfig, scheme: str, host: str) -> str:
    """The callback URL registered at the provider.

    An explicit base URL wins. Otherwise the request's own scheme and host are
    used; the provider rejects any redirect URI it was not configured with, so
    a spoofed Host header cannot send the code elsewhere.
    """
    if cfg.redirect_base_url:
        return cfg.redirect_base_url + CALLBACK_PATH
    return f"{scheme}://{host}{CALLBACK_PATH}"


def begin_login(
    cfg: OIDCConfig,
    meta: Mapping[str, Any],
    *,
    callback_url: str,
    next_url: str,
    secret: str,
) -> tuple[str, str]:
    """Return the provider's authorization URL and the signed state cookie."""
    state = secrets.token_urlsafe(32)
    nonce = secrets.token_urlsafe(32)
    verifier = secrets.token_urlsafe(64)
    params = {
        "response_type": "code",
        "client_id": cfg.client_id,
        "redirect_uri": callback_url,
        "scope": cfg.scopes,
        "state": state,
        "nonce": nonce,
        "code_challenge": pkce_challenge(verifier),
        "code_challenge_method": "S256",
    }
    endpoint = str(meta["authorization_endpoint"])
    sep = "&" if "?" in endpoint else "?"
    cookie = _serializer(secret).dumps(
        {"s": state, "n": nonce, "v": verifier, "next": next_url, "r": callback_url}
    )
    return endpoint + sep + urlencode(params), cookie


def load_state(cookie: str | None, secret: str) -> dict[str, str]:
    if not cookie:
        raise OIDCError("state", "no state cookie")
    try:
        data = _serializer(secret).loads(cookie, max_age=STATE_MAX_AGE)
    except SignatureExpired as exc:
        raise OIDCError("state", "state cookie expired") from exc
    except BadSignature as exc:
        raise OIDCError("state", "state cookie tampered") from exc
    if not isinstance(data, dict) or not all(
        isinstance(data.get(k), str) for k in ("s", "n", "v", "next", "r")
    ):
        raise OIDCError("state", "state cookie malformed")
    return data


# --- Callback ---


async def complete_login(
    cfg: OIDCConfig,
    params: Mapping[str, str],
    cookie: str | None,
    secret: str,
) -> tuple[dict[str, Any], str]:
    """Finish the callback. Returns the verified claims and the ``next`` URL."""
    flow = load_state(cookie, secret)
    if params.get("error"):
        raise OIDCError(
            "denied",
            f"{params.get('error')}: {params.get('error_description', '')}",
        )
    if not secrets.compare_digest(
        str(params.get("state", "")).encode(), flow["s"].encode()
    ):
        raise OIDCError("state", "state mismatch")
    code = params.get("code")
    if not code:
        raise OIDCError("state", "no authorization code")

    meta = await provider_metadata(cfg)
    tokens = await _exchange_code(cfg, meta, code, flow)
    id_token = tokens.get("id_token")
    if not isinstance(id_token, str) or not id_token:
        raise OIDCError("token", "token response has no id_token")
    claims = await verify_id_token(cfg, meta, id_token, nonce=flow["n"])

    access_token = tokens.get("access_token")
    if (
        claim_value(claims, cfg.groups_claim) is None
        and meta.get("userinfo_endpoint")
        and isinstance(access_token, str)
    ):
        info = await _get_json(
            str(meta["userinfo_endpoint"]),
            headers={"Authorization": f"Bearer {access_token}"},
        )
        if info.get("sub") != claims.get("sub"):
            raise OIDCError("token", "userinfo subject does not match the ID token")
        for k, v in info.items():
            claims.setdefault(k, v)
    return claims, flow["next"]


async def _exchange_code(
    cfg: OIDCConfig,
    meta: Mapping[str, Any],
    code: str,
    flow: Mapping[str, str],
) -> dict[str, Any]:
    data = {
        "grant_type": "authorization_code",
        "code": code,
        "redirect_uri": flow["r"],
        "code_verifier": flow["v"],
    }
    methods = meta.get("token_endpoint_auth_methods_supported") or [
        "client_secret_basic"
    ]
    auth: tuple[str, str] | None = None
    if "client_secret_basic" in methods:
        auth = (cfg.client_id, cfg.client_secret)
    else:
        data["client_id"] = cfg.client_id
        data["client_secret"] = cfg.client_secret
    url = str(meta["token_endpoint"])
    try:
        async with _client() as client:
            resp = await client.post(
                url, data=data, auth=auth, headers={"Accept": "application/json"}
            )
    except httpx.HTTPError as exc:
        raise OIDCError("provider", f"POST {url}: {exc!r}") from exc
    try:
        body = resp.json()
    except ValueError:
        body = {}
    if resp.status_code != 200 or not isinstance(body, dict):
        err = body.get("error") if isinstance(body, dict) else None
        raise OIDCError("token", f"token endpoint HTTP {resp.status_code} {err or ''}")
    return body


async def verify_id_token(
    cfg: OIDCConfig,
    meta: Mapping[str, Any],
    id_token: str,
    *,
    nonce: str,
) -> dict[str, Any]:
    advertised = meta.get("id_token_signing_alg_values_supported") or ["RS256"]
    algs = [a for a in advertised if a in _ASYMMETRIC_ALGS]
    if not algs:
        raise OIDCError("provider", f"no supported ID token algorithm in {advertised}")

    keys = await _key_set(cfg, meta, refresh=False)
    try:
        try:
            token = jwt.decode(id_token, keys, algorithms=algs)
        except InvalidKeyIdError:
            # The provider rotated its keys since we cached them.
            keys = await _key_set(cfg, meta, refresh=True)
            token = jwt.decode(id_token, keys, algorithms=algs)
    except JoseError as exc:
        raise OIDCError("token", f"ID token rejected: {exc}") from exc
    except ValueError as exc:
        raise OIDCError("token", f"ID token malformed: {exc}") from exc

    claims = dict(token.claims)
    registry = jwt.JWTClaimsRegistry(
        leeway=_CLOCK_SKEW_S,
        iss={"essential": True, "value": str(meta["issuer"])},
        aud={"essential": True, "value": cfg.client_id},
        sub={"essential": True},
        exp={"essential": True},
        iat={"essential": True},
        nonce={"essential": True, "value": nonce},
    )
    try:
        registry.validate(claims)
    except JoseError as exc:
        raise OIDCError("token", f"ID token claims rejected: {exc}") from exc
    aud = claims.get("aud")
    if isinstance(aud, list) and len(aud) > 1 and claims.get("azp") != cfg.client_id:
        raise OIDCError("token", "ID token has several audiences and azp is not us")
    if not isinstance(claims.get("sub"), str) or not claims["sub"]:
        raise OIDCError("token", "ID token sub is empty")
    return claims


# --- Mapping claims to an account ---


def claim_value(claims: Mapping[str, Any], path: str) -> Any:
    """Look up *path* in the claims; dots walk into nested objects.

    A claim whose own name contains dots (some providers use URLs as claim
    names) is matched whole first.
    """
    if path in claims:
        return claims[path]
    cur: Any = claims
    for part in path.split("."):
        if not isinstance(cur, Mapping) or part not in cur:
            return None
        cur = cur[part]
    return cur


def groups_of(cfg: OIDCConfig, claims: Mapping[str, Any]) -> list[str]:
    value = claim_value(claims, cfg.groups_claim)
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple)):
        return [str(v) for v in value if isinstance(v, (str, int))]
    if isinstance(value, Mapping):
        # Zitadel-style {"role": {...}} maps.
        return [str(k) for k in value]
    return []


def resolve_roles(cfg: OIDCConfig, claims: Mapping[str, Any]) -> tuple[bool, bool]:
    """Return ``(allowed, is_admin)`` for the signed-in user."""
    groups = set(groups_of(cfg, claims))
    is_admin = bool(cfg.admin_group) and cfg.admin_group in groups
    allowed = not cfg.allowed_group or cfg.allowed_group in groups or is_admin
    return allowed, is_admin


_USERNAME_BAD = re.compile(r"[^a-z0-9._-]+")


def _clean_username(raw: Any) -> str | None:
    if not isinstance(raw, str):
        return None
    s = _USERNAME_BAD.sub("-", raw.strip().lower()).strip("-.")
    s = s[:64].rstrip("-.")
    return s if len(s) >= 2 else None


def base_username(claims: Mapping[str, Any]) -> str:
    """The preferred dashboard username for these claims (before de-duplication)."""
    email = claims.get("email")
    local = email.split("@", 1)[0] if isinstance(email, str) else None
    for raw in (claims.get("preferred_username"), local, claims.get("sub")):
        name = _clean_username(raw)
        if name:
            return name
    digest = hashlib.sha256(str(claims.get("sub", "")).encode()).hexdigest()[:10]
    return f"user-{digest}"


def with_suffix(base: str, n: int) -> str:
    suffix = f"-{n}"
    return base[: 64 - len(suffix)] + suffix


def display_name_of(claims: Mapping[str, Any]) -> str | None:
    for key in ("name", "preferred_username", "email"):
        v = claims.get(key)
        if isinstance(v, str) and v.strip():
            return v.strip()[:200]
    return None


def email_of(claims: Mapping[str, Any]) -> str | None:
    v = claims.get("email")
    return v.strip()[:320] if isinstance(v, str) and v.strip() else None


async def sign_in(db, cfg: OIDCConfig, claims: Mapping[str, Any]) -> dict:
    """Create or refresh the account for *claims*. Returns the user row.

    Accounts are matched only by ``(iss, sub)``; an email address from the
    provider never takes over an existing local account.
    """
    allowed, is_admin = resolve_roles(cfg, claims)
    if not allowed:
        raise OIDCError("not_member", f"sub {claims.get('sub')!r} not in group")
    issuer = str(claims["iss"])
    subject = str(claims["sub"])
    email = email_of(claims)
    display = display_name_of(claims)

    row = await db.get_dashboard_user_by_oidc(issuer, subject)
    if row is not None:
        if row["is_admin"] and not is_admin:
            logger.info("OIDC user %s is no longer an administrator", row["username"])
            if await db.count_dashboard_admins() <= 1:
                logger.warning(
                    "The last dashboard administrator (%s) lost admin through "
                    "the identity provider",
                    row["username"],
                )
        await db.update_oidc_dashboard_user(
            row["id"], email=email, display_name=display, is_admin=is_admin
        )
        return await db.get_dashboard_user_by_id(row["id"])

    if await db.count_dashboard_users() == 0:
        if not cfg.admin_group:
            # Same rule as local accounts: the first account administers.
            is_admin = True
        elif not is_admin:
            # A non-admin first account would close the open bootstrap
            # dashboard with nobody able to manage it.
            raise OIDCError("first_admin", "first sign-in is not an administrator")

    base = base_username(claims)
    for n in range(1, 1000):
        name = base if n == 1 else with_suffix(base, n)
        if await db.get_dashboard_user_by_username(name) is not None:
            continue
        try:
            uid = await db.create_oidc_dashboard_user(
                name,
                issuer,
                subject,
                email=email,
                display_name=display,
                is_admin=is_admin,
            )
        except Exception:
            # Lost a race for the name or identity; look again.
            again = await db.get_dashboard_user_by_oidc(issuer, subject)
            if again is not None:
                return again
            continue
        logger.info(
            "Created dashboard user %s from OIDC%s",
            name,
            " (administrator)" if is_admin else "",
        )
        return await db.get_dashboard_user_by_id(uid)
    raise OIDCError("token", "could not choose a free username")


def logout_url(
    cfg: OIDCConfig, meta: Mapping[str, Any], post_logout_redirect: str
) -> str | None:
    endpoint = meta.get("end_session_endpoint")
    if not cfg.logout_at_idp or not isinstance(endpoint, str):
        return None
    params = {
        "client_id": cfg.client_id,
        "post_logout_redirect_uri": post_logout_redirect,
    }
    sep = "&" if "?" in endpoint else "?"
    return endpoint + sep + urlencode(params)
