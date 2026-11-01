"""Password hashing and signed session tokens for the dashboard."""

from __future__ import annotations

import hashlib
import os
import secrets
import time
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from itsdangerous import BadSignature, SignatureExpired, URLSafeTimedSerializer

AUTH_COOKIE = "lr_session"
SESSION_MAX_AGE = 14 * 24 * 3600
# Single sign-on sessions are shorter, so a user removed or demoted at the
# identity provider loses the dashboard within half a day.
OIDC_SESSION_MAX_AGE = 12 * 3600


@dataclass(frozen=True)
class DashboardUser:
    id: int
    username: str
    is_admin: bool
    auth_source: str = "local"

    @property
    def is_sso(self) -> bool:
        return self.auth_source == "oidc"


def hash_password(password: str) -> str:
    salt = os.urandom(16)
    dk = hashlib.scrypt(
        password.encode("utf-8"),
        salt=salt,
        n=2**14,
        r=8,
        p=1,
        dklen=32,
    )
    return f"{salt.hex()}:{dk.hex()}"


def verify_password(password: str, stored: str | None) -> bool:
    if not stored:
        # Accounts that sign in through OIDC have no password.
        return False
    try:
        sh, kh = stored.split(":")
        salt = bytes.fromhex(sh)
        expected = bytes.fromhex(kh)
    except ValueError, TypeError:
        return False
    dk = hashlib.scrypt(
        password.encode("utf-8"),
        salt=salt,
        n=2**14,
        r=8,
        p=1,
        dklen=32,
    )
    return secrets.compare_digest(dk, expected)


_DUMMY_HASH: str | None = None


def burn_password_check(password: str) -> None:
    """Take as long as :func:`verify_password` does on a real account.

    Sign-in for an unknown or single sign-on username runs this instead, so
    the response time does not reveal which local usernames exist.
    """
    global _DUMMY_HASH
    if _DUMMY_HASH is None:
        _DUMMY_HASH = hash_password(secrets.token_urlsafe(16))
    verify_password(password, _DUMMY_HASH)


def session_max_age(auth_source: str) -> int:
    return OIDC_SESSION_MAX_AGE if auth_source == "oidc" else SESSION_MAX_AGE


@dataclass(frozen=True)
class SessionToken:
    user_id: int
    # The account's session_version when the cookie was issued; bumping the
    # column revokes every older cookie.
    version: int
    # Guards against a new account reusing a deleted one's id. None only in
    # tokens minted without one (tests); the dashboard always binds it.
    username: str | None
    issued_at: float

    def valid_for(self, row: Mapping[str, Any], now: float | None = None) -> bool:
        """Whether this cookie still signs in the account *row*."""
        if int(row.get("session_version") or 0) != self.version:
            return False
        if self.username is not None and row.get("username") != self.username:
            return False
        if not row.get("approved", True):
            return False
        age = (time.time() if now is None else now) - self.issued_at
        return age <= session_max_age(row.get("auth_source") or "local")


def sign_session(
    user_id: int, secret: str, version: int = 0, username: str | None = None
) -> str:
    ser = URLSafeTimedSerializer(secret, salt="lr-dash-v1")
    payload: dict[str, Any] = {"u": user_id, "v": version}
    if username is not None:
        payload["n"] = username
    return ser.dumps(payload)


def unsign_session(token: str, secret: str) -> SessionToken | None:
    """Decode a session cookie; cookies without a session version are refused."""
    ser = URLSafeTimedSerializer(secret, salt="lr-dash-v1")
    try:
        d, ts = ser.loads(token, max_age=SESSION_MAX_AGE, return_timestamp=True)
        name = d.get("n")
        return SessionToken(
            user_id=int(d["u"]),
            version=int(d["v"]),
            username=name if isinstance(name, str) else None,
            issued_at=ts.timestamp(),
        )
    except (
        BadSignature,
        SignatureExpired,
        KeyError,
        TypeError,
        ValueError,
        AttributeError,
    ):
        return None


def normalize_username(username: str) -> str:
    s = username.strip().lower()
    if len(s) < 2 or len(s) > 64:
        raise ValueError("Username must be 2–64 characters")  # noqa: RUF001
    allowed = set("abcdefghijklmnopqrstuvwxyz0123456789._-")
    if not all(c in allowed for c in s):
        raise ValueError(
            "Username may only contain letters, digits, period, underscore, hyphen"
        )
    return s


_NON_ADMIN_FORBIDDEN_PREFIXES = (
    "/api/keys",
    "/api/management-keys",
    "/api/workers",
    "/api/worker-link",
    "/api/cache",
    "/api/fallbacks",
    "/api/pulls",
    "/api/auth/allow-unauthenticated",
)


def non_admin_forbidden_get_path(path: str) -> bool:
    if path == "/users" or path.startswith("/users/"):
        return True
    for prefix in _NON_ADMIN_FORBIDDEN_PREFIXES:
        if path == prefix or path.startswith(prefix + "/"):
            return True
    return path.startswith("/api/benchmarks/fill-missing")
