"""Password hashing and signed session tokens for the dashboard."""

from __future__ import annotations

import hashlib
import os
import secrets
from dataclasses import dataclass

from itsdangerous import BadSignature, SignatureExpired, URLSafeTimedSerializer

AUTH_COOKIE = "lr_session"
SESSION_MAX_AGE = 14 * 24 * 3600


@dataclass(frozen=True)
class DashboardUser:
    id: int
    username: str
    is_admin: bool


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


def verify_password(password: str, stored: str) -> bool:
    try:
        sh, kh = stored.split(":")
        salt = bytes.fromhex(sh)
        expected = bytes.fromhex(kh)
    except (ValueError, TypeError):
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


def sign_session(user_id: int, secret: str) -> str:
    ser = URLSafeTimedSerializer(secret, salt="lr-dash-v1")
    return ser.dumps({"u": user_id})


def unsign_session(token: str, secret: str) -> int | None:
    ser = URLSafeTimedSerializer(secret, salt="lr-dash-v1")
    try:
        d = ser.loads(token, max_age=SESSION_MAX_AGE)
        return int(d["u"])
    except (BadSignature, SignatureExpired, KeyError, TypeError, ValueError):
        return None


def normalize_username(username: str) -> str:
    s = username.strip().lower()
    if len(s) < 2 or len(s) > 64:
        raise ValueError("Username must be 2–64 characters")
    allowed = set("abcdefghijklmnopqrstuvwxyz0123456789._-")
    if not all(c in allowed for c in s):
        raise ValueError(
            "Username may only contain letters, digits, period, underscore, hyphen"
        )
    return s


_NON_ADMIN_FORBIDDEN_PREFIXES = (
    "/api/keys",
    "/api/wireguard",
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
    if path.startswith("/api/benchmarks/fill-missing"):
        return True
    return False
