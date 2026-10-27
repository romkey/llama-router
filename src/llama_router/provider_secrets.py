"""Encryption at rest for provider API keys.

Keys must be recoverable (they are sent upstream), so they are encrypted with
Fernet rather than hashed. The encryption key derives from
``LLAMA_ROUTER_SESSION_SECRET``. Without that setting a fixed development
key is used, which only obscures the value in the database; set the secret
in production. Changing the secret makes stored keys undecryptable, and the
affected providers report that their key must be entered again.
"""

from __future__ import annotations

import base64
import hashlib

from cryptography.fernet import Fernet, InvalidToken

from .config import settings

_DEV_SECRET = b"llama-router-provider-key-dev"
_CONTEXT = b"llama-router provider api key v1\x00"

DEFAULT_HEADER = "Authorization"


class ApiKeyUnavailable(Exception):
    """A stored provider API key could not be decrypted."""


def _fernet() -> Fernet:
    secret = (settings.session_secret or "").strip().encode("utf-8") or _DEV_SECRET
    digest = hashlib.sha256(_CONTEXT + secret).digest()
    return Fernet(base64.urlsafe_b64encode(digest))


def encrypt_api_key(plaintext: str) -> str:
    return _fernet().encrypt(plaintext.encode("utf-8")).decode("ascii")


def decrypt_api_key(token: str) -> str:
    try:
        return _fernet().decrypt(token.encode("ascii")).decode("utf-8")
    except (InvalidToken, ValueError) as exc:
        raise ApiKeyUnavailable(
            "stored API key cannot be decrypted (was LLAMA_ROUTER_SESSION_SECRET "
            "changed?); enter the key again"
        ) from exc


def _header_names(name: str | None) -> list[str]:
    return [h.strip() for h in (name or "").split(",") if h.strip()]


def normalize_header(name: str | None) -> str:
    """Header(s) that carry the key, comma-separated. Empty means ``Authorization``.

    Some services read the key from different headers on different routes
    (Anthropic: ``Authorization`` for its OpenAI-compatible routes, ``x-api-key``
    for its model list), so several may be named.
    """
    names = _header_names(name)
    if not names:
        return DEFAULT_HEADER
    for value in names:
        if not all(c.isalnum() or c in "-_" for c in value) or len(value) > 64:
            raise ValueError("header name may contain only letters, digits, - and _")
    if len(names) > 4:
        raise ValueError("at most four key headers")
    return ", ".join(names)


def auth_headers(api_key: str, header: str | None) -> dict[str, str]:
    """Headers to send upstream. ``Authorization`` gets a Bearer token."""
    out: dict[str, str] = {}
    for name in _header_names(normalize_header(header)):
        if name.lower() == "authorization":
            out["Authorization"] = f"Bearer {api_key}"
        else:
            out[name] = api_key
    return out
