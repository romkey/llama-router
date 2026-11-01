"""Encryption at rest for provider API keys and other stored secrets.

Keys must be recoverable (they are sent upstream), so they are encrypted with
Fernet rather than hashed. The encryption key derives from
``LLAMA_ROUTER_SESSION_SECRET``. Without that setting a random secret is
generated on first use and kept in a file beside the database
(``LLAMA_ROUTER_SECRET_KEY_PATH``, default ``secret.key`` in the database
directory: ``/app/data/secret.key`` in the Docker image), readable only by
the router and never stored in the database itself. Changing the secret, or
losing that file, makes stored keys undecryptable, and the affected providers
report that their key must be entered again.

Before that file existed a fixed, public development key was used; values
encrypted with it still decrypt, and :func:`reencrypt_legacy` moves them to
the current key.
"""

from __future__ import annotations

import base64
import hashlib
import logging
import os
import secrets
import threading
from pathlib import Path

from cryptography.fernet import Fernet, InvalidToken

from .config import settings

logger = logging.getLogger(__name__)

# The fixed key older releases used without a session secret. Only read
# so existing values can be decrypted and re-encrypted; never used to encrypt.
_LEGACY_DEV_SECRET = b"llama-router-provider-key-dev"
_CONTEXT = b"llama-router provider api key v1\x00"

DEFAULT_HEADER = "Authorization"
SECRET_FILE_NAME = "secret.key"  # noqa: S105  # a file name, not a secret

_local_secrets: dict[Path, bytes] = {}
_local_lock = threading.Lock()


class ApiKeyUnavailableError(Exception):
    """A stored provider API key could not be decrypted."""


def secret_key_path() -> Path:
    """Where the generated secret lives when no session secret is set."""
    configured = settings.secret_key_path.strip()
    if configured:
        return Path(configured).expanduser()
    db_dir = Path(settings.database_path).expanduser().resolve().parent
    return db_dir / SECRET_FILE_NAME


def _read_secret(path: Path) -> bytes:
    value = path.read_bytes().strip()
    if not value:
        raise RuntimeError(f"secret key file {path} is empty; delete it or fill it")
    return value


def _local_secret() -> bytes:
    """The generated secret, created (mode 0600) the first time it is needed."""
    path = secret_key_path()
    with _local_lock:
        cached = _local_secrets.get(path)
        if cached is not None:
            return cached
        try:
            value = _read_secret(path)
        except FileNotFoundError:
            path.parent.mkdir(parents=True, exist_ok=True)
            value = secrets.token_urlsafe(48).encode("ascii")
            try:
                fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            except FileExistsError:
                # Another process created it first; use theirs.
                value = _read_secret(path)
            else:
                with os.fdopen(fd, "wb") as fh:
                    fh.write(value + b"\n")
                logger.warning(
                    "LLAMA_ROUTER_SESSION_SECRET is not set: generated a secret "
                    "in %s. Keep that file (and back it up with the database); "
                    "without it stored provider keys and client API keys stop "
                    "working.",
                    path,
                )
        _local_secrets[path] = value
        return value


def server_secret() -> bytes:
    """``LLAMA_ROUTER_SESSION_SECRET``, or the generated secret file's value."""
    configured = (settings.session_secret or "").strip()
    if configured:
        return configured.encode("utf-8")
    return _local_secret()


def _fernet_for(secret: bytes) -> Fernet:
    digest = hashlib.sha256(_CONTEXT + secret).digest()
    return Fernet(base64.urlsafe_b64encode(digest))


def _fernet() -> Fernet:
    return _fernet_for(server_secret())


def encrypt_api_key(plaintext: str) -> str:
    return _fernet().encrypt(plaintext.encode("utf-8")).decode("ascii")


def _decrypt(token: str) -> tuple[str, bool]:
    """Plaintext, and whether only the legacy development key opened it."""
    try:
        return _fernet().decrypt(token.encode("ascii")).decode("utf-8"), False
    except InvalidToken, ValueError:
        pass
    try:
        legacy = _fernet_for(_LEGACY_DEV_SECRET).decrypt(token.encode("ascii"))
    except (InvalidToken, ValueError) as exc:
        raise ApiKeyUnavailableError(
            "stored API key cannot be decrypted (was LLAMA_ROUTER_SESSION_SECRET "
            "changed, or the secret key file lost?); enter the key again"
        ) from exc
    return legacy.decode("utf-8"), True


def decrypt_api_key(token: str) -> str:
    return _decrypt(token)[0]


def reencrypt_legacy(token: str) -> str | None:
    """*token* re-encrypted with the current key if it used the legacy one.

    None when it already uses the current key or cannot be decrypted at all.
    """
    try:
        plaintext, legacy = _decrypt(token)
    except ApiKeyUnavailableError:
        return None
    return encrypt_api_key(plaintext) if legacy else None


# Marks encrypted values in app_settings; older releases stored them plain.
ENCRYPTED_SETTING_PREFIX = "enc:v1:"


def encrypt_setting(plaintext: str) -> str:
    """Encrypt a secret for a text column that may also hold plain values."""
    return ENCRYPTED_SETTING_PREFIX + encrypt_api_key(plaintext)


def decrypt_setting(stored: str) -> tuple[str, bool]:
    """Plaintext of a value from :func:`encrypt_setting`, and whether to re-save it.

    Values without the prefix are plain text from an older release; they are
    returned as they are, flagged for re-saving encrypted. Raises
    :class:`ApiKeyUnavailableError` if an encrypted value cannot be decrypted.
    """
    if not stored.startswith(ENCRYPTED_SETTING_PREFIX):
        return stored, bool(stored)
    plaintext, legacy = _decrypt(stored[len(ENCRYPTED_SETTING_PREFIX) :])
    return plaintext, legacy


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


_RESERVED_HEADERS = {
    "authorization",
    "api-key",
    "x-api-key",
    "host",
    "content-length",
    "content-type",
    "transfer-encoding",
    "connection",
}


def parse_extra_headers(text: str | None) -> dict[str, str]:
    """Parse ``Name: value`` lines into static headers for a provider.

    Keys belong in the API key field, which is stored encrypted, so the key
    headers are refused here.
    """
    out: dict[str, str] = {}
    for raw in (text or "").splitlines():
        line = raw.strip()
        if not line:
            continue
        name, sep, value = line.partition(":")
        name, value = name.strip(), value.strip()
        if not sep or not name:
            raise ValueError(f"extra header must be 'Name: value': {line!r}")
        if not all(c.isalnum() or c in "-_" for c in name) or len(name) > 64:
            raise ValueError(f"bad header name {name!r}")
        if name.lower() in _RESERVED_HEADERS:
            raise ValueError(f"{name} cannot be an extra header")
        if len(value) > 1024 or any(ord(c) < 32 for c in value):
            raise ValueError(f"bad value for header {name}")
        out[name] = value
    if len(out) > 16:
        raise ValueError("at most 16 extra headers")
    return out


def format_extra_headers(headers: dict[str, str] | None) -> str:
    return "\n".join(f"{k}: {v}" for k, v in (headers or {}).items())


def auth_headers(api_key: str, header: str | None) -> dict[str, str]:
    """Headers to send upstream. ``Authorization`` gets a Bearer token."""
    out: dict[str, str] = {}
    for name in _header_names(normalize_header(header)):
        if name.lower() == "authorization":
            out["Authorization"] = f"Bearer {api_key}"
        else:
            out[name] = api_key
    return out
