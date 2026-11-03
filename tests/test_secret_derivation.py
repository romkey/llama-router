"""HKDF per-purpose keys, HMAC API-key hashes and migration from older formats."""

from __future__ import annotations

import hashlib
import hmac

import pytest
from cryptography.fernet import InvalidToken
from pydantic import ValidationError

from llama_router import provider_secrets as ps
from llama_router.auth import _resolve_key, generate_api_key, key_prefix
from llama_router.config import MIN_SESSION_SECRET_LENGTH, Settings, settings
from llama_router.database import Database

SECRET = "current-" + "c" * 32
OLD_SECRET = "previous-" + "p" * 32


@pytest.fixture(autouse=True)
def configured_secret(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "session_secret", SECRET)
    monkeypatch.setattr(settings, "session_secret_previous", "")


def _rfc5869(secret: bytes, info: bytes) -> bytes:
    """HKDF-SHA256 written out, so the derivation cannot drift unnoticed:
    a change would make every deployment's stored data unreadable."""
    prk = hmac.new(b"llama-router secrets v1", secret, hashlib.sha256).digest()
    return hmac.new(prk, info + b"\x01", hashlib.sha256).digest()


def test_derived_keys_follow_hkdf_and_differ_per_purpose() -> None:
    keys = {
        purpose: ps.derive_key(purpose)
        for purpose in (
            ps.PURPOSE_SESSION,
            ps.PURPOSE_API_KEY_PEPPER,
            ps.PURPOSE_FERNET,
        )
    }
    for purpose, key in keys.items():
        assert key == _rfc5869(SECRET.encode(), purpose.encode())
    assert len(set(keys.values())) == 3
    assert ps.session_signing_key() == keys[ps.PURPOSE_SESSION].hex()
    assert SECRET not in ps.session_signing_key()


def test_key_hash_is_hmac_with_the_pepper_key() -> None:
    key = generate_api_key()
    pepper = ps.derive_key(ps.PURPOSE_API_KEY_PEPPER)
    expected = hmac.new(pepper, key.encode(), hashlib.sha256).hexdigest()
    assert ps.key_hash(key) == expected
    assert ps.key_hash(key) != ps.key_hash_pbkdf2(key, SECRET.encode())


def test_new_values_use_the_hkdf_fernet_key() -> None:
    token = ps.encrypt_api_key("sk-new")
    assert ps._hkdf_fernet().decrypt(token.encode()) == b"sk-new"
    with pytest.raises(InvalidToken):
        ps._fernet_for(SECRET.encode()).decrypt(token.encode())
    assert ps.reencrypt_legacy(token) is None


def test_pre_hkdf_fernet_values_decrypt_and_are_reencrypted() -> None:
    old = ps._fernet_for(SECRET.encode()).encrypt(b"sk-old").decode()
    assert ps.decrypt_api_key(old) == "sk-old"
    fresh = ps.reencrypt_legacy(old)
    assert fresh is not None
    assert ps._hkdf_fernet().decrypt(fresh.encode()) == b"sk-old"
    assert ps.decrypt_setting(ps.ENCRYPTED_SETTING_PREFIX + old) == ("sk-old", True)


@pytest.mark.parametrize("hkdf", [True, False])
def test_previous_secret_opens_old_values(
    monkeypatch: pytest.MonkeyPatch, hkdf: bool
) -> None:
    old_fernet = (
        ps._hkdf_fernet(OLD_SECRET.encode())
        if hkdf
        else ps._fernet_for(OLD_SECRET.encode())
    )
    token = old_fernet.encrypt(b"sk-rotated").decode()
    with pytest.raises(ps.ApiKeyUnavailableError):
        ps.decrypt_api_key(token)
    monkeypatch.setattr(settings, "session_secret_previous", OLD_SECRET)
    fresh = ps.reencrypt_legacy(token)
    assert fresh is not None
    monkeypatch.setattr(settings, "session_secret_previous", "")
    assert ps.decrypt_api_key(fresh) == "sk-rotated"


async def _stored(db: Database, key: str, fingerprint: str) -> int:
    return await db.create_api_key(
        key_prefix=key_prefix(key),
        key_hash=fingerprint,
        routing_mode="auto",
        allow_fallback=True,
    )


async def test_pbkdf2_key_authenticates_and_is_upgraded(db: Database) -> None:
    key = generate_api_key()
    old = ps.key_hash_pbkdf2(key, SECRET.encode())
    key_id = await _stored(db, key, old)
    resolved = await _resolve_key(db, key, ps.key_hash_legacy(key))
    assert resolved is not None
    assert resolved[1] == key_id
    assert await db.lookup_api_key(old) is None
    assert await db.lookup_api_key(ps.key_hash(key)) is not None


async def test_plain_sha256_key_is_upgraded_to_hmac(db: Database) -> None:
    key = generate_api_key()
    key_id = await _stored(db, key, ps.key_hash_legacy(key))
    resolved = await _resolve_key(db, key, ps.key_hash_legacy(key))
    assert resolved is not None
    assert resolved[1] == key_id
    assert await db.lookup_api_key(ps.key_hash(key)) is not None


@pytest.mark.parametrize("pbkdf2", [True, False])
async def test_previous_secret_keys_authenticate_and_are_upgraded(
    db: Database, monkeypatch: pytest.MonkeyPatch, pbkdf2: bool
) -> None:
    key = generate_api_key()
    monkeypatch.setattr(settings, "session_secret", OLD_SECRET)
    old = ps.key_hash_pbkdf2(key, OLD_SECRET.encode()) if pbkdf2 else ps.key_hash(key)
    monkeypatch.setattr(settings, "session_secret", SECRET)
    key_id = await _stored(db, key, old)
    assert await _resolve_key(db, key, ps.key_hash_legacy(key)) is None
    monkeypatch.setattr(settings, "session_secret_previous", OLD_SECRET)
    resolved = await _resolve_key(db, key, ps.key_hash_legacy(key))
    assert resolved is not None
    assert resolved[1] == key_id
    assert await db.lookup_api_key(ps.key_hash(key)) is not None


async def test_current_hmac_key_needs_no_pbkdf2(
    db: Database, monkeypatch: pytest.MonkeyPatch
) -> None:
    key = generate_api_key()
    await _stored(db, key, ps.key_hash(key))

    def no_pbkdf2(_key: str, _pepper: bytes) -> str:
        raise AssertionError("current keys must not be PBKDF2-hashed")

    monkeypatch.setattr("llama_router.auth.key_hash_pbkdf2", no_pbkdf2)
    assert await _resolve_key(db, key, ps.key_hash_legacy(key)) is not None


def test_short_session_secret_is_rejected() -> None:
    with pytest.raises(ValidationError, match="at least 32 characters"):
        Settings(session_secret="x" * (MIN_SESSION_SECRET_LENGTH - 1))


def test_long_empty_and_previous_secrets_are_accepted() -> None:
    assert Settings(session_secret="x" * MIN_SESSION_SECRET_LENGTH).session_secret
    assert Settings(session_secret="").session_secret == ""
    # The previous secret is only read, so a short old one can be retired.
    assert Settings(session_secret=SECRET, session_secret_previous="short")  # noqa: S106
