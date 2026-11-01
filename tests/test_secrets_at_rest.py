"""The generated secret file, legacy-key migration and Sentry scrubbing."""

from __future__ import annotations

import base64
import hashlib
import stat
from pathlib import Path

import pytest
from cryptography.fernet import Fernet

from llama_router import provider_secrets as ps
from llama_router.auth import (
    _resolve_key,
    generate_api_key,
    key_hash,
    key_hash_legacy_pepper,
    key_prefix,
)
from llama_router.config import settings
from llama_router.database import Database
from llama_router.models import ProviderProtocol
from llama_router.provider_manager import ProviderManager
from llama_router.sentry import FILTERED, scrub_event
from llama_router.worker_link.worker import WorkerLinkManager


def _legacy_encrypt(plaintext: str) -> str:
    digest = hashlib.sha256(ps._CONTEXT + ps._LEGACY_DEV_SECRET).digest()
    f = Fernet(base64.urlsafe_b64encode(digest))
    return f.encrypt(plaintext.encode()).decode()


@pytest.fixture
def no_session_secret(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "session_secret", "")


def test_secret_file_created_0600_and_reused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, no_session_secret: None
) -> None:
    path = tmp_path / "sub" / "secret.key"
    monkeypatch.setattr(settings, "secret_key_path", str(path))
    enc = ps.encrypt_api_key("sk-1")
    assert path.exists()
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    first = path.read_bytes()
    # A fresh process (empty cache) reads the same file back.
    monkeypatch.setattr(ps, "_local_secrets", {})
    assert ps.decrypt_api_key(enc) == "sk-1"
    assert path.read_bytes() == first


def test_secret_file_not_used_with_session_secret(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "secret.key"
    monkeypatch.setattr(settings, "secret_key_path", str(path))
    monkeypatch.setattr(settings, "session_secret", "configured-secret-0123456789")
    ps.decrypt_api_key(ps.encrypt_api_key("sk-1"))
    key_hash("lrk_x")
    assert not path.exists()


def test_default_path_is_beside_database(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(settings, "secret_key_path", "")
    monkeypatch.setattr(settings, "database_path", str(tmp_path / "llama_router.db"))
    assert ps.secret_key_path() == tmp_path.resolve() / "secret.key"


def test_lost_secret_file_makes_keys_unavailable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, no_session_secret: None
) -> None:
    enc = ps.encrypt_api_key("sk-1")
    monkeypatch.setattr(settings, "secret_key_path", str(tmp_path / "other.key"))
    with pytest.raises(ps.ApiKeyUnavailableError):
        ps.decrypt_api_key(enc)


def test_legacy_dev_key_still_decrypts_and_reencrypts(no_session_secret: None) -> None:
    old = _legacy_encrypt("sk-old")
    assert ps.decrypt_api_key(old) == "sk-old"
    fresh = ps.reencrypt_legacy(old)
    assert fresh is not None
    assert fresh != old
    assert ps.reencrypt_legacy(fresh) is None
    assert ps.decrypt_api_key(fresh) == "sk-old"


def test_setting_round_trip_and_plaintext_compat() -> None:
    stored = ps.encrypt_setting("lrw_tok")
    assert stored.startswith(ps.ENCRYPTED_SETTING_PREFIX)
    assert ps.decrypt_setting(stored) == ("lrw_tok", False)
    assert ps.decrypt_setting("lrw_plain") == ("lrw_plain", True)


async def test_plaintext_worker_token_is_resaved_encrypted(
    db: Database, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(settings, "worker_hub_url", "")
    monkeypatch.setattr(settings, "worker_token", "")
    await db.set_app_setting(WorkerLinkManager.DB_HUB_URL, "http://hub:11434")
    await db.set_app_setting(WorkerLinkManager.DB_TOKEN, "lrw_plain")
    link = WorkerLinkManager(db)
    url, token, source = await link.resolved_config()
    assert (url, token, source) == ("http://hub:11434", "lrw_plain", "dashboard")
    stored = await db.get_app_setting(WorkerLinkManager.DB_TOKEN)
    assert stored
    assert stored.startswith(ps.ENCRYPTED_SETTING_PREFIX)
    assert (await link.resolved_config())[1] == "lrw_plain"


async def test_startup_reencrypts_legacy_provider_keys(
    db: Database, no_session_secret: None
) -> None:
    pid = (await db.add_provider("cloud", ProviderProtocol.OPENAI)).require_id()
    old = _legacy_encrypt("sk-old")
    await db.set_provider_api_key(pid, old, None)
    pm = ProviderManager(db)
    await pm._reencrypt_legacy_keys(await db.list_providers())
    prov = await db.get_provider(pid)
    assert prov is not None
    assert prov.api_key_enc
    assert prov.api_key_enc != old
    assert ps.decrypt_api_key(prov.api_key_enc) == "sk-old"


async def test_legacy_pepper_key_matches_and_is_rehashed(
    db: Database, no_session_secret: None
) -> None:
    key = generate_api_key()
    key_id = await db.create_api_key(
        key_prefix=key_prefix(key),
        key_hash=key_hash_legacy_pepper(key),
        routing_mode="auto",
        allow_fallback=True,
    )
    resolved = await _resolve_key(db, key, hashlib.sha256(key.encode()).hexdigest())
    assert resolved is not None
    assert resolved[1] == key_id
    assert await db.lookup_api_key(key_hash(key)) is not None
    assert await db.lookup_api_key(key_hash_legacy_pepper(key)) is None


def test_scrub_event_drops_request_parts_and_masks_secrets() -> None:
    event = {
        "request": {
            "url": "http://x/api/chat",
            "method": "POST",
            "data": {"prompt": "secret plans"},
            "headers": {"Authorization": "Bearer sk-1"},
            "cookies": {"session": "abc"},
            "query_string": "api_key=sk-2",
        },
        "extra": {
            "API_KEY": "sk-3",
            "nested": [{"X-Auth-Token": "t"}, {"password": "p", "ok": 1}],
            "Client_Secret": "s",
        },
        "breadcrumbs": {"values": [{"data": {"authorization": "Bearer z"}}]},
    }
    out = scrub_event(event, {})
    assert out["request"] == {"url": "http://x/api/chat", "method": "POST"}
    assert out["extra"]["API_KEY"] == FILTERED
    assert out["extra"]["Client_Secret"] == FILTERED
    assert out["extra"]["nested"][0]["X-Auth-Token"] == FILTERED
    assert out["extra"]["nested"][1] == {"password": FILTERED, "ok": 1}
    assert out["breadcrumbs"]["values"][0]["data"]["authorization"] == FILTERED
