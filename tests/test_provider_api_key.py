"""Optional per-provider upstream API keys."""

from __future__ import annotations

import httpx
import pytest

from llama_router.config import settings
from llama_router.models import ProviderStatus, ProviderType
from llama_router.provider_manager import ProviderManager
from llama_router.provider_secrets import (
    ApiKeyUnavailable,
    auth_headers,
    decrypt_api_key,
    encrypt_api_key,
    normalize_header,
)
from tests.test_dashboard_auth import dash_client  # noqa: F401  (fixture)

KEY = "sk-test-0123456789"


def test_encrypt_round_trip_and_not_plaintext() -> None:
    enc = encrypt_api_key(KEY)
    assert KEY not in enc
    assert decrypt_api_key(enc) == KEY


def test_changed_secret_reports_unavailable(monkeypatch) -> None:
    monkeypatch.setattr(settings, "session_secret", "first-secret-0123456789")
    enc = encrypt_api_key(KEY)
    monkeypatch.setattr(settings, "session_secret", "second-secret-012345678")
    with pytest.raises(ApiKeyUnavailable):
        decrypt_api_key(enc)


def test_auth_headers_shapes() -> None:
    assert auth_headers(KEY, None) == {"Authorization": f"Bearer {KEY}"}
    assert auth_headers(KEY, "authorization") == {"Authorization": f"Bearer {KEY}"}
    assert auth_headers(KEY, "API-KEY") == {"API-KEY": KEY}
    with pytest.raises(ValueError):
        normalize_header("Bad Header: x")


def _keyed_ollama(expect: dict[str, str]):
    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path == "/api/version":
            return httpx.Response(200, json={"version": "0.12.0"})
        for k, v in expect.items():
            if request.headers.get(k) != v:
                return httpx.Response(401, json={"error": "unauthorized"})
        if path == "/api/tags":
            return httpx.Response(200, json={"models": [{"name": "m:1b"}]})
        if path == "/api/ps":
            return httpx.Response(200, json={"models": []})
        return httpx.Response(404)

    return handler


@pytest.mark.httpx_mock(assert_all_responses_were_requested=False)
async def test_key_is_sent_upstream_and_brings_provider_online(db, httpx_mock) -> None:
    httpx_mock.add_callback(
        _keyed_ollama({"Authorization": f"Bearer {KEY}"}), is_reusable=True
    )
    pm = ProviderManager(db)

    without = await pm.add_provider("locked", "http://locked:11434")
    assert without.status == ProviderStatus.OFFLINE
    assert "authentication required" in (pm.status_detail(without.id) or "")

    await pm.update_provider(without.id, "locked", ProviderType.OLLAMA, api_key=KEY)
    p = await db.get_provider(without.id)
    assert p.status == ProviderStatus.IDLE
    assert p.has_api_key and KEY not in (p.api_key_enc or "")
    assert pm.status_detail(p.id) is None
    await pm.stop()


@pytest.mark.httpx_mock(assert_all_responses_were_requested=False)
async def test_custom_header_sends_raw_key(db, httpx_mock) -> None:
    httpx_mock.add_callback(_keyed_ollama({"API-KEY": KEY}), is_reusable=True)
    pm = ProviderManager(db)
    p = await pm.add_provider(
        "router", "http://router:11434", api_key=KEY, api_key_header="API-KEY"
    )
    assert p.status == ProviderStatus.IDLE
    assert (await db.get_provider(p.id)).api_key_header == "API-KEY"
    await pm.stop()


async def test_worker_provider_rejects_key(db) -> None:
    pm = ProviderManager(db)
    w = await pm.add_worker_provider("edge")
    with pytest.raises(ValueError):
        await pm._store_api_key(w, KEY, None)
    await pm.stop()


@pytest.mark.httpx_mock(assert_all_responses_were_requested=False)
async def test_dashboard_keeps_replaces_and_clears_key(
    dash_client, httpx_mock  # noqa: F811
) -> None:
    httpx_mock.add_callback(
        _keyed_ollama({"Authorization": f"Bearer {KEY}"}), is_reusable=True
    )
    client, db = dash_client
    r = await client.post(
        "/providers/add",
        data={"name": "gpu", "url": "http://gpu:11434", "api_key": KEY},
    )
    assert r.status_code == 303
    p = (await db.list_providers())[0]
    stored = p.api_key_enc
    assert stored and KEY not in stored

    page = await client.get(f"/providers/{p.id}")
    assert KEY not in page.text and stored not in page.text
    assert "Replace API key" in page.text

    form = {"name": "gpu", "provider_type": "ollama"}
    await client.post(f"/providers/{p.id}/edit", data=form)
    assert (await db.get_provider(p.id)).api_key_enc == stored  # blank keeps

    await client.post(f"/providers/{p.id}/edit", data={**form, "clear_api_key": "1"})
    assert (await db.get_provider(p.id)).api_key_enc is None


async def test_bad_header_rejects_edit_without_partial_write(db) -> None:
    pm = ProviderManager(db)
    p = await db.add_provider("gpu", ProviderType.OLLAMA, machine_type="old box")
    with pytest.raises(ValueError):
        await pm.update_provider(
            p.id,
            "renamed",
            ProviderType.LLAMACPP,
            machine_type="new box",
            api_key=KEY,
            api_key_header="Bad Header: x",
        )
    after = await db.get_provider(p.id)
    assert (after.name, after.provider_type, after.machine_type) == (
        "gpu",
        ProviderType.OLLAMA,
        "old box",
    )
    assert after.api_key_enc is None
    await pm.stop()
