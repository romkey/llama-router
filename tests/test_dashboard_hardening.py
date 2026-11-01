"""Dashboard hardening: escaping, non-admin limits, metadata addresses,
redacted status for non-admins."""

from __future__ import annotations

import asyncio
import ipaddress
import json
import socket
from pathlib import Path
from typing import Any

import httpx
import pytest

from llama_router import hf_download
from llama_router.config import settings
from llama_router.dashboard import deps as dash_deps
from llama_router.dashboard.auth_core import AUTH_COOKIE, hash_password, sign_session
from llama_router.dashboard.middleware import invalidate_dashboard_user_count_cache
from llama_router.dashboard.routes import REDACTED_ADDRESS
from llama_router.database import Database
from llama_router.llamacpp_client import OpenAIClient
from llama_router.models import ProviderModel, ProviderProtocol, ProviderStatus
from llama_router.probe import BlockedAddressError, check_not_metadata, is_metadata_ip
from llama_router.router import Router

DashClient = tuple[httpx.AsyncClient, Database]
TEMPLATES = Path(__file__).parent.parent / "src" / "llama_router" / "templates"


def _as(user_id: int) -> dict[str, str]:
    token = sign_session(user_id, settings.session_secret.strip())
    return {"Cookie": f"{AUTH_COOKIE}={token}"}


async def _users(db: Database) -> tuple[int, int]:
    admin = await db.create_dashboard_user("adm", hash_password("pw-123456789"), True)
    ann = await db.create_dashboard_user("ann", hash_password("pw-123456789"), False)
    invalidate_dashboard_user_count_cache()
    return admin, ann


async def _provider(db: Database, name: str, locality: str) -> int:
    """An online OpenAI-style provider serving "chatty", answering "ok"."""
    p = await db.add_provider(name, ProviderProtocol.OPENAI)
    pid = p.require_id()
    await db.update_provider_settings(pid, locality=locality)
    await db.update_provider_status(pid, ProviderStatus.IDLE)
    await db.set_provider_models(pid, [ProviderModel(provider_id=pid, name="chatty")])

    def handler(request: httpx.Request) -> httpx.Response:
        json.loads(request.content)
        return httpx.Response(
            200, json={"choices": [{"message": {"role": "assistant", "content": name}}]}
        )

    dash_deps.get_pm()._llamacpp_clients[pid] = OpenAIClient(
        f"http://{name}:8000", transport=httpx.MockTransport(handler)
    )
    return pid


# --- M1: escaping and pinned route names ---


def test_dashboard_template_escapes_interpolated_data() -> None:
    html = (TEMPLATES / "dashboard.html").read_text()
    for site in (
        "esc(p.model_name)",
        "esc(p.provider_name)",
        "esc(a.url)",
        "esc(md.name)",
        "esc(b.label)",
        "esc(p.status)",
        "esc(model)",
        "esc(err.message || 'Failed')",
    ):
        assert site in html, site
    # The pin's model name never lands inside the onclick JavaScript.
    assert "deleteKeyPin(encodeURIComponent(this.dataset.model))" in html
    # Server-rendered names go into handlers through tojson, not '{{ }}'.
    assert "removeFallback('{{" not in html
    assert "workerDelete({{ w.id }}, '{{" not in html
    detail = (TEMPLATES / "provider_detail.html").read_text()
    assert "esc(error)" in detail
    assert "confirm('Delete {{" not in detail


async def test_pin_model_name_is_validated(dash_client: DashClient) -> None:
    c, db = dash_client
    pid = await _provider(db, "gpu", "local")
    key_id = await db.create_api_key(
        key_prefix="sk-test", key_hash="h", routing_mode="balanced", allow_fallback=True
    )
    url = f"/api/keys/{key_id}/pins"

    for bad in ("x" * 256, "bad\nname", "tab\there", "nul\x00"):
        resp = await c.post(url, json={"model_name": bad, "provider_id": pid})
        assert resp.status_code == 400, bad
    resp = await c.post(url, json={"model_name": "chatty", "provider_id": "x"})
    assert resp.status_code == 400

    name = "<img src=x onerror=alert(1)>:latest"
    ok = await c.post(url, json={"model_name": name, "provider_id": pid})
    assert ok.status_code == 200
    ok = await c.post(url, json={"model_name": "m" * 255, "provider_id": pid})
    assert ok.status_code == 200
    names = {p["model_name"] for p in (await c.get(url)).json()["pins"]}
    assert names == {name, "m" * 255}


# --- M2: chat limits for non-admins, per-user caps ---


async def test_non_admin_chat_follows_self_service_locality(
    dash_client: DashClient,
) -> None:
    c, db = dash_client
    admin, ann = await _users(db)
    near = await _provider(db, "near", "local")
    far = await _provider(db, "far", "cloud")
    Router(db, dash_deps.get_pm())

    # Defaults: self-issued keys are local only, and so is chat.
    names = [
        p["name"]
        for p in (await c.get("/api/chat/targets", headers=_as(ann))).json()[
            "providers"
        ]
    ]
    assert names == ["near"]
    denied = await c.post(
        "/api/chat/send",
        json={"provider_id": far, "model": "chatty", "content": "hi"},
        headers=_as(ann),
    )
    assert denied.status_code == 403
    for _ in range(3):
        picked = await c.post(
            "/api/chat/send",
            json={"model": "chatty", "content": "hi"},
            headers=_as(ann),
        )
        assert picked.status_code == 200, picked.text
        assert picked.json()["messages"][1]["provider_name"] == "near"
    ok = await c.post(
        "/api/chat/send",
        json={"provider_id": near, "model": "chatty", "content": "hi"},
        headers=_as(ann),
    )
    assert ok.status_code == 200

    # Admins are not limited.
    ok = await c.post(
        "/api/chat/send",
        json={"provider_id": far, "model": "chatty", "content": "hi"},
        headers=_as(admin),
    )
    assert ok.status_code == 200

    # Any locality and no cap: cloud is allowed.
    await c.put("/api/keys/self-service", json={"locality": "any"}, headers=_as(admin))
    ok = await c.post(
        "/api/chat/send",
        json={"provider_id": far, "model": "chatty", "content": "hi"},
        headers=_as(ann),
    )
    assert ok.status_code == 200

    # A spend cap can't be tracked for chat: local only again.
    await c.put(
        "/api/keys/self-service",
        json={"locality": "any", "daily_spend_cap_usd": "1"},
        headers=_as(admin),
    )
    denied = await c.post(
        "/api/chat/send",
        json={"provider_id": far, "model": "chatty", "content": "hi"},
        headers=_as(ann),
    )
    assert denied.status_code == 403


async def test_chat_conversation_limit(
    dash_client: DashClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    c, db = dash_client
    _, ann = await _users(db)
    pid = await _provider(db, "near", "local")
    monkeypatch.setattr(settings, "max_chat_conversations_per_user", 2)
    body = {"provider_id": pid, "model": "chatty", "content": "hi"}
    first = await c.post("/api/chat/send", json=body, headers=_as(ann))
    assert first.status_code == 200
    assert (
        await c.post("/api/chat/send", json=body, headers=_as(ann))
    ).status_code == 200
    third = await c.post("/api/chat/send", json=body, headers=_as(ann))
    assert third.status_code == 409
    # Carrying on an existing conversation is fine.
    conv = first.json()["conversation"]["id"]
    more = await c.post(
        "/api/chat/send", json={**body, "conversation_id": conv}, headers=_as(ann)
    )
    assert more.status_code == 200


async def test_self_issued_key_limit(
    dash_client: DashClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    c, db = dash_client
    admin, ann = await _users(db)
    monkeypatch.setattr(settings, "max_api_keys_per_user", 2)
    for _ in range(2):
        resp = await c.post("/api/my-keys", json={}, headers=_as(ann))
        assert resp.status_code == 200
    resp = await c.post("/api/my-keys", json={}, headers=_as(ann))
    assert resp.status_code == 409
    # Keys an admin issues on the API Keys tab don't count.
    resp = await c.post("/api/keys/generate", json={}, headers=_as(admin))
    assert resp.status_code == 200


# --- M14: Hugging Face environment ---


async def test_hf_env_is_admin_only_and_masks_proxy_credentials(
    dash_client: DashClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    c, db = dash_client
    admin, ann = await _users(db)
    monkeypatch.setenv("HTTPS_PROXY", "http://user:s3cr@t@proxy.lan:3128")
    assert (await c.get("/api/hf/env", headers=_as(ann))).status_code == 403
    resp = await c.get("/api/hf/env", headers=_as(admin))
    assert resp.status_code == 200
    env = {v["name"]: v["value"] for v in resp.json()["env"]}
    assert env["HTTPS_PROXY"] == "http://***@proxy.lan:3128"
    assert "s3cr" not in resp.text


@pytest.mark.parametrize(
    ("raw", "masked"),
    [
        ("http://proxy:3128", "http://proxy:3128"),
        ("http://u:p@proxy:3128/x", "http://***@proxy:3128/x"),
        ("socks5h://only-user@h", "socks5h://***@h"),
        ("localhost,.lan", "localhost,.lan"),
    ],
)
def test_mask_userinfo(raw: str, masked: str) -> None:
    assert hf_download.mask_userinfo(raw) == masked


# --- M5: cloud metadata addresses ---


@pytest.mark.parametrize(
    ("ip", "blocked"),
    [
        ("169.254.169.254", True),
        ("169.254.0.1", True),
        ("fd00:ec2::254", True),
        ("::ffff:169.254.169.254", True),
        ("127.0.0.1", False),
        ("10.0.0.5", False),
        ("192.168.1.10", False),
        ("::1", False),
        ("fd00:ec2::253", False),
    ],
)
def test_is_metadata_ip(ip: str, blocked: bool) -> None:
    assert is_metadata_ip(ipaddress.ip_address(ip)) is blocked


async def test_check_not_metadata_resolves_names(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for url in ("http://169.254.169.254/latest", "http://[fd00:ec2::254]/"):
        with pytest.raises(BlockedAddressError):
            await check_not_metadata(url)
    await check_not_metadata("http://127.0.0.1:11434")
    await check_not_metadata("http://localhost:11434")

    loop = asyncio.get_running_loop()

    async def fake_getaddrinfo(host: str, *_a: Any, **_k: Any) -> list[Any]:
        if host == "sneaky.example":
            return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("169.254.169.254", 0))]
        raise socket.gaierror("no such host")

    monkeypatch.setattr(loop, "getaddrinfo", fake_getaddrinfo)
    with pytest.raises(BlockedAddressError):
        await check_not_metadata("http://sneaky.example:8080")
    # A name that doesn't resolve (yet) passes; the probe re-checks later.
    await check_not_metadata("http://offline.example:8080")


async def test_provider_routes_refuse_metadata_addresses(
    dash_client: DashClient,
) -> None:
    c, db = dash_client
    meta = "http://169.254.169.254"
    resp = await c.post("/api/providers/probe", json={"url": meta})
    assert resp.status_code == 400
    assert "metadata" in resp.json()["detail"]
    resp = await c.post(
        "/providers/add", data={"name": "m", "url": meta, "protocol": "ollama"}
    )
    assert resp.status_code == 400
    assert "metadata" in resp.json()["detail"]
    assert await db.list_providers() == []

    resp = await c.post(
        "/providers/add",
        data={"name": "ok", "url": "http://127.0.0.1:9", "protocol": "ollama"},
    )
    assert resp.status_code == 303
    [p] = await db.list_providers()
    pid = p.require_id()
    resp = await c.post(f"/providers/{pid}/addresses/add", data={"url": meta})
    assert resp.status_code == 400
    resp = await c.post(f"/providers/{pid}/addresses/save", data={"url": meta})
    assert resp.status_code == 400
    [addr] = await db.get_addresses(pid)
    resp = await c.post(
        f"/providers/{pid}/addresses/{addr.id}/edit", data={"url": meta}
    )
    assert resp.status_code == 400
    assert [a.url for a in await db.get_addresses(pid)] == ["http://127.0.0.1:9"]


# --- LOW: status redaction ---


async def test_status_hides_addresses_from_non_admins(dash_client: DashClient) -> None:
    c, db = dash_client
    admin, ann = await _users(db)
    p = await db.add_provider("gpu", ProviderProtocol.OLLAMA)
    await db.add_address(p.require_id(), "http://10.1.2.3:11434", True)

    seen = (await c.get("/api/status", headers=_as(ann))).json()
    [prov] = seen["providers"]
    assert [a["url"] for a in prov["addresses"]] == [REDACTED_ADDRESS]
    assert "10.1.2.3" not in json.dumps(seen)
    page = await c.get("/", headers=_as(ann))
    assert "10.1.2.3" not in page.text

    full = (await c.get("/api/status", headers=_as(admin))).json()
    assert [a["url"] for a in full["providers"][0]["addresses"]] == [
        "http://10.1.2.3:11434"
    ]
