"""Dashboard chat: pick a model on a provider and talk to it."""

from __future__ import annotations

import json
from typing import Any

import httpx

from llama_router.config import settings
from llama_router.dashboard import deps as dash_deps
from llama_router.dashboard.auth_core import AUTH_COOKIE, hash_password, sign_session
from llama_router.dashboard.chat import chat_messages, reply_image, reply_text
from llama_router.dashboard.middleware import invalidate_dashboard_user_count_cache
from llama_router.database import Database
from llama_router.llamacpp_client import OpenAIClient
from llama_router.models import ProviderModel, ProviderProtocol, ProviderStatus
from llama_router.router import Router

DashClient = tuple[httpx.AsyncClient, Database]
IMAGE = "data:image/png;base64,iVBORw0KGgo="


async def _backend(db: Database, seen: list[dict[str, Any]]) -> int:
    """An OpenAI-style provider serving "chatty" (backend name "chatty:q4")
    whose answers are faked; request bodies land in *seen*."""
    p = await db.add_provider("gpu", ProviderProtocol.OPENAI)
    pid = p.require_id()
    await db.update_provider_status(pid, ProviderStatus.IDLE)
    await db.set_provider_models(
        pid, [ProviderModel(provider_id=pid, name="chatty", raw_name="chatty:q4")]
    )

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        seen.append(body)
        if request.url.path.endswith("/images/generations"):
            return httpx.Response(200, json={"data": [{"b64_json": "QUJD"}]})
        if body["messages"][-1]["content"] == "fail":
            return httpx.Response(500, json={"error": "boom"})
        n = len(body["messages"])
        return httpx.Response(
            200,
            json={"choices": [{"message": {"role": "assistant", "content": f"#{n}"}}]},
        )

    pm = dash_deps.get_pm()
    pm._llamacpp_clients[pid] = OpenAIClient(
        "http://gpu:8000", transport=httpx.MockTransport(handler)
    )
    return pid


def _as(user_id: int) -> dict[str, str]:
    token = sign_session(user_id, settings.session_secret.strip())
    return {"Cookie": f"{AUTH_COOKIE}={token}"}


async def test_targets_list_online_providers(dash_client: DashClient) -> None:
    c, db = dash_client
    await _backend(db, [])
    off = await db.add_provider("off", ProviderProtocol.OPENAI)
    await db.update_provider_status(off.require_id(), ProviderStatus.OFFLINE)
    resp = await c.get("/api/chat/targets")
    assert resp.status_code == 200
    assert [(p["name"], p["models"]) for p in resp.json()["providers"]] == [
        ("gpu", ["chatty"])
    ]
    assert resp.json()["any"] == {"models": ["chatty"], "groups": [], "aliases": []}
    page = await c.get("/chat")
    assert page.status_code == 200
    assert "chat-provider" in page.text


async def test_conversation_keeps_history(dash_client: DashClient) -> None:
    c, db = dash_client
    seen: list[dict[str, Any]] = []
    pid = await _backend(db, seen)
    resp = await c.post(
        "/api/chat/send",
        json={"provider_id": pid, "model": "chatty", "content": "hello\nthere"},
    )
    assert resp.status_code == 200
    data = resp.json()
    conv = data["conversation"]
    assert conv["title"] == "hello"
    assert conv["model_name"] == "chatty"
    user, reply = data["messages"]
    assert (user["role"], reply["role"], reply["content"]) == (
        "user",
        "assistant",
        "#1",
    )
    assert reply["provider_name"] == "gpu"
    assert seen[0]["model"] == "chatty:q4"

    resp = await c.post(
        "/api/chat/send",
        json={
            "conversation_id": conv["id"],
            "provider_id": pid,
            "model": "chatty",
            "content": "what is this?",
            "image": IMAGE,
        },
    )
    assert resp.json()["messages"][1]["content"] == "#3"
    sent = seen[1]["messages"]
    assert [m["role"] for m in sent] == ["user", "assistant", "user"]
    assert sent[2]["content"][1] == {"type": "image_url", "image_url": {"url": IMAGE}}

    listed = (await c.get("/api/chat/conversations")).json()["conversations"]
    assert [x["id"] for x in listed] == [conv["id"]]
    full = (await c.get(f"/api/chat/conversations/{conv['id']}")).json()
    assert len(full["messages"]) == 4
    assert full["messages"][2]["image"] == IMAGE

    assert (await c.delete(f"/api/chat/conversations/{conv['id']}")).status_code == 200
    assert (await c.get(f"/api/chat/conversations/{conv['id']}")).status_code == 404


async def test_backend_error_is_kept_and_left_out_of_history(
    dash_client: DashClient,
) -> None:
    c, db = dash_client
    seen: list[dict[str, Any]] = []
    pid = await _backend(db, seen)
    target = {"provider_id": pid, "model": "chatty"}
    resp = await c.post("/api/chat/send", json={**target, "content": "fail"})
    assert resp.status_code == 200
    conv_id = resp.json()["conversation"]["id"]
    reply = resp.json()["messages"][1]
    assert reply["role"] == "error"
    assert "500" in reply["content"]
    resp = await c.post(
        "/api/chat/send", json={**target, "content": "ok", "conversation_id": conv_id}
    )
    # The failed turn's user message stays; the error does not go upstream.
    assert [m["role"] for m in seen[1]["messages"]] == ["user", "user"]


async def test_image_generation(dash_client: DashClient) -> None:
    c, db = dash_client
    seen: list[dict[str, Any]] = []
    pid = await _backend(db, seen)
    resp = await c.post(
        "/api/chat/send",
        json={
            "provider_id": pid,
            "model": "chatty",
            "content": "a cat",
            "mode": "image",
        },
    )
    reply = resp.json()["messages"][1]
    assert reply["role"] == "assistant"
    assert reply["image"] == "data:image/png;base64,QUJD"
    assert seen == [{"model": "chatty:q4", "prompt": "a cat", "n": 1}]


async def test_bad_requests(dash_client: DashClient) -> None:
    c, db = dash_client
    pid = await _backend(db, [])
    ok = {"provider_id": pid, "model": "chatty", "content": "hi"}
    cases = [
        ({**ok, "model": "nope"}, 404),
        ({**ok, "provider_id": 999}, 404),
        ({**ok, "content": ""}, 400),
        ({**ok, "mode": "video"}, 400),
        ({**ok, "image": "http://example.com/x.png"}, 400),
        ({**ok, "conversation_id": 12345}, 404),
    ]
    for body, status in cases:
        resp = await c.post("/api/chat/send", json=body)
        assert resp.status_code == status, (body, resp.text)


async def test_each_user_sees_only_their_chats(dash_client: DashClient) -> None:
    c, db = dash_client
    pid = await _backend(db, [])
    admin = await db.create_dashboard_user("adm", hash_password("pw-123456789"), True)
    viewer = await db.create_dashboard_user("ann", hash_password("pw-123456789"), False)
    invalidate_dashboard_user_count_cache()

    # A non-admin may chat, though other writes are admin-only.
    resp = await c.post(
        "/api/chat/send",
        json={"provider_id": pid, "model": "chatty", "content": "mine"},
        headers=_as(viewer),
    )
    assert resp.status_code == 200
    conv_id = resp.json()["conversation"]["id"]

    others = await c.get("/api/chat/conversations", headers=_as(admin))
    assert others.json()["conversations"] == []
    peek = await c.get(f"/api/chat/conversations/{conv_id}", headers=_as(admin))
    assert peek.status_code == 404
    gone = await c.delete(f"/api/chat/conversations/{conv_id}", headers=_as(admin))
    assert gone.status_code == 404
    mine = await c.get("/api/chat/conversations", headers=_as(viewer))
    assert [x["id"] for x in mine.json()["conversations"]] == [conv_id]


async def _names(db: Database) -> Router:
    """A router, with the alias "chat" for chatty and the group "team"."""
    await db.set_model_alias("chat", "chatty")
    group = await db.create_model_group("team")
    assert group.id is not None
    await db.add_model_group_member(group.id, "chatty")
    return Router(db, dash_deps.get_pm())


async def test_targets_list_groups_and_aliases(dash_client: DashClient) -> None:
    c, db = dash_client
    pid = await _backend(db, [])
    other = await db.add_provider("cpu", ProviderProtocol.OPENAI)
    oid = other.require_id()
    await db.update_provider_status(oid, ProviderStatus.IDLE)
    await db.set_provider_models(oid, [ProviderModel(provider_id=oid, name="tiny")])
    await _names(db)
    data = (await c.get("/api/chat/targets")).json()
    by_id = {p["id"]: p for p in data["providers"]}
    assert by_id[pid]["groups"] == ["team"]
    assert by_id[pid]["aliases"] == ["chat"]
    # Neither the group nor the alias reaches the other provider.
    assert (by_id[oid]["groups"], by_id[oid]["aliases"]) == ([], [])
    assert data["any"] == {
        "models": ["chatty", "tiny"],
        "groups": ["team"],
        "aliases": ["chat"],
    }


async def test_any_provider_lets_the_router_pick(dash_client: DashClient) -> None:
    c, db = dash_client
    seen: list[dict[str, Any]] = []
    await _backend(db, seen)
    await _names(db)
    resp = await c.post("/api/chat/send", json={"model": "team", "content": "hi"})
    assert resp.status_code == 200, resp.text
    assert seen[0]["model"] == "chatty:q4"
    reply = resp.json()["messages"][1]
    assert (reply["model_name"], reply["provider_name"]) == ("team → chatty", "gpu")
    # The chat remembers "any provider", not the router's pick.
    conv = resp.json()["conversation"]
    assert (conv["provider_id"], conv["model_name"]) == (None, "team")

    resp = await c.post("/api/chat/send", json={"model": "nope", "content": "hi"})
    assert resp.status_code == 404
    assert "No provider" in resp.json()["detail"]


async def test_group_and_alias_on_a_chosen_provider(dash_client: DashClient) -> None:
    c, db = dash_client
    seen: list[dict[str, Any]] = []
    pid = await _backend(db, seen)
    await _names(db)
    for name in ("team", "chat"):
        resp = await c.post(
            "/api/chat/send",
            json={"provider_id": pid, "model": name, "content": "hi"},
        )
        assert resp.status_code == 200, resp.text
        assert resp.json()["conversation"]["provider_id"] == pid
    assert [b["model"] for b in seen] == ["chatty:q4", "chatty:q4"]

    other = await db.add_provider("cpu", ProviderProtocol.OPENAI)
    oid = other.require_id()
    await db.update_provider_status(oid, ProviderStatus.IDLE)
    await db.set_provider_models(oid, [ProviderModel(provider_id=oid, name="tiny")])
    resp = await c.post(
        "/api/chat/send", json={"provider_id": oid, "model": "team", "content": "hi"}
    )
    assert resp.status_code == 404


async def test_group_hides_a_model_on_a_chosen_provider(
    dash_client: DashClient,
) -> None:
    c, db = dash_client
    seen: list[dict[str, Any]] = []
    pid = await _backend(db, seen)
    await db.set_provider_models(
        pid,
        [
            ProviderModel(provider_id=pid, name="chatty", raw_name="chatty:q4"),
            ProviderModel(provider_id=pid, name="big"),
        ],
    )
    # A group named like a served model; requests for "big" go to the group.
    group = await db.create_model_group("big")
    assert group.id is not None
    await db.add_model_group_member(group.id, "chatty")
    Router(db, dash_deps.get_pm())
    resp = await c.post(
        "/api/chat/send", json={"provider_id": pid, "model": "big", "content": "hi"}
    )
    assert resp.status_code == 200, resp.text
    assert seen[0]["model"] == "chatty:q4"


def test_chat_messages_and_replies() -> None:
    history: list[dict[str, Any]] = [
        {"role": "user", "content": "draw", "image": None},
        {"role": "assistant", "content": "", "image": "data:image/png;base64,x"},
        {"role": "error", "content": "boom", "image": None},
    ]
    assert chat_messages(history) == [
        {"role": "user", "content": "draw"},
        {"role": "assistant", "content": "[image]"},
    ]
    thinking = {"choices": [{"message": {"content": "", "reasoning_content": "hm"}}]}
    assert reply_text(thinking) == "hm"
    parts = {"choices": [{"message": {"content": [{"text": "a"}, {"text": "b"}]}}]}
    assert reply_text(parts) == "ab"
    assert reply_text({}) == ""
    assert reply_image({"data": [{"url": "http://x/y.png"}]}) == ("http://x/y.png", "")
    assert reply_image({"data": []}) == (None, "")
