"""Dashboard chat: talk to a model, for trying models out.

The model can be a served model, a model group or an alias. With a provider
chosen the request goes to it; without one the router picks, as it would for
an API request. Either way it goes through the provider's OpenAI-compatible
endpoints: ``/v1/chat/completions`` for chat, with an
attached image sent as an ``image_url`` part, and ``/v1/images/generations``
when asking for an image. Each dashboard user keeps their own history.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import httpx
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, Response

from ..database import Database
from ..httpx_errors import describe_httpx_error
from ..models import Provider, ProviderStatus
from ..request_logger import log_request_background
from ..router import Router
from ..routing_table import RoutingTable, alias_groups
from ..v1_client import get_v1_client
from . import deps
from .routes import merge_dash_template_ctx, templates

router = APIRouter()

MODES = ("chat", "image")
MAX_CONTENT = 100_000
# A data URL; base64 makes it a third larger than the image.
MAX_IMAGE_CHARS = 14_000_000
TITLE_LEN = 60
PATHS = {"chat": "/v1/chat/completions", "image": "/v1/images/generations"}
NAME_KINDS = ("models", "groups", "aliases")


def _user_id(request: Request) -> int | None:
    user = getattr(request.state, "dashboard_user", None)
    return user.id if user is not None else None


async def _body(request: Request) -> dict[str, Any]:
    try:
        body = await request.json()
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="body must be JSON") from exc
    if not isinstance(body, dict):
        raise HTTPException(status_code=400, detail="body must be a JSON object")
    return body


@router.get("/chat", response_class=HTMLResponse)
async def chat_page(request: Request) -> Response:
    """Chat with a model on a provider."""
    return templates.TemplateResponse(
        request, "chat.html", merge_dash_template_ctx(request, {})
    )


async def _table() -> RoutingTable | None:
    rt = deps.get_pm().router
    return await rt.current_table() if rt is not None else None


@router.get("/api/chat/targets")
async def chat_targets() -> JSONResponse:
    """Providers that are not offline, each with the models it serves and
    the model groups and aliases that reach it, and under "any" every name
    the router can place on one of them."""
    db = deps.get_db()
    served: dict[int, set[str]] = {}
    for m in await db.get_all_provider_models():
        served.setdefault(m.provider_id, set()).add(m.name)
    aliases = alias_groups(await db.get_model_aliases())
    table = await _table()
    group_names = set(table.groups) if table is not None else set()
    groups: dict[int, set[str]] = {}
    if table is not None:
        for (name, _path), cands in table.all_candidates.items():
            if name in group_names:
                for c in cands:
                    groups.setdefault(c.provider_id, set()).add(name)
    rows: list[dict[str, Any]] = []
    for p in await db.list_providers():
        if p.id is None or p.status == ProviderStatus.OFFLINE:
            continue
        # A group hides a model or alias of the same name.
        models = served.get(p.id, set()) - group_names
        if not models:
            continue
        reach: set[str] = set()
        for n in models:
            reach |= aliases.get(n, frozenset())
        rows.append(
            {
                "id": p.id,
                "name": p.name,
                "status": p.status.value,
                "models": sorted(models),
                "groups": sorted(groups.get(p.id, set())),
                "aliases": sorted(reach - models - group_names),
            }
        )
    anywhere = {k: set().union(*(r[k] for r in rows)) for k in NAME_KINDS}
    anywhere["aliases"] -= anywhere["models"]
    return JSONResponse(
        {
            "providers": rows,
            "any": {k: sorted(v) for k, v in anywhere.items()},
        }
    )


@router.get("/api/chat/conversations")
async def list_conversations(request: Request) -> JSONResponse:
    rows = await deps.get_db().list_chat_conversations(_user_id(request))
    return JSONResponse({"conversations": rows})


@router.get("/api/chat/conversations/{conversation_id}")
async def get_conversation(conversation_id: int, request: Request) -> JSONResponse:
    conv = await deps.get_db().get_chat_conversation(conversation_id, _user_id(request))
    if conv is None:
        raise HTTPException(status_code=404, detail="conversation not found")
    return JSONResponse(conv)


@router.delete("/api/chat/conversations/{conversation_id}")
async def delete_conversation(conversation_id: int, request: Request) -> JSONResponse:
    if not await deps.get_db().delete_chat_conversation(
        conversation_id, _user_id(request)
    ):
        raise HTTPException(status_code=404, detail="conversation not found")
    return JSONResponse({"status": "ok"})


@dataclass(slots=True)
class Target:
    provider: Provider
    # The name the backend knows the model by.
    backend_model: str
    # The model that answers: a group's member, an alias's model.
    served: str
    # The user chose the provider; otherwise the router did.
    chosen: bool


def _not_found(model: str, provider: Provider | None) -> HTTPException:
    where = provider.name if provider is not None else "No provider"
    return HTTPException(status_code=404, detail=f"{where} does not serve {model}")


async def _target(db: Database, body: dict[str, Any], model: str, path: str) -> Target:
    """Where *model* goes: the chosen provider, or the router's pick."""
    raw = body.get("provider_id")
    if raw in (None, ""):
        rt = deps.get_pm().router
        result = await rt.route(model, path) if rt is not None else None
        if result is None:
            raise _not_found(model, None)
        return Target(result.provider, result.backend_model, result.model_key, False)
    try:
        pid = int(raw)
    except (TypeError, ValueError) as exc:
        raise HTTPException(
            status_code=400, detail="provider_id must be a number"
        ) from exc
    provider = await db.get_provider(pid)
    if provider is None:
        raise HTTPException(status_code=404, detail="provider not found")
    for m in await db.get_provider_models(pid):
        if m.name == model:
            return Target(provider, m.backend_name, model, True)
    # A model group or an alias with a member on this provider.
    table = await _table()
    if table is not None and any(
        c.provider_id == pid for c in table.all_candidates.get((model, path), ())
    ):
        c = Router._pinned(table, model, path, pid)
        if c is not None:
            return Target(provider, c.backend_model, c.model, True)
    raise _not_found(model, provider)


def _image(body: dict[str, Any]) -> str | None:
    raw = body.get("image")
    if raw in (None, ""):
        return None
    if not isinstance(raw, str) or not raw.startswith("data:image/"):
        raise HTTPException(status_code=400, detail="image must be an image data URL")
    if len(raw) > MAX_IMAGE_CHARS:
        raise HTTPException(status_code=413, detail="image is too large")
    return raw


def chat_messages(history: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The conversation as OpenAI chat messages. Errors are left out, and so
    are generated images: only images the user attached go to the model."""
    out: list[dict[str, Any]] = []
    for m in history:
        if m["role"] == "user" and m.get("image"):
            out.append(
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": m["content"]},
                        {"type": "image_url", "image_url": {"url": m["image"]}},
                    ],
                }
            )
        elif m["role"] in ("user", "assistant"):
            text = m["content"]
            if m["role"] == "assistant" and m.get("image") and not text:
                text = "[image]"
            out.append({"role": m["role"], "content": text})
    return out


def reply_text(resp: dict[str, Any]) -> str:
    """The assistant's text from a chat completion."""
    choices = resp.get("choices") or []
    if not choices:
        return ""
    msg = choices[0].get("message") or {}
    content = msg.get("content")
    if isinstance(content, list):
        content = "".join(
            str(p.get("text", "")) for p in content if isinstance(p, dict)
        )
    text = str(content or "")
    if not text.strip():
        # Reasoning models sometimes spend every token thinking.
        text = str(msg.get("reasoning_content") or msg.get("reasoning") or "")
    return text


def reply_image(resp: dict[str, Any]) -> tuple[str | None, str]:
    """The image (a data URL, or the upstream URL) and any revised prompt."""
    data = resp.get("data") or []
    if not data or not isinstance(data[0], dict):
        return None, ""
    first = data[0]
    revised = str(first.get("revised_prompt") or "")
    if first.get("b64_json"):
        return f"data:image/png;base64,{first['b64_json']}", revised
    if first.get("url"):
        return str(first["url"]), revised
    return None, revised


@router.post("/api/chat/send")
async def send(request: Request) -> JSONResponse:
    """Body: {conversation_id?, provider_id?, model, content, image?, mode?}.

    *model* is a model, a model group or an alias. Without *provider_id*
    the router picks the provider. *mode* is "chat" (default) or "image"
    (generate an image from *content*).
    Without *conversation_id* a new conversation is started. Returns the
    conversation and the two new messages: the user's and the reply, which
    is an "error" message when the backend failed.
    """
    body = await _body(request)
    db = deps.get_db()
    pm = deps.get_pm()
    uid = _user_id(request)
    mode = str(body.get("mode") or "chat")
    if mode not in MODES:
        raise HTTPException(status_code=400, detail=f"mode must be one of {MODES}")
    content = str(body.get("content") or "").strip()
    if len(content) > MAX_CONTENT:
        raise HTTPException(status_code=413, detail="message is too long")
    image = _image(body) if mode == "chat" else None
    if not content and image is None:
        raise HTTPException(status_code=400, detail="content is required")
    model = str(body.get("model") or "").strip()
    if not model:
        raise HTTPException(status_code=400, detail="model is required")
    target = await _target(db, body, model, PATHS[mode])
    provider, backend_model = target.provider, target.backend_model
    pid = provider.require_id()
    # Reopening the chat brings back the choice made, not the router's pick.
    chosen_pid = pid if target.chosen else None
    answered_by = model if target.served == model else f"{model} → {target.served}"

    conv_id = body.get("conversation_id")
    if conv_id in (None, ""):
        title = (content or "Image").splitlines()[0][:TITLE_LEN]
        conv_id = await db.create_chat_conversation(uid, title, chosen_pid, model)
        history: list[dict[str, Any]] = []
    else:
        try:
            conv_id = int(conv_id)
        except (TypeError, ValueError) as exc:
            raise HTTPException(
                status_code=400, detail="conversation_id must be a number"
            ) from exc
        conv = await db.get_chat_conversation(conv_id, uid)
        if conv is None:
            raise HTTPException(status_code=404, detail="conversation not found")
        history = conv["messages"]

    user_msg = await db.add_chat_message(
        conv_id, "user", content, image=image, model_name=model
    )
    endpoint = PATHS[mode]
    if mode == "image":
        upstream: dict[str, Any] = {"model": backend_model, "prompt": content, "n": 1}
    else:
        upstream = {
            "model": backend_model,
            "messages": chat_messages([*history, user_msg]),
            "stream": False,
        }

    client = get_v1_client(pm, pid)
    start = time.monotonic()
    log: dict[str, Any] = {
        "provider": provider,
        "protocol": "v1",
        "endpoint": endpoint,
        "request": request,
        "model": model,
        "request_body": upstream,
        "request_meta": f"dashboard chat ({mode})",
        "sample": False,
    }
    try:
        async with pm.acquire_provider(pid):
            if mode == "image":
                resp = await client.images_generations(upstream)
            else:
                resp = await client.chat_completions(upstream)
    except Exception as exc:
        duration = (time.monotonic() - start) * 1000
        detail = (
            describe_httpx_error(exc)
            if isinstance(exc, httpx.HTTPError)
            else str(exc) or type(exc).__name__
        )
        log_request_background(
            db, duration_ms=duration, status="error", error_detail=detail, **log
        )
        reply = await db.add_chat_message(
            conv_id,
            "error",
            detail,
            model_name=answered_by,
            provider_name=provider.name,
            duration_ms=duration,
        )
    else:
        duration = (time.monotonic() - start) * 1000
        log_request_background(db, duration_ms=duration, response_body=resp, **log)
        if mode == "image":
            out_image, text = reply_image(resp)
            if out_image is None:
                text = text or "The backend returned no image."
        else:
            out_image, text = None, reply_text(resp)
        reply = await db.add_chat_message(
            conv_id,
            "assistant" if mode == "chat" or out_image else "error",
            text,
            image=out_image,
            model_name=answered_by,
            provider_name=provider.name,
            duration_ms=duration,
        )
    await db.touch_chat_conversation(conv_id, chosen_pid, model)
    conversation = await db.get_chat_conversation(conv_id, uid)
    if conversation is not None:
        conversation.pop("messages", None)
    return JSONResponse({"conversation": conversation, "messages": [user_msg, reply]})
