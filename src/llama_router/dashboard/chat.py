"""Dashboard chat: talk to a model, for trying models out.

The model can be a served model, a model group or an alias. With a provider
chosen the request goes to it; without one the router picks, as it would for
an API request. Either way it goes through the provider's OpenAI-compatible
endpoints: ``/v1/chat/completions`` for chat, with an
attached image sent as an ``image_url`` part, and ``/v1/images/generations``
when asking for an image. Each dashboard user keeps their own history.
"""

from __future__ import annotations

import json
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

import anyio
import httpx
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import (
    HTMLResponse,
    JSONResponse,
    Response,
    StreamingResponse,
)

from ..config import settings
from ..database import Database
from ..httpx_errors import describe_httpx_error
from ..models import Provider, ProviderStatus
from ..request_logger import STATUS_CANCELLED, log_request_background
from ..router import Router, RoutingPreferences
from ..routing_table import RoutingTable, alias_groups
from ..v1_client import get_v1_client
from . import deps
from .my_keys import self_service_defaults
from .pages import _is_admin, dash_page_ctx, templates

router = APIRouter()

MODES = ("chat", "image")
MAX_CONTENT = 100_000
# A data URL; base64 makes it a third larger than the image.
MAX_IMAGE_CHARS = 14_000_000
TITLE_LEN = 60
PATHS = {"chat": "/v1/chat/completions", "image": "/v1/images/generations"}
NAME_KINDS = ("models", "groups", "aliases")


_GENERIC_ERROR = (
    "The model's provider returned an error. An administrator can find the "
    "details in the request log."
)


def _user_id(request: Request) -> int | None:
    user = getattr(request.state, "dashboard_user", None)
    return user.id if user is not None else None


@router.get("/chat", response_class=HTMLResponse)
async def chat_page(request: Request) -> Response:
    """Chat with a model on a provider."""
    return templates.TemplateResponse(
        request, "chat.html", await dash_page_ctx(request, {})
    )


async def _preferences(db: Database, request: Request) -> RoutingPreferences | None:
    """Routing limits for the signed-in user's chats: None (no limits) for an
    admin; otherwise the self-service defaults that keys users issue
    themselves get, so chat can't reach providers those keys can't.

    Chat requests are not tracked against a spend cap, so while the defaults
    set one a non-admin chats on local providers only (and not at all when
    the defaults allow cloud providers only).
    """
    if _is_admin(request):
        return None
    defaults = await self_service_defaults(db)
    locality: str | None = defaults["locality"]
    if defaults["daily_spend_cap_usd"] is not None:
        if locality == "cloud":
            raise HTTPException(
                status_code=403,
                detail="Chat is not available: your providers are cloud only "
                "and a daily spend cap applies",
            )
        locality = "local"
    return RoutingPreferences(locality=locality)


def _allowed(provider: Provider, prefs: RoutingPreferences | None) -> bool:
    return prefs is None or prefs.locality in (None, provider.locality)


async def _table() -> RoutingTable | None:
    rt = deps.get_pm().router
    return await rt.current_table() if rt is not None else None


@router.get("/api/chat/targets")
async def chat_targets(request: Request) -> JSONResponse:
    """Providers that are not offline (and that the user may chat with),
    each with the models it serves and the model groups and aliases that
    reach it, and under "any" every name the router can place on one of
    them."""
    db = deps.get_db()
    try:
        prefs = await _preferences(db, request)
    except HTTPException:
        return JSONResponse({"providers": [], "any": {k: [] for k in NAME_KINDS}})
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
        if not _allowed(p, prefs):
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


async def _target(
    db: Database,
    body: dict[str, Any],
    model: str,
    path: str,
    prefs: RoutingPreferences | None = None,
) -> Target:
    """Where *model* goes: the chosen provider, or the router's pick, within
    *prefs* (the user's limits; None for none)."""
    raw = body.get("provider_id")
    if raw in (None, ""):
        rt = deps.get_pm().router
        result = await rt.route(model, path, prefs) if rt is not None else None
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
    if not _allowed(provider, prefs):
        raise HTTPException(
            status_code=403,
            detail=f"{provider.name} is a {provider.locality} provider, "
            "which you may not use",
        )
    table = await _table()
    # A group hides a model of the same name, as it does when routing.
    if table is None or not table.is_group(model):
        for m in await db.get_provider_models(pid):
            if m.name == model:
                return Target(provider, m.backend_name, model, True)
    # A model group or an alias with a member on this provider.
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


@dataclass(slots=True)
class Turn:
    """A user message saved and ready to go to the model."""

    mode: str
    uid: int | None
    conv_id: int
    provider: Provider
    # The provider the user chose, kept with the chat; None when the router
    # picked.
    chosen_pid: int | None
    answered_by: str
    user_msg: dict[str, Any]
    upstream: dict[str, Any]
    log: dict[str, Any]
    admin: bool


async def _start_turn(request: Request, body: dict[str, Any]) -> Turn:
    """Check *body*, find the target, open or start the conversation and
    save the user's message."""
    db = deps.get_db()
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
    target = await _target(
        db, body, model, PATHS[mode], await _preferences(db, request)
    )
    provider, backend_model = target.provider, target.backend_model
    pid = provider.require_id()
    # Reopening the chat brings back the choice made, not the router's pick.
    chosen_pid = pid if target.chosen else None
    answered_by = model if target.served == model else f"{model} → {target.served}"

    conv_id = body.get("conversation_id")
    if conv_id in (None, ""):
        limit = settings.max_chat_conversations_per_user
        if limit > 0 and len(await db.list_chat_conversations(uid)) >= limit:
            raise HTTPException(
                status_code=409,
                detail=f"You have {limit} conversations; delete one to start another",
            )
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
    return Turn(
        mode,
        uid,
        conv_id,
        provider,
        chosen_pid,
        answered_by,
        user_msg,
        upstream,
        log,
        _is_admin(request),
    )


def _error_detail(exc: Exception) -> str:
    if isinstance(exc, httpx.HTTPError):
        return describe_httpx_error(exc)
    return str(exc) or type(exc).__name__


async def _add_reply(
    turn: Turn,
    role: str,
    text: str,
    duration: float,
    image: str | None = None,
) -> dict[str, Any]:
    return await deps.get_db().add_chat_message(
        turn.conv_id,
        role,
        text,
        image=image,
        model_name=turn.answered_by,
        provider_name=turn.provider.name,
        duration_ms=duration,
    )


async def _finish_turn(turn: Turn) -> dict[str, Any] | None:
    """Touch the conversation; returns it without its messages."""
    db = deps.get_db()
    model = turn.log["model"]
    await db.touch_chat_conversation(turn.conv_id, turn.chosen_pid, model)
    conversation = await db.get_chat_conversation(turn.conv_id, turn.uid)
    if conversation is not None:
        conversation.pop("messages", None)
    return conversation


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
    body = await deps.json_body(request)
    turn = await _start_turn(request, body)
    db = deps.get_db()
    pm = deps.get_pm()
    pid = turn.provider.require_id()
    start = time.monotonic()
    try:
        async with pm.acquire_provider(pid):
            client = get_v1_client(pm, pid)
            if turn.mode == "image":
                resp = await client.images_generations(turn.upstream)
            else:
                resp = await client.chat_completions(turn.upstream)
    except Exception as exc:
        duration = (time.monotonic() - start) * 1000
        detail = _error_detail(exc)
        log_request_background(
            db, duration_ms=duration, status="error", error_detail=detail, **turn.log
        )
        # Upstream errors name internal URLs; only admins see them in chat.
        reply = await _add_reply(
            turn, "error", detail if turn.admin else _GENERIC_ERROR, duration
        )
    else:
        duration = (time.monotonic() - start) * 1000
        log_request_background(db, duration_ms=duration, response_body=resp, **turn.log)
        if turn.mode == "image":
            out_image, text = reply_image(resp)
            if out_image is None:
                text = text or "The backend returned no image."
        else:
            out_image, text = None, reply_text(resp)
        reply = await _add_reply(
            turn,
            "assistant" if turn.mode == "chat" or out_image else "error",
            text,
            duration,
            image=out_image,
        )
    conversation = await _finish_turn(turn)
    return JSONResponse(
        {"conversation": conversation, "messages": [turn.user_msg, reply]}
    )


class StreamDeltas:
    """Reply text from an OpenAI-style SSE stream of chat completion chunks.

    Chunks need not end on a line boundary. Reasoning text is kept apart:
    it stands in for the reply only when the model gave no other answer.
    """

    def __init__(self) -> None:
        self._buf = b""
        self.reasoning: list[str] = []

    def feed(self, chunk: bytes) -> list[str]:
        self._buf += chunk
        out: list[str] = []
        while b"\n" in self._buf:
            raw, self._buf = self._buf.split(b"\n", 1)
            line = raw.strip()
            if not line.startswith(b"data:"):
                continue
            data = line[5:].strip()
            if data == b"[DONE]":
                continue
            try:
                event = json.loads(data)
            except ValueError:
                continue
            if not isinstance(event, dict):
                continue
            for choice in event.get("choices") or []:
                if not isinstance(choice, dict):
                    continue
                delta = choice.get("delta") or {}
                text = delta.get("content")
                if isinstance(text, str) and text:
                    out.append(text)
                why = delta.get("reasoning_content") or delta.get("reasoning")
                if isinstance(why, str) and why:
                    self.reasoning.append(why)
        return out


def _event(obj: dict[str, Any]) -> bytes:
    return (json.dumps(obj) + "\n").encode()


STOPPED_EMPTY = "Stopped before the model answered."


async def _relay(request: Request, turn: Turn) -> AsyncIterator[bytes]:
    """Stream the reply as JSON lines: a "start" event with the saved user
    message, "delta" events with reply text, and a "done" event with the
    saved reply. When the browser hangs up (Stop), what arrived so far is
    saved as the reply.

    A hang-up is checked for between upstream chunks: behind the dashboard's
    middleware a failed send may not show it for several seconds.
    """
    db = deps.get_db()
    pm = deps.get_pm()
    pid = turn.provider.require_id()
    yield _event(
        {"type": "start", "conversation_id": turn.conv_id, "user": turn.user_msg}
    )
    parts: list[str] = []
    deltas = StreamDeltas()
    error: str | None = None
    finished = stopped = False
    reply: dict[str, Any] | None = None
    start = time.monotonic()
    try:
        async with pm.acquire_provider(pid):
            client = get_v1_client(pm, pid)
            async for chunk in client.chat_completions_stream(
                {**turn.upstream, "stream": True}
            ):
                if await request.is_disconnected():
                    stopped = True
                    break
                for text in deltas.feed(chunk):
                    parts.append(text)
                    yield _event({"type": "delta", "text": text})
        finished = not stopped
    except Exception as exc:
        error = _error_detail(exc)
    finally:
        # Also runs when the browser hung up and this task is being
        # cancelled; the shield lets the reply be saved anyway.
        with anyio.CancelScope(shield=True):
            duration = (time.monotonic() - start) * 1000
            text = "".join(parts)
            if not text.strip() and finished:
                text = "".join(deltas.reasoning)
            if error is not None:
                status = "error"
            elif finished:
                status = "ok"
            else:
                status = STATUS_CANCELLED
            log_request_background(
                db,
                duration_ms=duration,
                streamed=True,
                status=status,
                error_detail=error,
                response_body={
                    "choices": [{"message": {"role": "assistant", "content": text}}]
                },
                **turn.log,
            )
            if error is not None:
                # Upstream errors name internal URLs; only admins see them.
                shown = error if turn.admin else _GENERIC_ERROR
                reply = await _add_reply(turn, "error", shown, duration)
            elif text or finished:
                reply = await _add_reply(turn, "assistant", text, duration)
            else:
                reply = await _add_reply(turn, "error", STOPPED_EMPTY, duration)
            conversation = await _finish_turn(turn)
    yield _event({"type": "done", "conversation": conversation, "message": reply})


@router.post("/api/chat/stream")
async def stream(request: Request) -> StreamingResponse:
    """Like ``/api/chat/send`` for chat (not image) mode, but the reply
    comes back as it is written: newline-delimited JSON events (see
    ``_relay``). Request errors are answered before streaming, as there."""
    body = await deps.json_body(request)
    if str(body.get("mode") or "chat") != "chat":
        raise HTTPException(status_code=400, detail="only chat mode streams")
    turn = await _start_turn(request, body)
    return StreamingResponse(
        _relay(request, turn),
        media_type="application/x-ndjson",
        headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"},
    )
