"""Serve the Ollama API from an OpenAI-style provider.

Ollama clients send ``/api/chat``, ``/api/generate``, ``/api/embed`` and
``/api/embeddings``. When the router picks a provider that speaks the
OpenAI-style API (llama.cpp, vLLM, a cloud service), :class:`OllamaOverOpenAI`
stands in for an Ollama client: it turns the request into
``/v1/chat/completions`` or ``/v1/embeddings`` and the answer back into
Ollama's shape, streamed (NDJSON) or not.

What carries over: messages, images (as data URLs), tools and tool calls,
``format`` (JSON mode or a JSON schema), the common sampling options, and
token counts. What does not: ``raw``, ``template``, ``context``, ``suffix``
(infill), ``keep_alive`` and the Ollama-only options such as ``top_k`` and
``num_ctx``, which strict OpenAI services reject.
"""

from __future__ import annotations

import json
import time
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from typing import Any

# Ollama option -> OpenAI parameter.
_OPTIONS = {
    "temperature": "temperature",
    "top_p": "top_p",
    "seed": "seed",
    "stop": "stop",
    "num_predict": "max_tokens",
    "presence_penalty": "presence_penalty",
    "frequency_penalty": "frequency_penalty",
}

_FINISH = {"stop": "stop", "length": "length", "tool_calls": "stop"}


class TranslationError(ValueError):
    """The Ollama request uses something with no OpenAI-style equivalent."""


def _now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _image_url(data: str) -> str:
    if data.startswith("data:"):
        return data
    mime = "image/png"
    if data.startswith("/9j/"):
        mime = "image/jpeg"
    elif data.startswith("R0lGOD"):
        mime = "image/gif"
    elif data.startswith("UklGR"):
        mime = "image/webp"
    return f"data:{mime};base64,{data}"


def _content(text: Any, images: Any) -> Any:
    if not images:
        return "" if text is None else text
    parts: list[dict[str, Any]] = []
    if text:
        parts.append({"type": "text", "text": text})
    for img in images:
        parts.append({"type": "image_url", "image_url": {"url": _image_url(str(img))}})
    return parts


def _sampling(body: dict[str, Any], out: dict[str, Any]) -> None:
    options = body.get("options") or {}
    if isinstance(options, dict):
        for src, dst in _OPTIONS.items():
            if options.get(src) is not None:
                out[dst] = options[src]
    fmt = body.get("format")
    if fmt == "json":
        out["response_format"] = {"type": "json_object"}
    elif isinstance(fmt, dict):
        out["response_format"] = {
            "type": "json_schema",
            "json_schema": {"name": "response", "schema": fmt},
        }


def chat_request(body: dict[str, Any]) -> dict[str, Any]:
    """An Ollama /api/chat body as a /v1/chat/completions body."""
    messages: list[dict[str, Any]] = []
    pending_ids: list[str] = []
    counter = 0
    for m in body.get("messages") or []:
        if not isinstance(m, dict):
            continue
        role = m.get("role") or "user"
        msg: dict[str, Any] = {
            "role": role,
            "content": _content(m.get("content"), m.get("images")),
        }
        if role == "assistant" and m.get("tool_calls"):
            calls = []
            pending_ids = []
            for call in m["tool_calls"]:
                fn = (call or {}).get("function") or {}
                counter += 1
                call_id = str(call.get("id") or f"call_{counter}")
                pending_ids.append(call_id)
                args = fn.get("arguments", {})
                calls.append(
                    {
                        "id": call_id,
                        "type": "function",
                        "function": {
                            "name": fn.get("name", ""),
                            "arguments": (
                                args if isinstance(args, str) else json.dumps(args)
                            ),
                        },
                    }
                )
            msg["tool_calls"] = calls
            if not msg["content"]:
                msg["content"] = None
        elif role == "tool":
            # OpenAI ties a tool result to its call; Ollama sends them in order.
            msg["tool_call_id"] = m.get("tool_call_id") or (
                pending_ids.pop(0) if pending_ids else f"call_{counter}"
            )
        messages.append(msg)
    out: dict[str, Any] = {"model": body.get("model"), "messages": messages}
    if body.get("tools"):
        out["tools"] = body["tools"]
    _sampling(body, out)
    return out


def generate_request(body: dict[str, Any]) -> dict[str, Any]:
    """An Ollama /api/generate body as a /v1/chat/completions body."""
    if body.get("suffix"):
        raise TranslationError(
            "suffix (fill-in-the-middle) is not available on this provider"
        )
    if body.get("raw"):
        raise TranslationError("raw prompts are not available on this provider")
    messages: list[dict[str, Any]] = []
    if body.get("system"):
        messages.append({"role": "system", "content": body["system"]})
    messages.append(
        {"role": "user", "content": _content(body.get("prompt"), body.get("images"))}
    )
    out: dict[str, Any] = {"model": body.get("model"), "messages": messages}
    _sampling(body, out)
    return out


def _tool_calls(calls: Any) -> list[dict[str, Any]]:
    out = []
    for call in calls or []:
        fn = (call or {}).get("function") or {}
        args = fn.get("arguments")
        if isinstance(args, str):
            try:
                args = json.loads(args) if args.strip() else {}
            except ValueError:
                args = {"_raw": args}
        out.append({"function": {"name": fn.get("name", ""), "arguments": args or {}}})
    return out


def _counts(usage: Any, started: float) -> dict[str, Any]:
    out: dict[str, Any] = {
        "total_duration": int((time.monotonic() - started) * 1e9),
        "load_duration": 0,
    }
    if isinstance(usage, dict):
        if usage.get("prompt_tokens") is not None:
            out["prompt_eval_count"] = usage["prompt_tokens"]
        if usage.get("completion_tokens") is not None:
            out["eval_count"] = usage["completion_tokens"]
    return out


def chat_response(resp: dict[str, Any], model: str, started: float) -> dict[str, Any]:
    choice = (resp.get("choices") or [{}])[0]
    message = choice.get("message") or {}
    out_msg: dict[str, Any] = {
        "role": "assistant",
        "content": message.get("content") or "",
    }
    if message.get("tool_calls"):
        out_msg["tool_calls"] = _tool_calls(message["tool_calls"])
    return {
        "model": model,
        "created_at": _now(),
        "message": out_msg,
        "done": True,
        "done_reason": _FINISH.get(choice.get("finish_reason") or "stop", "stop"),
        **_counts(resp.get("usage"), started),
    }


def generate_response(
    resp: dict[str, Any], model: str, started: float
) -> dict[str, Any]:
    chat = chat_response(resp, model, started)
    out = {k: v for k, v in chat.items() if k != "message"}
    out["response"] = chat["message"]["content"]
    return out


class _StreamTranslator:
    """OpenAI SSE bytes in, Ollama NDJSON bytes out."""

    def __init__(self, model: str, mode: str, started: float) -> None:
        self._model = model
        self._mode = mode  # "chat" or "generate"
        self._started = started
        self._buf = b""
        self._tools: dict[int, dict[str, Any]] = {}
        self._finish: str | None = None
        self._usage: Any = None
        self.done = False

    def _line(self, text: str, extra: dict[str, Any] | None = None) -> bytes:
        obj: dict[str, Any] = {"model": self._model, "created_at": _now()}
        if self._mode == "chat":
            obj["message"] = {"role": "assistant", "content": text}
        else:
            obj["response"] = text
        obj["done"] = False
        if extra:
            obj.update(extra)
        return (json.dumps(obj) + "\n").encode()

    def feed(self, chunk: bytes) -> list[bytes]:
        self._buf += chunk
        out: list[bytes] = []
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
            if event.get("usage"):
                self._usage = event["usage"]
            for choice in event.get("choices") or []:
                delta = choice.get("delta") or {}
                for tc in delta.get("tool_calls") or []:
                    slot = self._tools.setdefault(
                        int(tc.get("index", 0)),
                        {"function": {"name": "", "arguments": ""}},
                    )
                    fn = tc.get("function") or {}
                    slot["function"]["name"] += fn.get("name") or ""
                    slot["function"]["arguments"] += fn.get("arguments") or ""
                text = delta.get("content")
                if text:
                    out.append(self._line(text))
                if choice.get("finish_reason"):
                    self._finish = choice["finish_reason"]
        return out

    def finish(self) -> list[bytes]:
        out: list[bytes] = []
        if self._tools and self._mode == "chat":
            calls = _tool_calls([self._tools[i] for i in sorted(self._tools)])
            out.append(
                self._line(
                    "",
                    {
                        "message": {
                            "role": "assistant",
                            "content": "",
                            "tool_calls": calls,
                        }
                    },
                )
            )
        final = self._line(
            "",
            {
                "done": True,
                "done_reason": _FINISH.get(self._finish or "stop", "stop"),
                **_counts(self._usage, self._started),
            },
        )
        out.append(final)
        self.done = True
        return out


def embed_response(resp: dict[str, Any], model: str) -> dict[str, Any]:
    vectors = [d.get("embedding") for d in resp.get("data") or []]
    out: dict[str, Any] = {"model": model, "embeddings": vectors}
    usage = resp.get("usage") or {}
    if usage.get("prompt_tokens") is not None:
        out["prompt_eval_count"] = usage["prompt_tokens"]
    return out


class OllamaOverOpenAI:
    """The Ollama client methods the API handlers call, served by an
    OpenAI-style client."""

    def __init__(self, client: Any) -> None:
        self._client = client

    @staticmethod
    def validate(kind: str, body: dict[str, Any]) -> None:
        """Raise TranslationError before a response starts, not mid-stream."""
        if kind == "generate":
            generate_request(body)
        elif kind == "chat":
            chat_request(body)

    async def chat(self, body: dict[str, Any]) -> dict[str, Any]:
        started = time.monotonic()
        resp = await self._client.chat_completions(chat_request(body))
        return chat_response(resp, body.get("model") or "", started)

    async def chat_stream(self, body: dict[str, Any]) -> AsyncIterator[bytes]:
        async for line in self._stream(chat_request(body), body, "chat"):
            yield line

    async def generate(self, body: dict[str, Any]) -> dict[str, Any]:
        started = time.monotonic()
        if not body.get("prompt") and not body.get("images"):
            return _load_only(body)
        resp = await self._client.chat_completions(generate_request(body))
        return generate_response(resp, body.get("model") or "", started)

    async def generate_stream(self, body: dict[str, Any]) -> AsyncIterator[bytes]:
        if not body.get("prompt") and not body.get("images"):
            yield (json.dumps(_load_only(body)) + "\n").encode()
            return
        async for line in self._stream(generate_request(body), body, "generate"):
            yield line

    async def _stream(
        self, req: dict[str, Any], body: dict[str, Any], mode: str
    ) -> AsyncIterator[bytes]:
        started = time.monotonic()
        # Ask for token counts in the last event; Ollama's final line has them.
        req["stream_options"] = {"include_usage": True}
        tr = _StreamTranslator(body.get("model") or "", mode, started)
        async for chunk in self._client.chat_completions_stream(req):
            for line in tr.feed(chunk):
                yield line
        for line in tr.feed(b"\n"):
            yield line
        for line in tr.finish():
            yield line

    async def embed(self, body: dict[str, Any]) -> dict[str, Any]:
        inp = body.get("input")
        req: dict[str, Any] = {"model": body.get("model"), "input": inp}
        if body.get("dimensions"):
            req["dimensions"] = body["dimensions"]
        resp = await self._client.embeddings(req)
        return embed_response(resp, body.get("model") or "")

    async def embeddings(self, body: dict[str, Any]) -> dict[str, Any]:
        resp = await self._client.embeddings(
            {"model": body.get("model"), "input": body.get("prompt") or ""}
        )
        data = resp.get("data") or [{}]
        return {"embedding": data[0].get("embedding") or []}


def _load_only(body: dict[str, Any]) -> dict[str, Any]:
    """Ollama loads a model on an empty generate; a remote API has nothing to
    load, so answer at once."""
    return {
        "model": body.get("model") or "",
        "created_at": _now(),
        "response": "",
        "done": True,
        "done_reason": "load",
    }
