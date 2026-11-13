"""Anthropic Messages pass-through: ``POST /v1/messages`` and
``POST /v1/messages/count_tokens``.

The request goes unchanged to a provider whose last probe found the route
(never translated to or from the OpenAI format): ``model`` becomes the
backend's name for it and ``max_tokens`` is clamped as for any request; the
rest of the body is left alone. The client's ``anthropic-version`` and
``anthropic-beta`` headers are forwarded; its key (``x-api-key`` or Bearer)
never is, the provider's credentials are sent instead. Replies, streamed or
not, are relayed as they come.

Errors the router raises on these routes have Anthropic's shape,
``{"type": "error", "error": {"type": ..., "message": ...}}``
(:class:`AnthropicErrorMiddleware`); upstream errors are sanitised as on
every route, then put in the same shape.
"""

from __future__ import annotations

import json
from collections.abc import MutableMapping
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import Response

from ..endpoints import COUNT_TOKENS_PATH, MESSAGES_PATH
from ..proxy import proxy_json

router = APIRouter()

FORWARDED_HEADERS = ("anthropic-version", "anthropic-beta")

# Status -> Anthropic error type.
_ERROR_TYPES = {
    400: "invalid_request_error",
    401: "authentication_error",
    403: "permission_error",
    404: "not_found_error",
    413: "request_too_large",
    429: "rate_limit_error",
    529: "overloaded_error",
}


def error_type(status: int) -> str:
    if status in _ERROR_TYPES:
        return _ERROR_TYPES[status]
    return "api_error" if status >= 500 else "invalid_request_error"


def anthropic_headers(request: Request) -> dict[str, str]:
    return {
        name: value
        for name in FORWARDED_HEADERS
        if (value := request.headers.get(name))
    }


async def _pass_through(request: Request, path: str, *, generation: bool) -> Response:
    headers = anthropic_headers(request)
    try:
        return await proxy_json(
            request,
            protocol="v1",
            endpoint=path,
            call=lambda client, body: client.messages(body, headers, path),
            stream_call=(
                (lambda client, body: client.messages_stream(body, headers))
                if generation
                else None
            ),
            generation=generation,
            media_type=lambda body: "text/event-stream",
        )
    except HTTPException as exc:
        if exc.status_code == 404 and str(exc.detail).startswith(
            "No available provider"
        ):
            model = str(exc.detail).split("'")[1] if "'" in str(exc.detail) else ""
            raise HTTPException(
                status_code=404,
                detail=(
                    f"No provider serves the Anthropic Messages API for model '{model}'"
                ),
            ) from exc
        raise


@router.post(MESSAGES_PATH)
async def messages(request: Request) -> Response:
    return await _pass_through(request, MESSAGES_PATH, generation=True)


@router.post(COUNT_TOKENS_PATH)
async def count_tokens(request: Request) -> Response:
    """Not generation: no decision policy, session affinity or token quota."""
    return await _pass_through(request, COUNT_TOKENS_PATH, generation=False)


def _message_of(payload: Any) -> str:
    if isinstance(payload, dict):
        detail = payload.get("detail")
        if isinstance(detail, str):
            return detail
        if detail is not None:
            return json.dumps(detail)
        err = payload.get("error")
        if isinstance(err, dict):
            return str(err.get("message") or err.get("type") or "error")
        if isinstance(err, str):
            return err
    return "error"


class AnthropicErrorMiddleware:
    """Puts error replies on the Messages routes in Anthropic's shape.

    Pure ASGI: only a JSON error reply (status 400 and up) is held and
    rewritten; anything else, streams included, passes straight through.
    """

    def __init__(self, app: Any) -> None:
        self.app = app

    async def __call__(
        self, scope: MutableMapping[str, Any], receive: Any, send: Any
    ) -> None:
        if scope["type"] != "http" or not str(scope.get("path", "")).startswith(
            MESSAGES_PATH
        ):
            await self.app(scope, receive, send)
            return
        start: dict[str, Any] | None = None
        body = b""

        async def wrap(message: Any) -> None:
            nonlocal start, body
            if message["type"] == "http.response.start":
                status = int(message["status"])
                ctype = dict(message.get("headers") or []).get(b"content-type", b"")
                if status >= 400 and b"json" in ctype:
                    start = message
                    return
                await send(message)
                return
            if start is None:
                await send(message)
                return
            body += message.get("body", b"")
            if message.get("more_body"):
                return
            status = int(start["status"])
            try:
                payload = json.loads(body)
            except ValueError:
                payload = None
            if isinstance(payload, dict) and payload.get("type") == "error":
                out = body
            else:
                out = json.dumps(
                    {
                        "type": "error",
                        "error": {
                            "type": error_type(status),
                            "message": _message_of(payload),
                        },
                    }
                ).encode()
            headers = [
                (k, v)
                for k, v in start.get("headers") or []
                if k.lower() != b"content-length"
            ]
            headers.append((b"content-length", str(len(out)).encode()))
            await send({**start, "headers": headers})
            await send({"type": "http.response.body", "body": out})

        await self.app(scope, receive, wrap)
