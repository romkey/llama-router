"""An upstream that refuses a streamed request must yield a real HTTP error."""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from llama_router.httpx_errors import raise_for_stream_status
from llama_router.request_logger import stream_response


async def _upstream(handler: Any) -> Any:
    client = httpx.AsyncClient(
        transport=httpx.MockTransport(handler), base_url="http://upstream"
    )

    async def chunks() -> Any:
        async with client.stream("POST", "/v1/chat/completions") as resp:
            await raise_for_stream_status(resp)
            async for chunk in resp.aiter_bytes():
                yield chunk

    return chunks()


class _Logged:
    def __init__(self, inner: Any) -> None:
        self._inner = inner

    def __aiter__(self) -> Any:
        return self._inner.__aiter__()


@pytest.mark.asyncio
async def test_upstream_error_becomes_http_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, json={"error": "model not found"})

    resp = await stream_response(_Logged(await _upstream(handler)), "text/event-stream")  # type: ignore[arg-type]

    assert resp.status_code == 404
    assert json.loads(bytes(resp.body)) == {"error": "model not found"}


@pytest.mark.asyncio
async def test_successful_stream_is_passed_through() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"data: one\n\ndata: [DONE]\n\n")

    resp = await stream_response(_Logged(await _upstream(handler)), "text/event-stream")  # type: ignore[arg-type]

    assert resp.status_code == 200
    body = b"".join([c async for c in resp.body_iterator])  # type: ignore[attr-defined]
    assert body == b"data: one\n\ndata: [DONE]\n\n"


@pytest.mark.asyncio
async def test_inner_stream_closed_when_response_abandoned() -> None:
    closed = False

    async def inner() -> Any:
        nonlocal closed
        try:
            yield b"one"
            yield b"two"
        finally:
            closed = True

    resp = await stream_response(_Logged(inner()), "text/event-stream")  # type: ignore[arg-type]
    body = resp.body_iterator  # type: ignore[attr-defined]
    assert await body.__anext__() == b"one"
    await body.aclose()

    assert closed
