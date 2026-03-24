"""Benchmark chat→embed fallback behavior (transient errors, 501 embed)."""

from __future__ import annotations

import httpx
import pytest

from llama_router.provider_manager import _transient_chat_benchmark_failure


def _status_error(status: int) -> httpx.HTTPStatusError:
    req = httpx.Request("POST", "http://example/api/chat")
    resp = httpx.Response(status, request=req)
    return httpx.HTTPStatusError("err", request=req, response=resp)


@pytest.mark.parametrize(
    "exc,transient",
    [
        (httpx.ReadTimeout("t"), True),
        (httpx.ConnectError("c"), True),
        (_status_error(503), True),
        (_status_error(500), True),
        (_status_error(502), True),
        (_status_error(504), True),
        (_status_error(429), True),
        (_status_error(408), True),
        (_status_error(400), False),
        (_status_error(404), False),
        (_status_error(501), False),
        (ValueError("x"), False),
    ],
)
def test_transient_chat_benchmark_failure(exc: BaseException, transient: bool) -> None:
    assert _transient_chat_benchmark_failure(exc) is transient
