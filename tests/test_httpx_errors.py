"""Tests for httpx error string formatting and relaying upstream errors."""

import json
from typing import Any

import httpx
from fastapi.responses import JSONResponse

from llama_router.httpx_errors import (
    MAX_ERROR_BODY,
    describe_httpx_error,
    forward_upstream_http_error,
    raise_for_stream_status,
    read_capped,
)

GENERIC = "upstream service returned an error"


def test_describe_connect_error_includes_url() -> None:
    req = httpx.Request("GET", "http://10.0.0.1:11434/api/tags")
    exc = httpx.ConnectError("All connection attempts failed", request=req)
    text = describe_httpx_error(exc)
    assert "10.0.0.1:11434" in text
    assert "ConnectError" in text


def test_describe_http_status_error() -> None:
    req = httpx.Request("POST", "http://localhost:11434/api/chat")
    resp = httpx.Response(503, request=req, text="no backends")
    exc = httpx.HTTPStatusError("oops", request=req, response=resp)
    text = describe_httpx_error(exc)
    assert "503" in text
    assert "POST" in text
    assert "localhost:11434" in text


def _status_error(
    status: int, headers: dict[str, str] | None = None, **kwargs: Any
) -> httpx.HTTPStatusError:
    req = httpx.Request("POST", "http://upstream/v1/chat/completions")
    resp = httpx.Response(status, request=req, headers=headers, **kwargs)
    return httpx.HTTPStatusError("oops", request=req, response=resp)


def _body(resp: JSONResponse) -> object:
    return json.loads(bytes(resp.body))


def test_client_errors_keep_only_type_code_and_message() -> None:
    upstream = {
        "error": {
            "type": "invalid_request_error",
            "code": "model_not_found",
            "message": "no such model",
            "param": "model",
            "internal": {"host": "10.0.0.5", "trace": "..."},
        },
        "request_id": "abc",
    }
    resp = forward_upstream_http_error(_status_error(404, json=upstream))
    assert resp.status_code == 404
    assert _body(resp) == {
        "error": {
            "type": "invalid_request_error",
            "code": "model_not_found",
            "message": "no such model",
        }
    }


def test_ollama_style_string_error_is_kept() -> None:
    resp = forward_upstream_http_error(_status_error(400, json={"error": "bad"}))
    assert resp.status_code == 400
    assert _body(resp) == {"error": "bad"}


def test_long_messages_are_cut_and_odd_bodies_become_generic() -> None:
    long = forward_upstream_http_error(_status_error(422, json={"error": "x" * 5000}))
    error = _body(long)["error"]  # type: ignore[index]
    assert isinstance(error, str)
    assert len(error) == 2000
    odd: list[dict[str, Any]] = [
        {"text": "<html>oops</html>"},
        {"json": ["not", "an", "object"]},
        {"json": {"detail": "no error key"}},
        {"json": {"error": {"stack": ["frames"]}}},
    ]
    for kwargs in odd:
        resp = forward_upstream_http_error(_status_error(400, **kwargs))
        assert resp.status_code == 400
        assert _body(resp) == {"error": GENERIC}


def test_rate_limit_keeps_retry_after() -> None:
    resp = forward_upstream_http_error(
        _status_error(429, headers={"retry-after": "7"}, json={"error": "slow"})
    )
    assert resp.status_code == 429
    assert resp.headers["retry-after"] == "7"
    assert _body(resp) == {"error": "slow"}


def test_upstream_credential_errors_become_bad_gateway() -> None:
    for status in (401, 403):
        exc = _status_error(status, json={"error": "invalid api key sk-live-123"})
        resp = forward_upstream_http_error(exc)
        assert resp.status_code == 502
        assert _body(resp) == {"error": GENERIC}


def test_server_errors_get_a_generic_body() -> None:
    exc = _status_error(500, json={"error": "Traceback ... /srv/secret/path.py"})
    resp = forward_upstream_http_error(exc)
    assert resp.status_code == 500
    assert _body(resp) == {"error": GENERIC}


async def _streamed(
    status: int, chunks: list[bytes]
) -> tuple[httpx.Response, list[int]]:
    sent: list[int] = []

    async def body() -> Any:
        for chunk in chunks:
            sent.append(len(chunk))
            yield chunk

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            status, content=body(), headers={"retry-after": "3", "x-y": "z"}
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    req = client.build_request("POST", "http://up.test/v1/chat/completions")
    return await client.send(req, stream=True), sent


async def test_error_body_read_is_capped() -> None:
    chunk = b"x" * 16 * 1024
    resp, sent = await _streamed(500, [chunk] * 64)
    capped = await read_capped(resp)
    assert len(capped.content) == MAX_ERROR_BODY
    # Reading stopped at the cap instead of draining the whole megabyte.
    assert sum(sent) == MAX_ERROR_BODY
    assert capped.status_code == 500
    assert capped.headers["retry-after"] == "3"
    assert resp.is_closed


async def test_stream_status_error_keeps_a_capped_readable_body() -> None:
    resp, _ = await _streamed(429, [b'{"error": "slow down"}'])
    try:
        await raise_for_stream_status(resp)
    except httpx.HTTPStatusError as exc:
        forwarded = forward_upstream_http_error(exc)
    else:
        raise AssertionError("expected HTTPStatusError")
    assert forwarded.status_code == 429
    assert json.loads(bytes(forwarded.body)) == {"error": "slow down"}
    assert forwarded.headers["retry-after"] == "3"
