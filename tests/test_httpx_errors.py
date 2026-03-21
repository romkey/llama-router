"""Tests for httpx error string formatting."""

import httpx

from llama_router.httpx_errors import describe_httpx_error


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
