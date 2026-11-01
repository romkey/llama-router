"""Human-readable summaries for failed upstream httpx calls."""

from __future__ import annotations

import contextlib
import logging
from typing import Any

import httpx
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)


def describe_httpx_error(exc: BaseException) -> str:
    """Return a single-line summary: URL, HTTP status, and message when known."""
    if isinstance(exc, httpx.HTTPStatusError):
        req = exc.request
        url = str(req.url) if req is not None else "?"
        method = req.method if req is not None else "?"
        snippet = ""
        with contextlib.suppress(Exception):
            text = exc.response.text
            if text:
                snippet = f"; response_body={text[:200]!r}"
        return (
            f"{type(exc).__name__} {exc.response.status_code} "
            f"{exc.response.reason_phrase} for {method} {url}{snippet}"
        )
    if isinstance(exc, httpx.RequestError):
        req = exc.request
        url = str(req.url) if req is not None else "(no request URL)"
        return f"{type(exc).__name__} for {url}: {exc}"
    if isinstance(exc, httpx.HTTPError):
        return f"{type(exc).__name__}: {exc}"
    return f"{type(exc).__name__}: {exc}"


async def raise_for_stream_status(resp: httpx.Response) -> None:
    """``raise_for_status`` for a streamed response.

    Reads an error body first, while the stream is still open, so the
    ``HTTPStatusError`` can still be turned into the upstream's own reply.
    """
    if resp.is_error:
        await resp.aread()
    resp.raise_for_status()


# Upstream statuses whose error the client caused (a bad request, an unknown
# model, a rate limit) and may need to read. Only the error's type, code and
# message are passed on.
_DETAILED_STATUSES = frozenset({400, 404, 422, 429})
# The upstream refused this router's own credentials. That is not the
# client's fault, so the client sees a bad gateway instead.
_CREDENTIAL_STATUSES = frozenset({401, 403})
_MAX_MESSAGE_CHARS = 2000
_MAX_RETRY_AFTER_CHARS = 64
_GENERIC_ERROR = "upstream service returned an error"


def _scalar(value: Any) -> str | int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        return value[:_MAX_MESSAGE_CHARS]
    return None


def _safe_error(payload: Any) -> str | dict[str, str | int] | None:
    """The upstream's ``error`` with only its type, code and message kept.

    Ollama sends ``{"error": "text"}``; OpenAI and llama.cpp send
    ``{"error": {"type": ..., "code": ..., "message": ...}}``. The shape is
    kept so clients of either API can still read it. None when there is no
    usable error.
    """
    if not isinstance(payload, dict):
        return None
    error = payload.get("error")
    if isinstance(error, str):
        return error[:_MAX_MESSAGE_CHARS] or None
    if not isinstance(error, dict):
        return None
    kept = {
        key: value
        for key in ("type", "code", "message")
        if (value := _scalar(error.get(key))) is not None
    }
    return kept or None


def forward_upstream_http_error(exc: httpx.HTTPStatusError) -> JSONResponse:
    """Answer the client for an upstream error without leaking upstream detail.

    For 400, 404, 422 and 429 the upstream's error type, code and message are
    passed on, and nothing else. Other errors get a generic body. An upstream
    401 or 403 means this router's provider key was refused, so it becomes 502
    rather than look like a problem with the client's own key. Exception text
    is never echoed.
    """
    status_code = exc.response.status_code
    if status_code in _CREDENTIAL_STATUSES:
        logger.warning(
            "Upstream refused this router's credentials: %s",
            describe_httpx_error(exc),
        )
        return JSONResponse(status_code=502, content={"error": _GENERIC_ERROR})
    if status_code not in _DETAILED_STATUSES:
        return JSONResponse(status_code=status_code, content={"error": _GENERIC_ERROR})

    headers: dict[str, str] | None = None
    retry_after = exc.response.headers.get("retry-after")
    if (
        status_code == 429
        and retry_after
        and len(retry_after) <= _MAX_RETRY_AFTER_CHARS
    ):
        headers = {"Retry-After": retry_after}
    try:
        payload: Any = exc.response.json()
    except Exception:
        logger.warning("Upstream returned HTTP %s with non-JSON body", status_code)
        payload = None
    error = _safe_error(payload)
    return JSONResponse(
        status_code=status_code,
        content={"error": error if error is not None else _GENERIC_ERROR},
        headers=headers,
    )
