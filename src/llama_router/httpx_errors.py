"""Human-readable summaries for failed upstream httpx calls."""

from __future__ import annotations

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
        try:
            text = exc.response.text
            if text:
                snippet = f"; response_body={text[:200]!r}"
        except Exception:
            pass
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


def forward_upstream_http_error(exc: httpx.HTTPStatusError) -> JSONResponse:
    """Return the upstream JSON body when safe; never echo exception text to clients."""
    status_code = exc.response.status_code
    try:
        payload: Any = exc.response.json()
    except ValueError:
        logger.warning("Upstream returned HTTP %s with non-JSON body", status_code)
        return JSONResponse(
            status_code=status_code,
            content={"error": "upstream service returned an error"},
        )
    except Exception:
        logger.warning("Upstream returned HTTP %s; failed to read body", status_code)
        return JSONResponse(
            status_code=status_code,
            content={"error": "upstream service returned an error"},
        )

    if isinstance(payload, dict):
        return JSONResponse(status_code=status_code, content=payload)
    return JSONResponse(
        status_code=status_code,
        content={"error": "upstream service returned an error"},
    )
