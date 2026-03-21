"""Human-readable summaries for failed upstream httpx calls."""

from __future__ import annotations

import httpx


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
