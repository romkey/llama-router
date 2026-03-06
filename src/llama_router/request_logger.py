"""Helpers for logging dispatched API requests to the database."""

from __future__ import annotations

import json
import time
from collections.abc import AsyncIterator
from typing import Any

from fastapi import Request

from .database import Database
from .models import Provider, RequestLog


def _client_ip(request: Request) -> str:
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def _payload_size(body: Any) -> int:
    """Approximate size of the JSON request body in bytes."""
    try:
        return len(json.dumps(body).encode())
    except Exception:
        return 0


async def log_request(
    db: Database,
    *,
    provider: Provider | None,
    protocol: str,
    endpoint: str,
    request: Request | None = None,
    source_ip: str | None = None,
    model: str | None,
    request_body: Any = None,
    response_size: int = 0,
    duration_ms: float,
    streamed: bool = False,
    status: str = "ok",
    error_detail: str | None = None,
) -> None:
    ip = source_ip or (_client_ip(request) if request else "internal")
    entry = RequestLog(
        provider_id=provider.id if provider else None,
        provider_name=provider.name if provider else None,
        protocol=protocol,
        endpoint=endpoint,
        source_ip=ip,
        model=model,
        request_size=_payload_size(request_body),
        response_size=response_size,
        duration_ms=duration_ms,
        status=status,
        streamed=streamed,
        error_detail=error_detail,
    )
    await db.save_request_log(entry)


class StreamLogger:
    """Wraps an async byte iterator to measure total response size and duration."""

    def __init__(
        self,
        inner: AsyncIterator[bytes],
        *,
        db: Database,
        provider: Provider,
        protocol: str,
        endpoint: str,
        request: Request,
        model: str | None,
        request_body: Any,
        start_time: float,
    ):
        self._inner = inner
        self._db = db
        self._provider = provider
        self._protocol = protocol
        self._endpoint = endpoint
        self._request = request
        self._model = model
        self._request_body = request_body
        self._start = start_time
        self._total_bytes = 0

    async def __aiter__(self):
        status = "ok"
        error_detail = None
        try:
            async for chunk in self._inner:
                self._total_bytes += len(chunk)
                yield chunk
        except Exception as exc:
            status = "error"
            error_detail = str(exc)[:500]
            raise
        finally:
            duration = (time.monotonic() - self._start) * 1000
            try:
                await log_request(
                    self._db,
                    provider=self._provider,
                    protocol=self._protocol,
                    endpoint=self._endpoint,
                    request=self._request,
                    model=self._model,
                    request_body=self._request_body,
                    response_size=self._total_bytes,
                    duration_ms=duration,
                    streamed=True,
                    status=status,
                    error_detail=error_detail,
                )
            except Exception:
                pass
