"""Replies for routing refusals raised past the handlers.

A key with strict pins whose pinned provider can't serve the model gets
:class:`~.router.PinUnavailableError` from the router, from whichever
handler routed it (``proxy_json``, images, audio, decisions). Each API app
answers it the same way here: 503 with ``Retry-After``, and a row in the
request log naming the model, so the refusal shows up beside the requests
that were served.
"""

from __future__ import annotations

import contextlib
import logging
from collections.abc import Callable

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from .config import settings
from .database import Database
from .request_logger import log_request_background
from .router import PinUnavailableError

logger = logging.getLogger(__name__)


def retry_after_seconds() -> int:
    """When to try again: the next health check may bring the provider back."""
    return max(1, settings.health_check_interval_seconds)


def install_routing_error_handlers(
    app: FastAPI, get_db: Callable[[], Database], default_protocol: str
) -> None:
    async def pin_unavailable(request: Request, exc: Exception) -> JSONResponse:
        assert isinstance(exc, PinUnavailableError)  # noqa: S101  # registered for it
        logger.warning("Refusing request: %s", exc)
        detail = str(exc)
        with contextlib.suppress(Exception):
            log_request_background(
                get_db(),
                provider=None,
                protocol="v1"
                if request.url.path.startswith("/v1/")
                else default_protocol,
                endpoint=request.url.path,
                request=request,
                model=exc.model,
                request_body=None,
                response_size=0,
                duration_ms=0.0,
                status="error",
                error_detail=f"HTTP 503: {detail}"[:500],
                sample=False,
            )
        return JSONResponse(
            status_code=503,
            content={"detail": detail},
            headers={"Retry-After": str(retry_after_seconds())},
        )

    app.add_exception_handler(PinUnavailableError, pin_unavailable)
