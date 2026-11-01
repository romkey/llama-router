from __future__ import annotations

import logging
import re
from typing import Any

import sentry_sdk
from sentry_sdk.integrations.fastapi import FastApiIntegration

from . import __version__
from .config import settings

logger = logging.getLogger(__name__)

# Values under keys like these are replaced wherever they appear in an event.
_SENSITIVE_KEY = re.compile(r"api[-_]?key|authorization|token|secret|password", re.I)
# Request parts that may carry prompts, keys or session cookies.
_DROPPED_REQUEST_FIELDS = ("data", "headers", "cookies", "query_string", "env")
FILTERED = "[Filtered]"


def _scrub(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            k: (
                FILTERED
                if isinstance(k, str) and _SENSITIVE_KEY.search(k)
                else _scrub(v)
            )
            for k, v in value.items()
        }
    if isinstance(value, list | tuple):
        return [_scrub(v) for v in value]
    return value


def scrub_event(event: Any, hint: Any = None) -> Any:
    """``before_send``: drop request bodies, headers and cookies; mask secrets.

    Upstream keys, client API keys and prompts must never reach Sentry.
    """
    request = event.get("request")
    if isinstance(request, dict):
        for field in _DROPPED_REQUEST_FIELDS:
            request.pop(field, None)
    return _scrub(event)


def init_sentry() -> None:
    dsn = (settings.sentry_dsn or "").strip()
    if not dsn:
        logger.info("Sentry disabled (LLAMA_ROUTER_SENTRY_DSN not set)")
        return

    sentry_sdk.init(
        dsn=dsn,
        environment=settings.sentry_environment,
        release=f"llama-router@{__version__}",
        traces_sample_rate=settings.sentry_traces_sample_rate,
        profiles_sample_rate=settings.sentry_profiles_sample_rate,
        send_default_pii=settings.sentry_send_default_pii,
        # Frame locals can hold API keys and prompts.
        include_local_variables=False,
        before_send=scrub_event,
        before_send_transaction=scrub_event,
        integrations=[FastApiIntegration()],
    )
    logger.info(
        "Sentry enabled (env=%s, traces=%.3f, profiles=%.3f)",
        settings.sentry_environment,
        settings.sentry_traces_sample_rate,
        settings.sentry_profiles_sample_rate,
    )
