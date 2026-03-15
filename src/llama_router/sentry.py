from __future__ import annotations

import logging

import sentry_sdk
from sentry_sdk.integrations.fastapi import FastApiIntegration

from . import __version__
from .config import settings

logger = logging.getLogger(__name__)


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
        integrations=[FastApiIntegration()],
    )
    logger.info(
        "Sentry enabled (env=%s, traces=%.3f, profiles=%.3f)",
        settings.sentry_environment,
        settings.sentry_traces_sample_rate,
        settings.sentry_profiles_sample_rate,
    )
