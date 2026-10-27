"""Entrypoint: runs the dashboard, Ollama API, llama.cpp API, and cache servers."""

from __future__ import annotations

import asyncio
import logging

import uvicorn

from .config import settings
from .dashboard.oidc import CALLBACK_PATH, OIDCConfig
from .database import Database
from .auto_benchmark import AutoBenchmarker
from .provider_manager import ProviderManager
from .request_logger import attach_stats
from .router import Router
from .sentry import init_sentry
from .worker_link.worker import WorkerLinkManager
from .api import deps as api_deps
from .api.app import app as api_app
from .dashboard import deps as dash_deps
from .dashboard.app import app as dash_app
from .llamacpp_api import deps as lcpp_deps
from .llamacpp_api.app import app as lcpp_app

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("llama_router")


def _log_oidc_config() -> None:
    cfg = OIDCConfig.from_settings(settings)
    if not cfg.enabled:
        return
    problems = cfg.problems()
    if problems:
        for p in problems:
            logger.error("OIDC sign-in disabled: %s", p)
        return
    callback = (
        cfg.redirect_base_url + CALLBACK_PATH
        if cfg.redirect_base_url
        else f"https://<dashboard host>{CALLBACK_PATH}"
    )
    logger.info(
        "OIDC sign-in enabled: issuer %s, callback %s, admin group %s%s",
        cfg.issuer,
        callback,
        repr(cfg.admin_group) if cfg.admin_group else "(none)",
        ", password sign-in disabled" if cfg.only else "",
    )


async def run() -> None:
    init_sentry()
    _log_oidc_config()

    db = Database()
    await db.connect()

    pm = ProviderManager(db)
    await pm.start()

    worker_link = WorkerLinkManager(db)
    await worker_link.reload()

    rt = Router(db, pm)
    pm.attach_router(rt)
    attach_stats(pm.stats)
    # Build the first routing table before any API accepts requests.
    await rt.start()
    auto_bench = AutoBenchmarker(db, pm)
    await auto_bench.start()

    api_deps.init(db, pm, rt)
    dash_deps.init(db, pm)
    dash_deps.init_worker_link(worker_link)
    dash_deps.init_auto_benchmark(auto_bench)
    lcpp_deps.init(db, pm, rt)

    dashboard_config = uvicorn.Config(
        dash_app,
        host=settings.dashboard_host,
        port=settings.dashboard_port,
        log_level="info",
    )
    api_config = uvicorn.Config(
        api_app,
        host=settings.api_host,
        port=settings.api_port,
        log_level="info",
    )
    lcpp_config = uvicorn.Config(
        lcpp_app,
        host=settings.llamacpp_host,
        port=settings.llamacpp_port,
        log_level="info",
    )

    servers = [
        uvicorn.Server(dashboard_config),
        uvicorn.Server(api_config),
        uvicorn.Server(lcpp_config),
    ]

    cache_blob_cache = None
    if settings.cache_enabled:
        from .registry_cache.cache import BlobCache
        from .registry_cache.app import app as cache_app, init_cache

        cache_blob_cache = BlobCache(
            settings.cache_dir, settings.cache_manifest_ttl_hours
        )
        init_cache(cache_blob_cache)
        dash_deps.init_cache(cache_blob_cache)

        cache_config = uvicorn.Config(
            cache_app,
            host=settings.cache_host,
            port=settings.cache_port,
            log_level="info",
        )
        servers.append(uvicorn.Server(cache_config))
        logger.info(
            "Starting llama-router: dashboard on :%d, Ollama API on :%d, "
            "llama.cpp API on :%d, registry cache on :%d",
            settings.dashboard_port,
            settings.api_port,
            settings.llamacpp_port,
            settings.cache_port,
        )
    else:
        logger.info(
            "Starting llama-router: dashboard on :%d, Ollama API on :%d, "
            "llama.cpp API on :%d (cache disabled)",
            settings.dashboard_port,
            settings.api_port,
            settings.llamacpp_port,
        )

    try:
        await asyncio.gather(*(s.serve() for s in servers))
    finally:
        await worker_link.stop()
        await auto_bench.stop()
        await rt.stop()
        await pm.worker_hub.close_all()
        await pm.stop()
        await db.close()


def main() -> None:
    asyncio.run(run())


if __name__ == "__main__":
    main()
