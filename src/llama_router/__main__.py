"""Entrypoint: runs the dashboard, Ollama API, llama.cpp API, and cache servers."""

from __future__ import annotations

import asyncio
import logging

import uvicorn

from .config import settings
from .database import Database
from .provider_manager import ProviderManager
from .router import Router
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


async def run() -> None:
    db = Database()
    await db.connect()

    pm = ProviderManager(db)
    await pm.start()

    rt = Router(db, pm)

    api_deps.init(db, pm, rt)
    dash_deps.init(db, pm)
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
        await pm.stop()
        await db.close()


def main() -> None:
    asyncio.run(run())


if __name__ == "__main__":
    main()
