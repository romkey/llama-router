"""Entrypoint: runs the dashboard and Ollama API servers concurrently."""

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

    dashboard_server = uvicorn.Server(dashboard_config)
    api_server = uvicorn.Server(api_config)

    logger.info(
        "Starting llama-router: dashboard on :%d, API on :%d",
        settings.dashboard_port,
        settings.api_port,
    )

    try:
        await asyncio.gather(
            dashboard_server.serve(),
            api_server.serve(),
        )
    finally:
        await pm.stop()
        await db.close()


def main() -> None:
    asyncio.run(run())


if __name__ == "__main__":
    main()
