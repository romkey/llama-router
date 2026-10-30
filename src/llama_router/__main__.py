"""Entrypoint: runs the dashboard, Ollama API, OpenAI API, and cache servers."""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import logging
import signal
from collections.abc import Iterator

import uvicorn

from . import __version__, router_identity
from .api import deps as api_deps
from .api.app import app as api_app
from .auto_benchmark import AutoBenchmarker
from .config import settings
from .dashboard import deps as dash_deps
from .dashboard.app import app as dash_app
from .dashboard.oidc import CALLBACK_PATH, OIDCConfig
from .database import Database
from .llamacpp_api import deps as lcpp_deps
from .llamacpp_api.app import app as lcpp_app
from .provider_manager import ProviderManager
from .request_logger import attach_pricer, attach_stats
from .router import Router
from .sentry import init_sentry
from .shared_port import SharedPortApp
from .worker_link.worker import WorkerLinkManager

logger = logging.getLogger("llama_router")

# Requests a health checker polls: liveness on the dashboard port and the
# provider summary on every API port.
_HEALTH_PATHS = frozenset({"/health", "/healthz"})


class QuietHealthChecks(logging.Filter):
    """Drops uvicorn access-log lines for health checks that succeeded.

    uvicorn logs each request as ``%s - "%s %s HTTP/%s" %d`` with the client,
    method, path, HTTP version and status as arguments. A ``GET`` of a health
    path that answered 2xx is dropped; anything else, including a failing
    health check, is logged as usual.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        args = record.args
        if not isinstance(args, tuple) or len(args) != 5:
            return True
        _client, method, path, _version, status = args
        if method != "GET" or not isinstance(status, int):
            return True
        if str(path).partition("?")[0] not in _HEALTH_PATHS:
            return True
        return not 200 <= status < 300


class _Server(uvicorn.Server):
    """A uvicorn server that leaves signal handling to ``run``.

    Each uvicorn server installs its own SIGINT/SIGTERM handlers, so with
    several in one process only the last one sees the signal, and on exit it
    re-raises it with the default handler, killing the process before the
    database and links are closed.
    """

    @contextlib.contextmanager
    def capture_signals(self) -> Iterator[None]:
        yield


def _log_oidc_config() -> None:
    cfg = OIDCConfig.from_settings(settings)
    if not cfg.enabled:
        return
    problems = cfg.problems()
    if problems:
        for p in problems:
            logger.error("OIDC sign-in disabled: %s", p)
        if cfg.only:
            logger.error(
                "LLAMA_ROUTER_OIDC_ONLY is ignored while OIDC is misconfigured; "
                "password sign-in stays enabled"
            )
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
    # Before any provider is contacted: relayed requests carry the ID.
    rid = await router_identity.load(db)
    logger.info("Router ID %s", rid)

    pm = ProviderManager(db)
    await pm.start()

    worker_link = WorkerLinkManager(db)
    await worker_link.reload()

    rt = Router(db, pm)
    pm.attach_router(rt)
    attach_stats(pm.stats)
    attach_pricer(rt.price_for)
    # Build the first routing table before any API accepts requests.
    await rt.start()
    auto_bench = AutoBenchmarker(db, pm)
    await auto_bench.start()

    api_deps.init(db, pm, rt)
    dash_deps.init(db, pm)
    dash_deps.init_worker_link(worker_link)
    dash_deps.init_auto_benchmark(auto_bench)
    lcpp_deps.init(db, pm, rt)

    dashboard_asgi = (
        SharedPortApp(dash_app, api_app) if settings.api_on_dashboard_port else dash_app
    )
    dashboard_config = uvicorn.Config(
        dashboard_asgi,
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
        _Server(dashboard_config),
        _Server(api_config),
        _Server(lcpp_config),
    ]

    cache_blob_cache = None
    if settings.cache_enabled:
        from .registry_cache.app import app as cache_app
        from .registry_cache.app import init_cache
        from .registry_cache.cache import BlobCache

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
        servers.append(_Server(cache_config))
        logger.info(
            "Starting llama-router: dashboard on :%d, Ollama API on :%d, "
            "OpenAI API on :%d, registry cache on :%d",
            settings.dashboard_port,
            settings.api_port,
            settings.llamacpp_port,
            settings.cache_port,
        )
    else:
        logger.info(
            "Starting llama-router: dashboard on :%d, Ollama API on :%d, "
            "OpenAI API on :%d (cache disabled)",
            settings.dashboard_port,
            settings.api_port,
            settings.llamacpp_port,
        )

    if settings.api_on_dashboard_port:
        logger.info(
            "Ollama API, OpenAI API and worker links also served on dashboard port :%d",
            settings.dashboard_port,
        )

    if settings.quiet_health_checks:
        # After the uvicorn configs: each one reconfigures uvicorn's loggers.
        logging.getLogger("uvicorn.access").addFilter(QuietHealthChecks())

    def stop(sig: signal.Signals) -> None:
        logger.info("Received %s, shutting down", sig.name)
        for s in servers:
            s.should_exit = True

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, stop, sig)

    try:
        await asyncio.gather(*(s.serve() for s in servers))
    finally:
        await worker_link.stop()
        await auto_bench.stop()
        await rt.stop()
        await pm.worker_hub.close_all()
        await pm.stop()
        await db.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="llama-router",
        description="Route Ollama and OpenAI API requests across LLM backends.",
    )
    parser.add_argument(
        "--version", action="version", version=f"%(prog)s {__version__}"
    )
    parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    asyncio.run(run())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
