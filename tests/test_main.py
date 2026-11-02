from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any

import pytest
import uvicorn

from llama_router import __main__ as main_mod
from llama_router import __version__
from llama_router.__main__ import QuietHealthChecks, main
from llama_router.config import settings
from llama_router.database import Database


def test_version_prints_and_exits(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        main(["--version"])
    assert exc.value.code == 0
    assert capsys.readouterr().out.strip() == f"llama-router {__version__}"


def _access_record(method: str, path: str, status: int) -> logging.LogRecord:
    """An access-log record shaped like uvicorn's."""
    return logging.LogRecord(
        "uvicorn.access",
        logging.INFO,
        __file__,
        0,
        '%s - "%s %s HTTP/%s" %d',
        ("127.0.0.1:1234", method, path, "1.1", status),
        None,
    )


@pytest.mark.parametrize("path", ["/healthz", "/health", "/healthz?x=1"])
def test_quiet_health_checks_drops_successful_checks(path: str) -> None:
    assert QuietHealthChecks().filter(_access_record("GET", path, 200)) is False


@pytest.mark.parametrize(
    ("method", "path", "status"),
    [
        ("GET", "/healthz", 503),
        ("GET", "/health", 500),
        ("POST", "/health", 200),
        ("GET", "/healthzz", 200),
        ("GET", "/api/tags", 200),
    ],
)
def test_quiet_health_checks_keeps_everything_else(
    method: str, path: str, status: int
) -> None:
    assert QuietHealthChecks().filter(_access_record(method, path, status)) is True


def test_quiet_health_checks_keeps_records_of_another_shape() -> None:
    record = logging.LogRecord(
        "uvicorn.access", logging.INFO, __file__, 0, "started", None, None
    )
    assert QuietHealthChecks().filter(record) is True


def test_second_factor_reset_command_does_not_start_servers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from llama_router.dashboard import mfa_recovery

    usernames: list[str] = []

    async def reset(username: str) -> int:
        usernames.append(username)
        return 0

    monkeypatch.setattr(mfa_recovery, "reset_second_factor", reset)
    assert main(["--reset-dashboard-second-factor", "alice"]) == 0
    assert usernames == ["alice"]


# --- run() wiring, with the servers stubbed out ---


@pytest.fixture
def isolated_run(
    monkeypatch: pytest.MonkeyPatch, database_url: str, tmp_path: Path
) -> list[uvicorn.Config]:
    """Run ``run()`` against the test database; servers return at once.

    Every module-level object ``run`` sets is put back afterwards.
    """
    from llama_router import request_logger, router_identity
    from llama_router.api import deps as api_deps
    from llama_router.dashboard import deps as dash_deps
    from llama_router.llamacpp_api import deps as lcpp_deps
    from llama_router.registry_cache import app as cache_app

    for module, names in (
        (api_deps, ("_db", "_pm", "_router")),
        (lcpp_deps, ("_db", "_pm", "_router")),
        (
            dash_deps,
            ("_db", "_pm", "_blob_cache", "_worker_link", "_auto_benchmark"),
        ),
        (cache_app, ("_cache",)),
        (router_identity, ("_router_id",)),
        (request_logger, ("_stats", "_pricer")),
    ):
        for name in names:
            monkeypatch.setattr(module, name, getattr(module, name))
    access = logging.getLogger("uvicorn.access")
    monkeypatch.setattr(access, "filters", list(access.filters))

    monkeypatch.setattr(main_mod, "Database", lambda: Database(database_url))
    monkeypatch.setattr(settings, "cache_dir", str(tmp_path / "cache"))
    monkeypatch.setattr(settings, "admin_username", "")
    monkeypatch.setattr(settings, "oidc_issuer", "")
    served: list[uvicorn.Config] = []

    async def serve(self: uvicorn.Server, sockets: Any = None) -> None:
        served.append(self.config)

    monkeypatch.setattr(uvicorn.Server, "serve", serve)
    return served


@pytest.mark.parametrize("cache", [True, False])
@pytest.mark.parametrize("shared_port", [True, False])
@pytest.mark.parametrize("quiet", [True, False])
async def test_run_starts_one_server_per_port_and_shuts_down(
    isolated_run: list[uvicorn.Config],
    monkeypatch: pytest.MonkeyPatch,
    cache: bool,
    shared_port: bool,
    quiet: bool,
) -> None:
    from llama_router.dashboard import deps as dash_deps
    from llama_router.shared_port import SharedPortApp

    monkeypatch.setattr(settings, "cache_enabled", cache)
    monkeypatch.setattr(settings, "api_on_dashboard_port", shared_port)
    monkeypatch.setattr(settings, "quiet_health_checks", quiet)
    monkeypatch.setattr(settings, "dashboard_port", 18000)
    monkeypatch.setattr(settings, "api_port", 18001)
    monkeypatch.setattr(settings, "llamacpp_port", 18002)
    monkeypatch.setattr(settings, "cache_port", 18003)

    await asyncio.wait_for(main_mod.run(), 10)

    ports = [c.port for c in isolated_run]
    assert ports == [18000, 18001, 18002] + ([18003] if cache else [])
    assert isinstance(isolated_run[0].app, SharedPortApp) is shared_port
    assert (dash_deps.get_cache() is not None) is cache
    assert (
        any(
            isinstance(f, QuietHealthChecks)
            for f in logging.getLogger("uvicorn.access").filters
        )
        is quiet
    )
    # Shutdown ran: the database run() opened is closed.
    with pytest.raises(RuntimeError):
        await dash_deps.get_db()._select_one("SELECT 1", ())


async def test_run_closes_everything_when_a_server_fails(
    isolated_run: list[uvicorn.Config], monkeypatch: pytest.MonkeyPatch
) -> None:
    from llama_router.dashboard import deps as dash_deps

    monkeypatch.setattr(settings, "cache_enabled", False)

    async def broken(self: uvicorn.Server, sockets: Any = None) -> None:
        raise OSError("address already in use")

    monkeypatch.setattr(uvicorn.Server, "serve", broken)
    with pytest.raises(OSError, match="in use"):
        await asyncio.wait_for(main_mod.run(), 10)
    with pytest.raises(RuntimeError):
        await dash_deps.get_db()._select_one("SELECT 1", ())
