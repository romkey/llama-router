from __future__ import annotations

import logging

import pytest

from llama_router import __version__
from llama_router.__main__ import QuietHealthChecks, main


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
