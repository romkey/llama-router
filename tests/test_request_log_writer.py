"""SQLite settings and the batched, bounded request log writer."""

from __future__ import annotations

import asyncio
import socket
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest
import sqlalchemy as sa
import uvicorn
from starlette.requests import Request

from llama_router import background, metrics, request_logger
from llama_router.__main__ import ServerStartupError, _Server, serve_all
from llama_router.dashboard import routes as dash_routes
from llama_router.database import Database
from llama_router.models import RequestLog


def _log(db: Database, **kwargs: Any) -> None:
    request_logger.log_request_background(
        db,
        provider=None,
        protocol="ollama",
        endpoint="/api/chat",
        model="m",
        duration_ms=1.0,
        **kwargs,
    )


async def test_sqlite_connections_use_wal(tmp_path: Path) -> None:
    db = Database(str(tmp_path / "wal.db"))
    await db.connect()
    try:
        row = await db._select_one("PRAGMA journal_mode")
        assert row is not None
        assert str(next(iter(row.values()))).lower() == "wal"
        row = await db._select_one("PRAGMA busy_timeout")
        assert row is not None
        assert next(iter(row.values())) == 5000
        assert db._engine is not None
        assert db._engine.pool._pre_ping is False
    finally:
        await db.close()


async def test_request_log_model_index(tmp_path: Path) -> None:
    path = tmp_path / "ix.db"
    db = Database(str(path))
    await db.connect()
    await db.close()
    engine = sa.create_engine(f"sqlite:///{path}")
    try:
        indexes = {
            ix["name"]: ix["column_names"]
            for ix in sa.inspect(engine).get_indexes("request_log")
        }
    finally:
        engine.dispose()
    assert indexes["idx_request_log_model_created"] == ["model", "created_at"]


async def test_concurrent_logs_are_batched_without_tasks(
    db: Database, monkeypatch: pytest.MonkeyPatch
) -> None:
    batches: list[int] = []
    save = db.save_request_logs

    async def counting_save(entries: Sequence[RequestLog]) -> None:
        batches.append(len(entries))
        await save(entries)

    monkeypatch.setattr(db, "save_request_logs", counting_save)
    tasks_before = len(background._tasks)

    async def one(i: int) -> None:
        _log(db, request_body={"i": i})

    await asyncio.gather(*(one(i) for i in range(500)))
    # Enqueued, not spawned: no task per request.
    assert len(background._tasks) == tasks_before
    await background.drain()

    assert await db.count_request_logs() == 500
    assert sum(batches) == 500
    assert max(batches) <= request_logger.BATCH_ROWS
    assert len(batches) < 10


async def test_full_queue_drops_and_counts(
    db: Database, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(request_logger, "QUEUE_MAX", 3)
    monkeypatch.setattr(request_logger, "_log_writer", None)
    dropped = metrics.request_log_dropped_total()

    for _ in range(5):  # no await: the writer cannot run in between
        _log(db)

    assert metrics.request_log_dropped_total() == dropped + 2
    await background.drain()
    assert await db.count_request_logs() == 3
    text = metrics.render(metrics.collect([], _NoStats()))
    assert f"llama_router_request_log_dropped_total {dropped + 2}" in text


class _NoStats:
    def model_entries(self) -> list[Any]:
        return []


async def test_stop_writes_queued_rows(db: Database) -> None:
    request_logger.start_log_writer()
    for _ in range(3):
        _log(db)
    await request_logger.stop_log_writer()
    assert await db.count_request_logs() == 3


def _request(headers: list[tuple[bytes, bytes]]) -> Request:
    return Request({"type": "http", "headers": headers, "path": "/x"})


async def test_request_size_is_the_raw_body(db: Database) -> None:
    # Content-Length, not the size of the body serialised again.
    _log(
        db,
        request=_request([(b"content-length", b"1234")]),
        request_body={"model": "m"},
    )
    cached = _request([])
    cached._body = b'{"model":   "m"}'
    _log(db, request=cached, request_body={"model": "m"})
    await background.drain()
    sizes = sorted(e.request_size for e in await db.get_request_logs(10))
    assert sizes == [16, 1234]


async def test_dashboard_log_count_is_cached(
    db: Database, monkeypatch: pytest.MonkeyPatch
) -> None:
    entry = RequestLog(protocol="ollama", endpoint="/api/chat")
    with monkeypatch.context() as m:
        m.setattr(dash_routes, "LOG_COUNT_TTL", 0.0)
        assert await dash_routes._log_count(db) == 0
        await db.save_request_log(entry)
        assert await dash_routes._log_count(db) == 1
    await db.save_request_log(entry)
    assert await dash_routes._log_count(db) == 2
    await db.save_request_log(entry)
    assert await dash_routes._log_count(db) == 2  # cached


async def _asgi_app(scope: Any, receive: Any, send: Any) -> None:
    raise NotImplementedError


async def test_bind_failure_stops_every_server() -> None:
    taken = socket.socket()
    taken.bind(("127.0.0.1", 0))
    taken.listen()
    port = taken.getsockname()[1]
    try:
        good = _Server(
            uvicorn.Config(_asgi_app, host="127.0.0.1", port=0, lifespan="off")
        )
        bad = _Server(
            uvicorn.Config(_asgi_app, host="127.0.0.1", port=port, lifespan="off")
        )
        with pytest.raises(ServerStartupError, match=str(port)):
            await asyncio.wait_for(serve_all([good, bad]), timeout=10)
        assert good.should_exit
    finally:
        taken.close()
