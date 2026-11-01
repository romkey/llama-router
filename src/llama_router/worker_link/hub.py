"""Hub side: accept worker sockets and expose each one as an ``httpx`` transport.

The provider manager builds ordinary :class:`OllamaClient` /
:class:`LlamaCppClient` objects for worker providers, but hands them a
:class:`WorkerTransport`. Every HTTP request those clients make is relayed
over the worker's WebSocket, so chat, streaming, embeddings, model listing and
benchmarks all work without worker-specific code paths.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import itertools
import logging
import secrets
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Any

import httpx
from starlette.websockets import WebSocket, WebSocketDisconnect, WebSocketState

from .. import router_identity
from ..config import settings
from . import protocol as p

logger = logging.getLogger(__name__)

TOKEN_PREFIX = "lrw_"  # noqa: S105  # public prefix, not a secret
_HELLO_TIMEOUT = 15.0
# A job whose response buffer stays full this long (its client is not
# reading) is failed, so one slow client cannot stall the whole socket.
_STALL_TIMEOUT = 30.0


def generate_worker_token() -> str:
    return TOKEN_PREFIX + secrets.token_urlsafe(32)


def worker_token_hash(token: str) -> str:
    """Tokens are 256-bit random strings, so a plain SHA-256 fingerprint is enough."""
    return hashlib.sha256(token.strip().encode("utf-8")).hexdigest()


def worker_token_prefix(token: str) -> str:
    return token[:12] + "…" if len(token) > 12 else token


def _bearer(websocket: WebSocket) -> str:
    raw = (websocket.headers.get("authorization") or "").strip()
    if raw.lower().startswith("bearer "):
        return raw[7:].strip()
    return ""


class _Pending:
    __slots__ = ("buffered", "finished", "id", "queue", "received", "room", "started")

    def __init__(self, req_id: int) -> None:
        self.id = req_id
        loop = asyncio.get_running_loop()
        self.started: asyncio.Future[tuple[int, list[list[str]]]] = loop.create_future()
        self.queue: asyncio.Queue[bytes | BaseException | None] = asyncio.Queue()
        self.finished = False
        # Body bytes queued and not yet read, and received in all.
        self.buffered = 0
        self.received = 0
        # Set while the queue has room for more body bytes.
        self.room = asyncio.Event()
        self.room.set()

    def take(self, item: bytes) -> None:
        """The reader took *item* off the queue."""
        self.buffered -= len(item)
        if self.buffered < settings.worker_response_buffer_bytes:
            self.room.set()

    def fail(self, exc: BaseException) -> None:
        """End the response with *exc*, dropping whatever is still queued."""
        while not self.queue.empty():
            self.queue.get_nowait()
        self.buffered = 0
        self.room.set()
        if not self.started.done():
            self.started.set_exception(exc)
            self.started.exception()
        else:
            self.queue.put_nowait(exc)


def _error_from_worker(kind: str, message: str, request: httpx.Request) -> Exception:
    text = f"worker: {message}" if message else "worker error"
    if kind == "connect":
        return httpx.ConnectError(text, request=request)
    if kind == "timeout":
        return httpx.ReadTimeout(text, request=request)
    return httpx.RemoteProtocolError(text, request=request)


class WorkerConnection:
    """One live worker socket, multiplexing many HTTP exchanges by request id."""

    def __init__(
        self,
        websocket: WebSocket,
        *,
        worker_id: int,
        name: str,
        provider_id: int,
        protocols: list[str],
        max_concurrency: int,
        software_version: str | None,
        remote_addr: str | None,
        router_id: str | None = None,
        expose: str = p.EXPOSE_COMBINED,
        on_catalog: Callable[[WorkerConnection], None] | None = None,
    ) -> None:
        self._ws = websocket
        self.worker_id = worker_id
        self.name = name
        self.provider_id = provider_id
        self.protocols = protocols
        self.max_concurrency = max_concurrency
        self.software_version = software_version
        self.remote_addr = remote_addr
        # The worker's router ID; None from workers older than router IDs.
        self.router_id = router_id
        # "individual": the worker lists its providers in catalog frames and
        # each appears here as a provider of its own.
        self.expose = expose
        self.catalog: list[dict[str, Any]] | None = None
        self._on_catalog = on_catalog
        self.connected_at = time.time()
        self.jobs_total = 0
        self._pending: dict[int, _Pending] = {}
        # Each request carries a ``httpx.Request`` for building exceptions later.
        self._requests: dict[int, httpx.Request] = {}
        self._ids = itertools.count(1)
        self._slots = asyncio.Semaphore(max_concurrency)
        self._send_lock = asyncio.Lock()
        self._closed = False

    @property
    def in_flight(self) -> int:
        return len(self._pending)

    @property
    def closed(self) -> bool:
        return self._closed

    def summary(self) -> dict[str, Any]:
        return {
            "connected": not self._closed,
            "connected_at": self.connected_at,
            "remote_addr": self.remote_addr,
            "protocols": list(self.protocols),
            "max_concurrency": self.max_concurrency,
            "in_flight": self.in_flight,
            "jobs_total": self.jobs_total,
            "software_version": self.software_version,
            "router_id": self.router_id,
            "expose": self.expose,
            "shared_providers": len(self.catalog) if self.catalog is not None else None,
        }

    # --- sending ---

    async def _send_text(self, msg: dict[str, Any]) -> None:
        if self._closed:
            raise ConnectionError("worker disconnected")
        async with self._send_lock:
            await self._ws.send_text(p.encode_json(msg))

    async def _send_bytes(self, data: bytes) -> None:
        if self._closed:
            raise ConnectionError("worker disconnected")
        async with self._send_lock:
            await self._ws.send_bytes(data)

    # --- request lifecycle ---

    def _finish(self, req_id: int) -> None:
        pend = self._pending.pop(req_id, None)
        self._requests.pop(req_id, None)
        if pend is not None and not pend.finished:
            pend.finished = True
            pend.room.set()
            self._slots.release()

    async def cancel(self, req_id: int) -> None:
        if req_id not in self._pending:
            return
        self._finish(req_id)
        with contextlib.suppress(Exception):
            await self._send_text({"type": "cancel", "id": req_id})

    async def open(
        self, target: str, request: httpx.Request, remote: int | None = None
    ) -> httpx.Response:
        """Relay *request*; *remote* names the provider on the worker it is for."""
        timeouts = request.extensions.get("timeout") or {}
        pool_t = timeouts.get("pool")
        read_t = timeouts.get("read")
        write_t = timeouts.get("write")

        if self._closed:
            raise httpx.ConnectError("worker disconnected", request=request)
        try:
            await asyncio.wait_for(self._slots.acquire(), timeout=pool_t)
        except TimeoutError:
            raise httpx.PoolTimeout(
                f"worker {self.name!r} is at its concurrency limit "
                f"({self.max_concurrency})",
                request=request,
            ) from None

        req_id = next(self._ids)
        pend = _Pending(req_id)
        self._pending[req_id] = pend
        self._requests[req_id] = request
        self.jobs_total += 1

        try:
            body = await request.aread()
            header = {
                "type": "request",
                "id": req_id,
                "target": target,
                "method": request.method,
                "path": request.url.raw_path.decode("ascii", errors="replace"),
                "headers": p.filter_headers(request.headers.multi_items()),
            }
            if remote is not None:
                header["provider"] = remote

            async def _send_all() -> None:
                await self._send_text(header)
                for chunk in p.chunks(body):
                    await self._send_bytes(p.encode_binary(p.REQ_BODY, req_id, chunk))
                await self._send_bytes(p.encode_binary(p.REQ_END, req_id))

            try:
                await asyncio.wait_for(_send_all(), timeout=write_t)
            except TimeoutError:
                raise httpx.WriteTimeout(
                    "timed out sending request to worker", request=request
                ) from None
            except (OSError, RuntimeError, WebSocketDisconnect) as exc:
                raise httpx.WriteError(
                    f"worker socket closed while sending: {exc}", request=request
                ) from exc

            try:
                status, headers = await asyncio.wait_for(
                    asyncio.shield(pend.started), timeout=read_t
                )
            except TimeoutError:
                raise httpx.ReadTimeout(
                    "timed out waiting for worker response", request=request
                ) from None
        except BaseException:
            await self.cancel(req_id)
            raise

        return httpx.Response(
            status,
            headers=[(k, v) for k, v in headers],
            stream=_WorkerStream(self, pend, read_t, request),
            request=request,
        )

    # --- receiving ---

    def _on_text(self, msg: dict[str, Any]) -> None:
        kind = msg.get("type")
        req_id = msg.get("id")
        pend = self._pending.get(req_id) if isinstance(req_id, int) else None
        if kind == "response":
            if pend is None or pend.started.done():
                return
            status = int(msg.get("status") or 502)
            headers = p.filter_headers(msg.get("headers") or [])
            pend.started.set_result((status, headers))
        elif kind == "error":
            if pend is None:
                return
            request = self._requests.get(pend.id)
            exc = _error_from_worker(
                str(msg.get("kind") or ""),
                str(msg.get("message") or ""),
                request or httpx.Request("GET", "worker://unknown"),
            )
            if not pend.started.done():
                pend.started.set_exception(exc)
            else:
                pend.queue.put_nowait(exc)
            self._finish(pend.id)
        elif kind == "catalog":
            self._take_catalog(msg.get("providers"))
        else:
            logger.debug("Worker %s sent unknown frame type %r", self.name, kind)

    def _take_catalog(self, raw: Any) -> None:
        if self.expose != p.EXPOSE_INDIVIDUAL or not isinstance(raw, list):
            return
        entries: list[dict[str, Any]] = []
        seen: set[int] = set()
        for item in raw[: max(0, settings.worker_max_shared_providers)]:
            if not isinstance(item, dict):
                continue
            rid, name = item.get("id"), item.get("name")
            if not isinstance(rid, int) or isinstance(rid, bool) or rid in seen:
                continue
            if not isinstance(name, str) or not name.strip():
                continue
            seen.add(rid)
            locality = item.get("locality")
            entries.append(
                {
                    "id": rid,
                    "name": name.strip()[:128],
                    "locality": locality if locality in ("local", "cloud") else "local",
                }
            )
        self.catalog = entries
        if self._on_catalog is not None:
            self._on_catalog(self)

    async def _on_bytes(self, data: bytes) -> None:
        kind, req_id, payload = p.decode_binary(data)
        pend = self._pending.get(req_id)
        if pend is None:
            return
        if kind == p.RESP_BODY:
            pend.received += len(payload)
            if pend.received > settings.worker_max_response_bytes:
                await self._abort(
                    pend,
                    f"response from worker {self.name!r} is over "
                    f"{settings.worker_max_response_bytes} bytes",
                )
                return
            if pend.buffered >= settings.worker_response_buffer_bytes:
                # Backpressure: stop reading from the socket until the client
                # catches up (TCP slows the worker down meanwhile).
                pend.room.clear()
                try:
                    await asyncio.wait_for(pend.room.wait(), _STALL_TIMEOUT)
                except TimeoutError:
                    await self._abort(
                        pend, "client stopped reading the worker's response"
                    )
                    return
                if pend.finished:
                    return
            pend.buffered += len(payload)
            pend.queue.put_nowait(payload)
        elif kind == p.RESP_END:
            pend.queue.put_nowait(None)
            self._finish(req_id)

    async def _abort(self, pend: _Pending, reason: str) -> None:
        """Fail one response and tell the worker to stop sending it."""
        logger.warning("Worker %s: %s", self.name, reason)
        request = self._requests.get(pend.id) or httpx.Request(
            "GET", "worker://unknown"
        )
        pend.fail(httpx.RemoteProtocolError(reason, request=request))
        await self.cancel(pend.id)

    async def read_loop(self) -> None:
        while True:
            message = await self._ws.receive()
            mtype = message.get("type")
            if mtype == "websocket.disconnect":
                return
            if message.get("text") is not None:
                try:
                    self._on_text(p.decode_json(message["text"]))
                except ValueError as exc:
                    logger.warning(
                        "Worker %s sent bad control frame: %s", self.name, exc
                    )
            elif message.get("bytes") is not None:
                try:
                    await self._on_bytes(message["bytes"])
                except ValueError as exc:
                    logger.warning("Worker %s sent bad data frame: %s", self.name, exc)

    def fail_all(self, reason: str) -> None:
        self._closed = True
        for req_id, pend in list(self._pending.items()):
            request = self._requests.get(req_id) or httpx.Request(
                "GET", "worker://unknown"
            )
            exc = httpx.RemoteProtocolError(
                f"worker {self.name!r} disconnected: {reason}", request=request
            )
            if not pend.started.done():
                pend.started.set_exception(exc)
                # Nobody may await it if the caller already gave up.
                pend.started.exception()
            else:
                pend.queue.put_nowait(exc)
            self._finish(req_id)

    async def close(self, code: int, reason: str = "") -> None:
        self.fail_all(reason or "closed by hub")
        if self._ws.application_state == WebSocketState.CONNECTED:
            with contextlib.suppress(Exception):
                await self._ws.close(code=code, reason=reason)


class _WorkerStream(httpx.AsyncByteStream):
    def __init__(
        self,
        conn: WorkerConnection,
        pend: _Pending,
        read_timeout: float | None,
        request: httpx.Request,
    ) -> None:
        self._conn = conn
        self._pend = pend
        self._read_timeout = read_timeout
        self._request = request
        self._eof = False

    async def __aiter__(self) -> AsyncIterator[bytes]:
        while not self._eof:
            try:
                item = await asyncio.wait_for(
                    self._pend.queue.get(), timeout=self._read_timeout
                )
            except TimeoutError:
                await self._conn.cancel(self._pend.id)
                raise httpx.ReadTimeout(
                    "timed out reading from worker", request=self._request
                ) from None
            if item is None:
                self._eof = True
                return
            if isinstance(item, bytes):
                self._pend.take(item)
            if isinstance(item, BaseException):
                self._eof = True
                if isinstance(item, httpx.HTTPError):
                    raise httpx.ReadError(str(item), request=self._request) from item
                raise item
            yield item

    async def aclose(self) -> None:
        if not self._eof:
            self._eof = True
            await self._conn.cancel(self._pend.id)


class WorkerTransport(httpx.AsyncBaseTransport):
    """``httpx`` transport that sends every request through a worker socket."""

    def __init__(self, hub: WorkerHub, provider_id: int, target: str) -> None:
        if target not in p.TARGETS:
            raise ValueError(f"unknown worker target {target!r}")
        self._hub = hub
        self._provider_id = provider_id
        self._target = target

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        conn = self._hub.get(self._provider_id)
        if conn is None or conn.closed:
            raise httpx.ConnectError("worker is not connected", request=request)
        self._hub.count_job(self._provider_id)
        return await conn.open(
            self._target, request, self._hub.remote_id(self._provider_id)
        )


# Callbacks receive the connection itself so a handler that runs late can tell
# whether its socket is still the provider's current one.
ConnectCallback = Callable[[WorkerConnection], Awaitable[None]]
DisconnectCallback = Callable[[WorkerConnection], Awaitable[None]]
CatalogCallback = Callable[[WorkerConnection], Awaitable[None]]


class WorkerHub:
    """Registry of live worker connections, keyed by provider id."""

    def __init__(self) -> None:
        self._conns: dict[int, WorkerConnection] = {}
        self._on_connect: ConnectCallback | None = None
        self._on_disconnect: DisconnectCallback | None = None
        self._on_catalog: CatalogCallback | None = None
        # Providers a worker shares individually: their provider id ->
        # (the worker's own provider id, the provider's id on the worker).
        self._children: dict[int, tuple[int, int]] = {}
        self._bg: set[asyncio.Task[None]] = set()
        # Jobs dispatched per provider since start-up, counted as they are
        # sent so the total never dips while a socket is being replaced.
        self._jobs: dict[int, int] = {}
        # Catalog syncs, per worker provider: the connection whose latest
        # catalog still has to be applied, the task applying them, and when
        # the last one started.
        self._catalog_due: dict[int, WorkerConnection] = {}
        self._catalog_tasks: dict[int, asyncio.Task[None]] = {}
        self._catalog_at: dict[int, float] = {}

    def set_callbacks(
        self,
        on_connect: ConnectCallback | None,
        on_disconnect: DisconnectCallback | None,
        on_catalog: CatalogCallback | None = None,
    ) -> None:
        self._on_connect = on_connect
        self._on_disconnect = on_disconnect
        self._on_catalog = on_catalog

    def add_child(self, provider_id: int, parent_id: int, remote_id: int) -> None:
        """Reach *provider_id* through the socket of worker provider *parent_id*."""
        self._children[provider_id] = (parent_id, remote_id)

    def remove_child(self, provider_id: int) -> None:
        self._children.pop(provider_id, None)

    def children_of(self, parent_id: int) -> list[int]:
        return [c for c, (par, _) in self._children.items() if par == parent_id]

    def remote_id(self, provider_id: int) -> int | None:
        child = self._children.get(provider_id)
        return child[1] if child is not None else None

    def transport(self, provider_id: int, target: str) -> WorkerTransport:
        return WorkerTransport(self, provider_id, target)

    def get(self, provider_id: int) -> WorkerConnection | None:
        """The socket that serves *provider_id*: its own, or its worker's."""
        child = self._children.get(provider_id)
        if child is not None:
            conn = self._conns.get(child[0])
            # A worker that went back to sharing as one provider no longer
            # serves its old individual providers.
            if conn is None or conn.expose != p.EXPOSE_INDIVIDUAL:
                return None
            return conn
        return self._conns.get(provider_id)

    def is_connected(self, provider_id: int) -> bool:
        conn = self.get(provider_id)
        return conn is not None and not conn.closed

    def count_job(self, provider_id: int) -> None:
        # Jobs for a provider shared individually count toward its worker.
        child = self._children.get(provider_id)
        pid = child[0] if child is not None else provider_id
        self._jobs[pid] = self._jobs.get(pid, 0) + 1

    def jobs_total(self, provider_id: int) -> int:
        """Requests sent to the worker since start-up, across reconnects."""
        return self._jobs.get(provider_id, 0)

    def connections(self) -> list[WorkerConnection]:
        return list(self._conns.values())

    async def disconnect(self, provider_id: int, code: int, reason: str) -> None:
        conn = self._conns.get(provider_id)
        if conn is not None:
            await conn.close(code, reason)

    async def close_all(self) -> None:
        for conn in list(self._conns.values()):
            await conn.close(1001, "hub shutting down")

    def _catalog_received(self, conn: WorkerConnection) -> None:
        """Apply *conn*'s catalog soon: at most once per
        ``worker_catalog_min_interval_seconds`` per worker, and only the
        latest of the catalogs that arrive in between."""
        if self._on_catalog is None:
            return
        pid = conn.provider_id
        self._catalog_due[pid] = conn
        task = self._catalog_tasks.get(pid)
        if task is None or task.done():
            task = asyncio.ensure_future(self._apply_catalogs(pid))
            self._catalog_tasks[pid] = task
            self._bg.add(task)
            task.add_done_callback(self._bg.discard)

    async def _apply_catalogs(self, pid: int) -> None:
        while pid in self._catalog_due:
            interval = max(0.0, settings.worker_catalog_min_interval_seconds)
            last = self._catalog_at.get(pid)
            if last is not None:
                wait = last + interval - time.monotonic()
                if wait > 0:
                    await asyncio.sleep(wait)
            # The newest catalog, which may have come in while we waited.
            conn = self._catalog_due.pop(pid)
            self._catalog_at[pid] = time.monotonic()
            if conn.closed or self._on_catalog is None:
                continue
            try:
                await self._on_catalog(conn)
            except Exception:
                logger.exception("Worker %s: provider list sync failed", conn.name)
        # Checked and dropped with no await in between, so a catalog that
        # arrives later starts a new task.
        if self._catalog_tasks.get(pid) is asyncio.current_task():
            del self._catalog_tasks[pid]

    def _spawn(self, coro: Awaitable[None]) -> None:
        task = asyncio.ensure_future(coro)
        self._bg.add(task)
        task.add_done_callback(self._bg.discard)

    async def serve(self, websocket: WebSocket, db: Any) -> None:
        """Run one worker session from handshake to disconnect."""
        token = _bearer(websocket)
        row = (
            await db.get_worker_by_token_hash(worker_token_hash(token))
            if token
            else None
        )
        remote = websocket.client.host if websocket.client else None
        if not row or not row.get("enabled", True) or row.get("provider_id") is None:
            logger.warning("Rejected worker connection from %s: bad token", remote)
            await websocket.close(code=p.CLOSE_UNAUTHORIZED)
            return

        await websocket.accept()
        try:
            raw = await asyncio.wait_for(websocket.receive_text(), _HELLO_TIMEOUT)
            hello = p.decode_json(raw)
            if hello.get("type") != "hello":
                raise ValueError("first frame must be hello")
            version = int(hello.get("version") or 0)
            if version != p.PROTOCOL_VERSION:
                raise ValueError(
                    f"unsupported protocol version {version}; "
                    f"hub speaks {p.PROTOCOL_VERSION}"
                )
            protocols = [x for x in hello.get("protocols") or [] if x in p.TARGETS]
            if not protocols:
                raise ValueError("hello must list at least one protocol")
            max_c = max(1, min(int(hello.get("max_concurrency") or 4), 64))
            expose = (
                p.EXPOSE_INDIVIDUAL
                if hello.get("expose") == p.EXPOSE_INDIVIDUAL
                else p.EXPOSE_COMBINED
            )
            worker_rid = router_identity.parse_id(hello.get("router_id"))
        except (TimeoutError, ValueError, TypeError, WebSocketDisconnect) as exc:
            logger.warning("Worker %s handshake failed: %s", row.get("name"), exc)
            if websocket.application_state == WebSocketState.CONNECTED:
                await websocket.close(code=p.CLOSE_PROTOCOL, reason=str(exc)[:120])
            return

        own_rid = router_identity.router_id()
        if worker_rid is not None and worker_rid == own_rid:
            logger.warning(
                "Refused worker %s from %s: it is this router (router ID %s)",
                row.get("name"),
                remote,
                own_rid,
            )
            await websocket.close(
                code=p.CLOSE_LOOP, reason="worker and hub are the same router"
            )
            return

        provider_id = int(row["provider_id"])
        name = str(row["name"])
        conn = WorkerConnection(
            websocket,
            worker_id=int(row["id"]),
            name=name,
            provider_id=provider_id,
            protocols=protocols,
            max_concurrency=max_c,
            software_version=str(hello.get("software_version") or "") or None,
            remote_addr=remote,
            router_id=worker_rid,
            expose=expose,
            on_catalog=self._catalog_received,
        )

        previous = self._conns.get(provider_id)
        if previous is not None:
            logger.info("Worker %s reconnected; closing previous socket", name)
            await previous.close(p.CLOSE_REPLACED, "replaced by a newer connection")
        self._conns[provider_id] = conn

        try:
            await websocket.send_text(
                p.encode_json(
                    {
                        "type": "welcome",
                        "version": p.PROTOCOL_VERSION,
                        "worker_name": name,
                        "provider_id": provider_id,
                        "router_id": own_rid,
                    }
                )
            )
            logger.info(
                "Worker %s connected from %s (protocols=%s, max_concurrency=%d)",
                name,
                remote,
                ",".join(protocols),
                max_c,
            )
            try:
                await db.touch_worker_connected(int(row["id"]), remote)
            except Exception as exc:
                logger.debug("Could not record worker connect time: %s", exc)
            if self._on_connect is not None:
                self._spawn(self._on_connect(conn))
            await conn.read_loop()
        except WebSocketDisconnect:
            pass
        except Exception as exc:
            logger.warning("Worker %s session error: %s", name, exc)
        finally:
            conn.fail_all("socket closed")
            if self._conns.get(provider_id) is conn:
                del self._conns[provider_id]
                logger.info("Worker %s disconnected", name)
                if self._on_disconnect is not None:
                    self._spawn(self._on_disconnect(conn))
