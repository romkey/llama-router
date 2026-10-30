"""Worker side: dial out to a hub and serve the jobs it sends.

The worker holds a single outbound WebSocket to the hub, announces which
protocols it serves, and replays each relayed HTTP request against a local
upstream (by default this router's own Ollama and llama.cpp APIs). It
reconnects forever with capped exponential backoff and jitter.

Only an allowlist of inference and model-listing endpoints is served. Model
pulls and deletes are refused unless ``worker_allow_admin`` is set.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import random
import time
from typing import Any

import httpx
from websockets.asyncio.client import ClientConnection, connect
from websockets.exceptions import ConnectionClosed, InvalidStatus, WebSocketException

from .. import __version__, router_identity
from . import protocol as p

logger = logging.getLogger(__name__)

_READ_PATHS = {
    ("GET", "/api/version"),
    ("GET", "/api/tags"),
    ("GET", "/api/ps"),
    ("POST", "/api/show"),
    ("GET", "/v1/models"),
    ("GET", "/health"),
    ("GET", "/props"),
    ("GET", "/v1/audio/voices"),
}
_INFERENCE_PATHS = {
    ("POST", path)
    for path in (
        "/api/chat",
        "/api/generate",
        "/api/embed",
        "/api/embeddings",
        "/v1/chat/completions",
        "/v1/completions",
        "/v1/embeddings",
        "/v1/responses",
        "/v1/images/generations",
        "/v1/images/edits",
        "/v1/audio/speech",
        "/v1/audio/transcriptions",
        # Decision questions (Jev's routes and their aliases).
        "/v1/decisions",
        "/decisions",
        "/v1/systemone",
        "/systemone",
    )
}
_ADMIN_PATHS = {("POST", "/api/pull"), ("DELETE", "/api/delete")}

_MAX_REQUEST_BODY = 64 * 1024 * 1024
_HTTP_TIMEOUT = httpx.Timeout(15.0, read=1800.0)
_BACKOFF_MAX = 60.0
_REJECTED_RETRY = 300.0
_REPLACED_RETRY = 30.0
_STABLE_SESSION = 30.0


def path_allowed(method: str, raw_path: str, allow_admin: bool) -> bool:
    path = raw_path.split("?", 1)[0]
    if not path.startswith("/") or "//" in path or "/../" in path + "/":
        return False
    if path.startswith("/v1/models/") and method == "GET":
        return True
    key = (method.upper(), path)
    if key in _READ_PATHS or key in _INFERENCE_PATHS:
        return True
    return allow_admin and key in _ADMIN_PATHS


class _Job:
    __slots__ = ("body", "body_done", "body_size", "id", "meta", "task")

    def __init__(self, req_id: int, meta: dict[str, Any]) -> None:
        self.id = req_id
        self.meta = meta
        self.body: list[bytes] = []
        self.body_size = 0
        self.body_done = asyncio.Event()
        self.task: asyncio.Task[None] | None = None


class _RejectedError(Exception):
    pass


class WorkerClient:
    """Maintains the outbound link to one hub."""

    def __init__(
        self,
        *,
        hub_url: str,
        token: str,
        ollama_url: str | None,
        llamacpp_url: str | None,
        max_concurrency: int = 4,
        allow_admin: bool = False,
        upstream_api_key: str = "",
    ) -> None:
        self.hub_url = hub_url
        self._token = token
        self._upstreams = {
            "ollama": (ollama_url or "").rstrip("/") or None,
            "llamacpp": (llamacpp_url or "").rstrip("/") or None,
        }
        self._max_concurrency = max(1, max_concurrency)
        self._allow_admin = allow_admin
        self._upstream_api_key = upstream_api_key.strip()
        self._http = httpx.AsyncClient(timeout=_HTTP_TIMEOUT)
        self._task: asyncio.Task[None] | None = None
        self._jobs: dict[int, _Job] = {}
        self._slots = asyncio.Semaphore(self._max_concurrency)
        self._send_lock = asyncio.Lock()
        self._ws: ClientConnection | None = None

        self.state = "idle"
        self.last_error: str | None = None
        self.connected_since: float | None = None
        self.worker_name: str | None = None
        # The hub's router ID; None until welcomed or from an older hub.
        self.hub_router_id: str | None = None
        self.jobs_total = 0
        self.next_retry_at: float | None = None

    # --- public ---

    @property
    def protocols(self) -> list[str]:
        return [t for t in p.TARGETS if self._upstreams.get(t)]

    def status(self) -> dict[str, Any]:
        return {
            "state": self.state,
            "hub_url": self.hub_url,
            "worker_name": self.worker_name,
            "connected_since": self.connected_since,
            "last_error": self.last_error,
            "in_flight": len(self._jobs),
            "jobs_total": self.jobs_total,
            "protocols": self.protocols,
            "next_retry_at": self.next_retry_at,
            "hub_router_id": self.hub_router_id,
        }

    def start(self) -> None:
        if self._task is None or self._task.done():
            self._task = asyncio.create_task(self._run(), name="worker-link")

    async def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
            self._task = None
        await self._http.aclose()
        self.state = "stopped"

    # --- connection loop ---

    async def _run(self) -> None:
        backoff = 1.0
        headers = {"Authorization": f"Bearer {self._token}"}
        while True:
            delay: float
            started = time.monotonic()
            self.state = "connecting"
            self.next_retry_at = None
            try:
                async with connect(
                    self.hub_url,
                    additional_headers=headers,
                    open_timeout=15,
                    ping_interval=20,
                    ping_timeout=20,
                    max_size=p.MAX_MESSAGE_SIZE,
                    user_agent_header=f"llama-router-worker/{__version__}",
                ) as ws:
                    await self._session(ws)
                self.last_error = "hub closed the connection"
                delay = backoff
            except asyncio.CancelledError:
                raise
            except InvalidStatus as exc:
                code = exc.response.status_code
                if code in (401, 403):
                    self.state = "rejected"
                    self.last_error = (
                        f"hub rejected the token (HTTP {code}); "
                        "check the worker token or ask the hub admin for a new one"
                    )
                    delay = _REJECTED_RETRY
                else:
                    self.last_error = f"hub returned HTTP {code}"
                    delay = backoff
            except _RejectedError as exc:
                self.state = "rejected"
                self.last_error = str(exc)
                delay = _REJECTED_RETRY
            except ConnectionClosed as exc:
                rcvd = exc.rcvd
                if rcvd is not None and rcvd.code in (
                    p.CLOSE_UNAUTHORIZED,
                    p.CLOSE_REVOKED,
                ):
                    self.state = "rejected"
                    self.last_error = rcvd.reason or "token revoked by hub"
                    delay = _REJECTED_RETRY
                elif rcvd is not None and rcvd.code == p.CLOSE_REPLACED:
                    self.last_error = (
                        "another worker connected with the same token; "
                        "each machine needs its own token"
                    )
                    delay = _REPLACED_RETRY
                elif rcvd is not None and rcvd.code == p.CLOSE_LOOP:
                    self.state = "rejected"
                    self.last_error = (
                        "the hub is this router (same router ID); a router cannot "
                        "be its own worker"
                    )
                    delay = _REJECTED_RETRY
                elif rcvd is not None and rcvd.code == p.CLOSE_PROTOCOL:
                    self.last_error = f"protocol error: {rcvd.reason}"
                    delay = _REJECTED_RETRY
                else:
                    self.last_error = f"connection lost: {exc}"
                    delay = backoff
            except (TimeoutError, OSError, WebSocketException) as exc:
                self.last_error = (
                    f"{type(exc).__name__}: {exc}" if str(exc) else type(exc).__name__
                )
                delay = backoff
            except Exception as exc:
                logger.exception("Worker link crashed")
                self.last_error = f"internal error: {exc}"
                delay = backoff
            finally:
                self.connected_since = None
                self._ws = None
                self._cancel_all_jobs()

            if time.monotonic() - started > _STABLE_SESSION:
                backoff = 1.0
            else:
                backoff = min(backoff * 2, _BACKOFF_MAX)
            delay = delay * (0.8 + random.random() * 0.4)  # noqa: S311  # reconnect jitter
            if self.state != "rejected":
                self.state = "backoff"
            self.next_retry_at = time.time() + delay
            logger.warning(
                "Worker link to %s down (%s); retrying in %.0fs",
                self.hub_url,
                self.last_error,
                delay,
            )
            await asyncio.sleep(delay)

    async def _session(self, ws: ClientConnection) -> None:
        self._ws = ws
        await ws.send(
            p.encode_json(
                {
                    "type": "hello",
                    "version": p.PROTOCOL_VERSION,
                    "protocols": self.protocols,
                    "max_concurrency": self._max_concurrency,
                    "software_version": __version__,
                    "router_id": router_identity.router_id(),
                }
            )
        )
        raw = await asyncio.wait_for(ws.recv(), timeout=15)
        if not isinstance(raw, str):
            raise _RejectedError("hub sent binary data before welcome")
        welcome = p.decode_json(raw)
        if welcome.get("type") != "welcome":
            raise _RejectedError(f"unexpected handshake reply {welcome.get('type')!r}")
        self.worker_name = str(welcome.get("worker_name") or "") or None
        hub_rid = router_identity.parse_id(welcome.get("router_id"))
        if hub_rid is not None and hub_rid == router_identity.router_id():
            await ws.close(
                code=p.CLOSE_LOOP, reason="worker and hub are the same router"
            )
            raise _RejectedError(
                "the hub is this router (same router ID); a router cannot "
                "be its own worker"
            )
        self.hub_router_id = hub_rid
        # This router leaves out its remotes that lead back to the hub.
        router_identity.set_hub_id(hub_rid)
        self.state = "connected"
        self.last_error = None
        self.connected_since = time.time()
        logger.info(
            "Worker link connected to %s as %r (protocols=%s)",
            self.hub_url,
            self.worker_name,
            ",".join(self.protocols),
        )

        async for message in ws:
            if isinstance(message, str):
                self._on_text(p.decode_json(message))
            else:
                self._on_bytes(message)

    # --- inbound frames ---

    def _on_text(self, msg: dict[str, Any]) -> None:
        kind = msg.get("type")
        if kind == "request":
            req_id = msg.get("id")
            if not isinstance(req_id, int) or req_id in self._jobs:
                return
            job = _Job(req_id, msg)
            self._jobs[req_id] = job
            self.jobs_total += 1
            job.task = asyncio.create_task(self._run_job(job))
        elif kind == "cancel":
            cancel_id = msg.get("id")
            target = self._jobs.get(cancel_id) if isinstance(cancel_id, int) else None
            if target is not None and target.task is not None:
                target.task.cancel()
        else:
            logger.debug("Worker ignoring hub frame %r", kind)

    def _on_bytes(self, data: bytes) -> None:
        kind, req_id, payload = p.decode_binary(data)
        job = self._jobs.get(req_id)
        if job is None:
            return
        if kind == p.REQ_BODY:
            job.body_size += len(payload)
            if job.body_size <= _MAX_REQUEST_BODY:
                job.body.append(payload)
        elif kind == p.REQ_END:
            job.body_done.set()

    def _cancel_all_jobs(self) -> None:
        for job in list(self._jobs.values()):
            if job.task is not None:
                job.task.cancel()
        self._jobs.clear()

    # --- outbound frames ---

    async def _send_text(self, msg: dict[str, Any]) -> None:
        ws = self._ws
        if ws is None:
            raise ConnectionError("not connected")
        async with self._send_lock:
            await ws.send(p.encode_json(msg))

    async def _send_bytes(self, data: bytes) -> None:
        ws = self._ws
        if ws is None:
            raise ConnectionError("not connected")
        async with self._send_lock:
            await ws.send(data)

    async def _reply_json(self, req_id: int, status: int, body: dict[str, Any]) -> None:
        data = json.dumps(body).encode()
        await self._send_text(
            {
                "type": "response",
                "id": req_id,
                "status": status,
                "headers": [["content-type", "application/json"]],
            }
        )
        await self._send_bytes(p.encode_binary(p.RESP_BODY, req_id, data))
        await self._send_bytes(p.encode_binary(p.RESP_END, req_id))

    # --- job execution ---

    async def _run_job(self, job: _Job) -> None:
        req_id = job.id
        try:
            async with self._slots:
                await self._execute(job)
        except asyncio.CancelledError:
            pass
        except ConnectionError, ConnectionClosed:
            pass
        except Exception as exc:
            logger.exception("Worker job %d failed", req_id)
            with contextlib.suppress(Exception):
                await self._send_text(
                    {
                        "type": "error",
                        "id": req_id,
                        "kind": "internal",
                        "message": str(exc),
                    }
                )
        finally:
            self._jobs.pop(req_id, None)

    async def _execute(self, job: _Job) -> None:
        req_id = job.id
        meta = job.meta
        target = str(meta.get("target") or "")
        method = str(meta.get("method") or "GET").upper()
        path = str(meta.get("path") or "/")

        await asyncio.wait_for(job.body_done.wait(), timeout=120)
        if job.body_size > _MAX_REQUEST_BODY:
            await self._reply_json(req_id, 413, {"error": "request body too large"})
            return

        base = self._upstreams.get(target)
        if not base:
            await self._reply_json(
                req_id, 502, {"error": f"worker does not serve {target or 'that'} API"}
            )
            return
        if not path_allowed(method, path, self._allow_admin):
            await self._reply_json(
                req_id,
                403,
                {"error": f"{p.REFUSED_PREFIX} {method} {path.split('?', 1)[0]}"},
            )
            return

        headers = p.filter_headers(meta.get("headers") or [])
        headers = [h for h in headers if h[0].lower() != p.HUB_HEADER.lower()]
        headers.append(["accept-encoding", "identity"])
        # Tells this router (the usual upstream) to use only the providers
        # shared with the hub; any other upstream ignores it.
        headers.append([p.HUB_HEADER, p.HUB_HEADER_VALUE])
        if self._upstream_api_key:
            headers.append(["API-KEY", self._upstream_api_key])

        request = self._http.build_request(
            method,
            base + path,
            headers=[(k, v) for k, v in headers],
            content=b"".join(job.body) or None,
        )
        try:
            response = await self._http.send(request, stream=True)
        except httpx.ConnectError as exc:
            await self._send_text(
                {"type": "error", "id": req_id, "kind": "connect", "message": str(exc)}
            )
            return
        except httpx.TimeoutException as exc:
            await self._send_text(
                {"type": "error", "id": req_id, "kind": "timeout", "message": str(exc)}
            )
            return

        try:
            await self._send_text(
                {
                    "type": "response",
                    "id": req_id,
                    "status": response.status_code,
                    "headers": p.filter_headers(response.headers.multi_items()),
                }
            )
            try:
                async for data in response.aiter_bytes():
                    for piece in p.chunks(data):
                        await self._send_bytes(
                            p.encode_binary(p.RESP_BODY, req_id, piece)
                        )
            except httpx.HTTPError as exc:
                kind = (
                    "timeout" if isinstance(exc, httpx.TimeoutException) else "upstream"
                )
                await self._send_text(
                    {"type": "error", "id": req_id, "kind": kind, "message": str(exc)}
                )
                return
            await self._send_bytes(p.encode_binary(p.RESP_END, req_id))
        finally:
            await response.aclose()


class WorkerLinkManager:
    """Owns the optional worker link and rebuilds it when settings change.

    Environment variables win over values saved from the dashboard so that
    container deployments can pin the hub URL and token.
    """

    DB_HUB_URL = "worker_hub_url"
    DB_TOKEN = "worker_token"  # noqa: S105  # settings key name, not a secret

    def __init__(self, db: Any) -> None:
        self._db = db
        self._client: WorkerClient | None = None
        self._lock = asyncio.Lock()
        self.source: str = "none"

    @property
    def db(self) -> Any:
        return self._db

    @staticmethod
    def env_configured() -> bool:
        from ..config import settings

        return bool(settings.worker_hub_url.strip() and settings.worker_token.strip())

    async def resolved_config(self) -> tuple[str, str, str]:
        from ..config import settings

        if self.env_configured():
            return settings.worker_hub_url.strip(), settings.worker_token.strip(), "env"
        url = (await self._db.get_app_setting(self.DB_HUB_URL) or "").strip()
        token = (await self._db.get_app_setting(self.DB_TOKEN) or "").strip()
        if url and token:
            return url, token, "dashboard"
        return url, token, "none"

    async def reload(self) -> None:
        from ..config import settings

        async with self._lock:
            if self._client is not None:
                await self._client.stop()
                self._client = None
                router_identity.set_hub_id(None)
            url, token, source = await self.resolved_config()
            self.source = source
            if source == "none":
                return
            ollama = settings.worker_ollama_url.strip() or (
                f"http://127.0.0.1:{settings.api_port}"
            )
            lcpp = settings.worker_llamacpp_url.strip() or (
                f"http://127.0.0.1:{settings.llamacpp_port}"
            )
            if ollama.lower() == "none":
                ollama = ""
            if lcpp.lower() == "none":
                lcpp = ""
            self._client = WorkerClient(
                hub_url=normalize_hub_url(url),
                token=token,
                ollama_url=ollama,
                llamacpp_url=lcpp,
                max_concurrency=settings.worker_max_concurrency,
                allow_admin=settings.worker_allow_admin,
                upstream_api_key=settings.worker_upstream_api_key,
            )
            self._client.start()

    async def save(self, hub_url: str, token: str | None) -> None:
        await self._db.set_app_setting(self.DB_HUB_URL, hub_url.strip())
        if token is not None:
            await self._db.set_app_setting(self.DB_TOKEN, token.strip())
        await self.reload()

    async def status(self) -> dict[str, Any]:
        url, token, source = await self.resolved_config()
        out: dict[str, Any] = {
            "configured": source != "none",
            "source": source,
            "env_locked": source == "env",
            "hub_url": url,
            "token_set": bool(token),
            "state": "disabled",
            "router_id": router_identity.router_id(),
        }
        if self._client is not None:
            live = self._client.status()
            live["socket_url"] = live.pop("hub_url")
            out.update(live)
        return out

    async def stop(self) -> None:
        async with self._lock:
            if self._client is not None:
                await self._client.stop()
                self._client = None
                router_identity.set_hub_id(None)


def normalize_hub_url(raw: str) -> str:
    """Accept ``https://hub``, ``wss://hub`` or a full socket URL."""
    url = raw.strip().rstrip("/")
    if url.startswith("https://"):
        url = "wss://" + url[len("https://") :]
    elif url.startswith("http://"):
        url = "ws://" + url[len("http://") :]
    elif not url.startswith(("ws://", "wss://")):
        url = "wss://" + url
    scheme, _, rest = url.partition("://")
    if "/" not in rest:
        url = f"{scheme}://{rest}{p.WS_PATH}"
    return url
