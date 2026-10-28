"""Hub <-> worker links over a real WebSocket, with a fake Ollama upstream."""

from __future__ import annotations

import asyncio
import json
import math
import time

import httpx
import pytest
import pytest_asyncio
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from llama_router.api import deps as api_deps
from llama_router.api.app import app as api_app
from llama_router.database import Database
from llama_router.llamacpp_api import deps as v1_deps
from llama_router.models import ProviderProtocol, ProviderStatus
from llama_router.probe import MISSING, NEEDS_KEY, _classify
from llama_router.provider_manager import ProviderManager
from llama_router.router import Router
from llama_router.worker_link import protocol as p
from llama_router.worker_link import worker as worker_mod
from llama_router.worker_link.hub import (
    generate_worker_token,
    worker_token_hash,
    worker_token_prefix,
)
from llama_router.worker_link.worker import (
    WorkerClient,
    normalize_hub_url,
    path_allowed,
)

# --- pure helpers ---


def test_binary_frame_roundtrip() -> None:
    data = p.encode_binary(p.RESP_BODY, 70000, b"hello")
    assert p.decode_binary(data) == (p.RESP_BODY, 70000, b"hello")


def test_binary_frame_too_short() -> None:
    with pytest.raises(ValueError, match="binary frame too short"):
        p.decode_binary(b"\x01\x00")


def test_filter_headers_drops_hop_by_hop_and_credentials() -> None:
    out = p.filter_headers(
        [
            ("Content-Type", "application/json"),
            ("Host", "x"),
            ("Authorization", "Bearer secret"),
            ("API-KEY", "k"),
            ("Transfer-Encoding", "chunked"),
            ("X-Custom", "1"),
        ]
    )
    assert out == [["Content-Type", "application/json"], ["X-Custom", "1"]]


@pytest.mark.parametrize(
    ("method", "path", "admin", "ok"),
    [
        ("POST", "/api/chat", False, True),
        ("GET", "/api/tags", False, True),
        ("GET", "/v1/models/llama3", False, True),
        ("POST", "/v1/chat/completions?x=1", False, True),
        ("POST", "/api/pull", False, False),
        ("POST", "/api/pull", True, True),
        ("DELETE", "/api/delete", False, False),
        ("GET", "/api/chat", False, False),
        ("POST", "/api/../admin", False, False),
        ("POST", "//api/chat", False, False),
        ("GET", "/users", True, False),
        ("POST", "/v1/systemone", False, True),
        ("POST", "/v1/decisions", False, True),
        ("POST", "/systemone", False, True),
        ("GET", "/v1/decisions", False, False),
    ],
)
def test_path_allowed(method: str, path: str, admin: bool, ok: bool) -> None:
    assert path_allowed(method, path, admin) is ok


def test_probe_reads_worker_refusal_as_missing() -> None:
    req = httpx.Request("POST", "http://worker/v1/systemone")
    refusal = httpx.Response(
        403, json={"error": "worker does not allow POST /v1/systemone"}, request=req
    )
    assert _classify(refusal) == MISSING
    keyed = httpx.Response(403, json={"error": "invalid api key"}, request=req)
    assert _classify(keyed) == NEEDS_KEY


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("https://hub.example.org", "wss://hub.example.org/worker/ws"),
        ("http://10.0.0.5:11434/", "ws://10.0.0.5:11434/worker/ws"),
        ("hub.example.org", "wss://hub.example.org/worker/ws"),
        ("wss://hub/custom/path", "wss://hub/custom/path"),
    ],
)
def test_normalize_hub_url(raw: str, expected: str) -> None:
    assert normalize_hub_url(raw) == expected


# --- end-to-end fixtures ---


def _fake_ollama() -> tuple[FastAPI, dict]:
    app = FastAPI()
    seen: dict = {
        "chat_bodies": [],
        "stream_cancelled": False,
        "api_keys": [],
        "v1_chat_bodies": [],
        "systemone_bodies": [],
        "serve_systemone": True,
        "via": [],
    }

    @app.get("/api/version")
    async def version(request: Request):
        seen["api_keys"].append(request.headers.get("api-key"))
        return {"version": "0.0-test"}

    @app.get("/api/tags")
    async def tags(request: Request):
        seen["via"].append(request.headers.get("x-llama-router-via"))
        return {"models": [{"name": "tiny:1b", "size": 123, "digest": "sha256:abc"}]}

    @app.get("/api/ps")
    async def ps():
        return {"models": [{"name": "tiny:1b", "size_vram": 1}]}

    @app.post("/api/chat")
    async def chat(request: Request):
        body = await request.json()
        seen["chat_bodies"].append(body)
        if not body.get("stream"):
            return {"message": {"role": "assistant", "content": "hi"}, "done": True}

        n = int(body.get("options", {}).get("num_chunks", 3))
        delay = float(body.get("options", {}).get("delay", 0.0))

        async def gen():
            try:
                for i in range(n):
                    yield (
                        json.dumps({"message": {"content": f"t{i}"}}) + "\n"
                    ).encode()
                    await asyncio.sleep(delay)
                yield (json.dumps({"done": True}) + "\n").encode()
            except asyncio.CancelledError:
                seen["stream_cancelled"] = True
                raise

        return StreamingResponse(gen(), media_type="application/x-ndjson")

    @app.post("/api/embed")
    async def embed(request: Request):
        body = await request.json()
        return JSONResponse({"embeddings": [[0.1] * 4], "echo": body})

    @app.post("/v1/chat/completions")
    async def v1_chat(request: Request):
        body = await request.json()
        if not body.get("messages"):
            return JSONResponse({"error": "messages required"}, status_code=422)
        seen["v1_chat_bodies"].append(body)
        choice: dict = {"message": {"role": "assistant", "content": "A"}}
        if body.get("logprobs"):
            top = [
                {"token": "A", "logprob": math.log(0.9)},
                {"token": "B", "logprob": math.log(0.1)},
            ][: body.get("top_logprobs") or 1]
            choice["logprobs"] = {"content": [{**top[0], "top_logprobs": top}]}
        return {
            "choices": [choice],
            "usage": {"prompt_tokens": 20, "completion_tokens": 1},
        }

    @app.post("/v1/systemone")
    async def systemone(request: Request):
        if not seen["serve_systemone"]:
            return JSONResponse({"detail": "Not Found"}, status_code=404)
        body = await request.json()
        if not body.get("questions"):
            return JSONResponse({"error": "questions required"}, status_code=422)
        seen["systemone_bodies"].append(body)
        return {
            "model": body["model"],
            "answers": {"is_urgent": {"type": "noul", "noul": 0.25}},
        }

    @app.post("/api/pull")
    async def pull():
        return {"status": "should never be reached"}

    return app, seen


async def _serve(app) -> tuple[uvicorn.Server, asyncio.Task, int]:
    config = uvicorn.Config(
        app, host="127.0.0.1", port=0, log_level="warning", lifespan="off"
    )
    server = uvicorn.Server(config)
    task = asyncio.create_task(server.serve())
    for _ in range(200):
        if server.started:
            break
        await asyncio.sleep(0.01)
    port = server.servers[0].sockets[0].getsockname()[1]
    return server, task, port


async def _stop(server: uvicorn.Server, task: asyncio.Task) -> None:
    server.should_exit = True
    try:
        await asyncio.wait_for(task, 5)
    except TimeoutError:
        task.cancel()


async def _wait_for(cond, timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        res = cond()
        if asyncio.iscoroutine(res):
            res = await res
        if res:
            return
        await asyncio.sleep(0.02)
    raise AssertionError("condition not met in time")


@pytest_asyncio.fixture
async def link(tmp_path):
    db = Database(str(tmp_path / "hub.db"))
    await db.connect()
    pm = ProviderManager(db)
    rt = Router(db, pm)
    pm.attach_router(rt)
    api_deps.init(db, pm, rt)
    v1_deps.init(db, pm, rt)
    await db.set_allow_unauthenticated(True)

    upstream_app, seen = _fake_ollama()
    up_server, up_task, up_port = await _serve(upstream_app)
    hub_server, hub_task, hub_port = await _serve(api_app)

    provider = await pm.add_worker_provider("alice-gpu")
    token = generate_worker_token()
    worker_id = await db.create_worker(
        "alice-gpu", worker_token_hash(token), worker_token_prefix(token), provider.id
    )

    def make_worker(tok: str = token, **kw) -> WorkerClient:
        return WorkerClient(
            hub_url=normalize_hub_url(f"http://127.0.0.1:{hub_port}"),
            token=tok,
            ollama_url=f"http://127.0.0.1:{up_port}",
            llamacpp_url=None,
            **kw,
        )

    ctx = {
        "db": db,
        "pm": pm,
        "provider_id": provider.id,
        "worker_id": worker_id,
        "token": token,
        "seen": seen,
        "make_worker": make_worker,
        "workers": [],
        "rt": rt,
        "hub_url": f"http://127.0.0.1:{hub_port}",
    }
    yield ctx

    for w in ctx["workers"]:
        await w.stop()
    await pm.worker_hub.close_all()
    await _stop(hub_server, hub_task)
    await _stop(up_server, up_task)
    await pm.stop()
    await db.close()


async def _connected_worker(ctx, **kw) -> WorkerClient:
    w = ctx["make_worker"](**kw)
    ctx["workers"].append(w)
    w.start()
    pm, pid, db = ctx["pm"], ctx["provider_id"], ctx["db"]

    async def ready():
        prov = await db.get_provider(pid)
        return w.state == "connected" and prov.status == ProviderStatus.IDLE

    await _wait_for(ready)
    assert pm.worker_hub.is_connected(pid)
    return w


# --- end-to-end tests ---


async def test_worker_connects_and_models_are_discovered(link) -> None:
    w = await _connected_worker(link)
    db, pid = link["db"], link["provider_id"]
    models = await db.get_provider_models(pid)
    assert [m.name for m in models] == ["tiny:1b"]
    prov = await db.get_provider(pid)
    assert prov.protocol == ProviderProtocol.OLLAMA
    assert prov.is_worker
    assert w.worker_name == "alice-gpu"
    row = await db.get_worker(link["worker_id"])
    assert row["last_connected_at"] is not None
    assert row["last_remote_addr"] == "127.0.0.1"


async def test_non_streaming_and_streaming_chat_through_socket(link) -> None:
    await _connected_worker(link)
    client = link["pm"].get_ollama_client(link["provider_id"])

    reply = await client.chat({"model": "tiny:1b", "messages": []})
    assert reply["message"]["content"] == "hi"

    raw = b""
    async for chunk in client.chat_stream(
        {"model": "tiny:1b", "messages": [], "options": {"num_chunks": 5}}
    ):
        raw += chunk
    lines = [json.loads(x) for x in raw.decode().splitlines() if x.strip()]
    assert [ln.get("message", {}).get("content") for ln in lines[:5]] == [
        "t0",
        "t1",
        "t2",
        "t3",
        "t4",
    ]
    assert lines[-1] == {"done": True}
    conn = link["pm"].worker_hub.get(link["provider_id"])
    assert conn.in_flight == 0


async def test_large_body_is_chunked_both_ways(link) -> None:
    await _connected_worker(link)
    client = link["pm"].get_ollama_client(link["provider_id"])
    big = "x" * (p.MAX_CHUNK * 3 + 17)
    out = await client.embed({"model": "tiny:1b", "input": big})
    assert out["echo"]["input"] == big


async def test_admin_paths_are_refused_by_default(link) -> None:
    await _connected_worker(link)
    client = link["pm"].get_ollama_client(link["provider_id"])
    with pytest.raises(httpx.HTTPStatusError) as exc:
        await client.delete_model("tiny:1b")
    assert exc.value.response.status_code == 403


async def test_worker_marks_relayed_jobs_as_from_the_hub(link) -> None:
    """Every relayed job carries X-Llama-Router-Via: hub, so the worker's own
    router limits it to the providers shared with the hub."""
    await _connected_worker(link)
    assert link["seen"]["via"]
    assert link["seen"]["via"][-1] == "hub"


async def test_upstream_api_key_is_added_by_worker(link) -> None:
    await _connected_worker(link, upstream_api_key="local-key")
    client = link["pm"].get_ollama_client(link["provider_id"])
    assert await client.is_reachable()
    assert "local-key" in link["seen"]["api_keys"]


async def test_abandoned_stream_cancels_upstream(link) -> None:
    w = await _connected_worker(link)
    client = link["pm"].get_ollama_client(link["provider_id"])
    gen = client.chat_stream(
        {
            "model": "tiny:1b",
            "messages": [],
            "options": {"num_chunks": 200, "delay": 0.05},
        }
    )
    first = await gen.__anext__()
    assert b"t0" in first
    await gen.aclose()

    conn = link["pm"].worker_hub.get(link["provider_id"])
    await _wait_for(lambda: conn.in_flight == 0)
    await _wait_for(lambda: w.status()["in_flight"] == 0)
    await _wait_for(lambda: link["seen"]["stream_cancelled"])


async def test_concurrency_limit_is_respected(link) -> None:
    await _connected_worker(link, max_concurrency=1)
    pm, pid = link["pm"], link["provider_id"]
    conn = pm.worker_hub.get(pid)
    assert conn.max_concurrency == 1
    client = pm.get_ollama_client(pid)

    async def slow():
        out = b""
        async for c in client.chat_stream(
            {
                "model": "tiny:1b",
                "messages": [],
                "options": {"num_chunks": 5, "delay": 0.05},
            }
        ):
            out += c
        return out

    results = await asyncio.gather(slow(), slow(), slow())
    assert all(b'"done": true' in r for r in results)
    assert conn.in_flight == 0


async def test_bad_token_is_rejected(link) -> None:
    w = link["make_worker"](tok="lrw_not-a-real-token")
    link["workers"].append(w)
    w.start()
    await _wait_for(lambda: w.state == "rejected")
    assert "rejected" in (w.last_error or "")
    assert not link["pm"].worker_hub.is_connected(link["provider_id"])


async def test_disconnect_marks_offline_and_worker_reconnects(link) -> None:
    w = await _connected_worker(link)
    pm, pid, db = link["pm"], link["provider_id"], link["db"]
    first = pm.worker_hub.get(pid)
    # Discovery already sent jobs; the hub's total must survive the restart.
    before = pm.worker_hub.jobs_total(pid)
    assert before > 0

    await pm.worker_hub.disconnect(pid, 1011, "test restart")

    async def offline():
        return (await db.get_provider(pid)).status == ProviderStatus.OFFLINE

    await _wait_for(offline)
    # Backoff starts at about one second.
    await _wait_for(
        lambda: w.state == "connected" and pm.worker_hub.get(pid) not in (None, first),
        timeout=8,
    )

    async def idle():
        return (await db.get_provider(pid)).status == ProviderStatus.IDLE

    await _wait_for(idle)
    assert pm.worker_hub.jobs_total(pid) > before


async def test_revoked_token_stops_reconnecting(link) -> None:
    w = await _connected_worker(link)
    pm, pid = link["pm"], link["provider_id"]
    await pm.worker_hub.disconnect(pid, p.CLOSE_REVOKED, "token rotated")
    await _wait_for(lambda: w.state == "rejected")
    assert w.last_error == "token rotated"
    assert w.next_retry_at is not None
    assert w.next_retry_at - time.time() > 200


async def test_second_connection_with_same_token_replaces_first(link) -> None:
    w1 = await _connected_worker(link)
    w2 = link["make_worker"]()
    link["workers"].append(w2)
    w2.start()
    await _wait_for(lambda: w2.state == "connected")
    await _wait_for(lambda: w1.state != "connected")
    assert "same token" in (w1.last_error or "")


async def test_removing_worker_provider_deletes_worker_row(link) -> None:
    w = await _connected_worker(link)
    pm, pid, db = link["pm"], link["provider_id"], link["db"]
    await pm.remove_provider(pid)
    assert await db.get_worker(link["worker_id"]) is None
    await _wait_for(lambda: w.state == "rejected")


async def test_requests_fail_fast_when_worker_offline(link) -> None:
    pm, pid = link["pm"], link["provider_id"]
    prov = await link["db"].get_provider(pid)
    await pm._rebuild_clients(prov)
    client = pm.get_ollama_client(pid)
    with pytest.raises(httpx.ConnectError):
        await client.chat({"model": "tiny:1b", "messages": []})
    assert not await client.is_reachable()


async def test_stale_handlers_from_replaced_socket_do_not_change_status(link) -> None:
    """A late connect/disconnect handler for an old socket must not undo the new one."""
    w1 = await _connected_worker(link)
    pm, pid, db = link["pm"], link["provider_id"], link["db"]
    old_conn = pm.worker_hub.get(pid)

    w2 = link["make_worker"]()
    link["workers"].append(w2)
    w2.start()
    await _wait_for(lambda: w2.state == "connected")
    await _wait_for(lambda: pm.worker_hub.get(pid) not in (None, old_conn))

    async def idle():
        return (await db.get_provider(pid)).status == ProviderStatus.IDLE

    await _wait_for(idle)
    await w1.stop()

    # Replay the old session's handlers after the new session is established.
    await pm._on_worker_disconnected(old_conn)
    assert (await db.get_provider(pid)).status == ProviderStatus.IDLE
    assert all(a.is_live for a in await db.get_addresses(pid))

    await db.update_provider_status(pid, ProviderStatus.BUSY)
    await pm._on_worker_connected(old_conn)
    assert (await db.get_provider(pid)).status == ProviderStatus.BUSY


async def test_disconnect_handler_still_marks_offline_when_no_newer_socket(
    link,
) -> None:
    w = await _connected_worker(link)
    pm, pid, db = link["pm"], link["provider_id"], link["db"]
    conn = pm.worker_hub.get(pid)
    await w.stop()
    await _wait_for(lambda: not pm.worker_hub.is_connected(pid))
    await db.update_provider_status(pid, ProviderStatus.IDLE)
    await pm._on_worker_disconnected(conn)
    assert (await db.get_provider(pid)).status == ProviderStatus.OFFLINE
    assert not any(a.is_live for a in await db.get_addresses(pid))


async def test_worker_connect_and_disconnect_clear_stale_status_detail(link) -> None:
    w = await _connected_worker(link)
    pm, pid = link["pm"], link["provider_id"]
    conn = pm.worker_hub.get(pid)

    # A successful (re)connect clears a reason left from an earlier failure.
    pm._status_detail[pid] = "authentication required (HTTP 401 from /api/tags)"
    await pm._on_worker_connected(conn)
    assert pm.status_detail(pid) is None

    # Losing the socket clears any "reachable but unusable" reason too.
    pm._status_detail[pid] = "stale"
    await w.stop()
    await _wait_for(lambda: not pm.worker_hub.is_connected(pid))
    await pm._on_worker_disconnected(conn)
    assert pm.status_detail(pid) is None


async def test_connect_with_failing_discovery_keeps_reachable_address_live(
    link,
) -> None:
    await _connected_worker(link)
    pm, pid, db = link["pm"], link["provider_id"], link["db"]
    conn = pm.worker_hub.get(pid)

    async def fail(provider):
        req = httpx.Request("GET", "http://worker/api/tags")
        raise httpx.HTTPStatusError(
            "401", request=req, response=httpx.Response(401, request=req)
        )

    real = pm._discover_provider
    pm._discover_provider = fail
    try:
        await pm._on_worker_connected(conn)
    finally:
        pm._discover_provider = real
    # The socket and upstream answer, so the address is live; the provider is
    # offline with the reason, matching what a health check would record.
    assert all(a.is_live for a in await db.get_addresses(pid))
    assert (await db.get_provider(pid)).status == ProviderStatus.OFFLINE
    assert "authentication required" in (pm.status_detail(pid) or "")


async def test_worker_health_check_steps_aside_and_is_bounded(
    link, monkeypatch
) -> None:
    import llama_router.provider_manager as pm_mod

    await _connected_worker(link)
    pm, pid, db = link["pm"], link["provider_id"], link["db"]
    provider = await db.get_provider(pid)

    # While a connect/disconnect handler holds the lock, a health check skips.
    probed: list[int] = []
    real_probe = pm._probe_address

    async def counting_probe(p, addr):
        probed.append(1)
        return await real_probe(p, addr)

    pm._probe_address = counting_probe
    async with pm._worker_lock(pid):
        await pm.check_provider(provider)
    assert probed == []

    # A hung probe holds the lock only until the cap, then releases it.
    monkeypatch.setattr(pm_mod, "_WORKER_CHECK_TIMEOUT", 0.2)
    pm._hot_models[pid] = [{"name": "stale"}]

    async def hung_probe(p, addr):
        await asyncio.sleep(30)
        return True

    pm._probe_address = hung_probe
    status = await asyncio.wait_for(pm.check_provider(provider), timeout=5)
    assert not pm._worker_lock(pid).locked()
    # The timeout is committed as a consistent offline result and logged.
    assert status == ProviderStatus.OFFLINE
    assert (await db.get_provider(pid)).status == ProviderStatus.OFFLINE
    logs = [e for e in await db.get_request_logs(50, 0) if e.protocol == "system"]
    assert any("timed out" in (e.error_detail or "") for e in logs)
    assert pm.get_hot_models(pid) == []

    # Back from offline, the next check rediscovers and gets the longer cap.
    caps: list[float] = []
    real_wait_for = asyncio.wait_for

    async def spy(coro, timeout):
        caps.append(timeout)
        return await real_wait_for(coro, timeout)

    monkeypatch.setattr(pm_mod.asyncio, "wait_for", spy)
    pm._probe_address = real_probe
    await pm.check_provider(await db.get_provider(pid))
    assert caps[0] == pm_mod._WORKER_MANUAL_TIMEOUT


async def test_disconnected_worker_reason_and_no_relog(link) -> None:
    pm, pid, db = link["pm"], link["provider_id"], link["db"]
    provider = await db.get_provider(pid)
    await db.update_provider_status(pid, ProviderStatus.IDLE)
    provider.status = ProviderStatus.IDLE

    # Was online, socket gone: one log entry that says so in worker terms.
    await pm.check_provider(provider)
    await pm.check_provider(provider)
    logs = [e for e in await db.get_request_logs(50, 0) if e.protocol == "system"]
    assert [e.error_detail for e in logs] == ["worker is not connected"]


async def test_live_disconnect_is_logged_once(link) -> None:
    w = await _connected_worker(link)
    pm, pid, db = link["pm"], link["provider_id"], link["db"]

    async def idle():
        return (await db.get_provider(pid)).status == ProviderStatus.IDLE

    await _wait_for(idle)
    await w.stop()

    async def offline():
        return (await db.get_provider(pid)).status == ProviderStatus.OFFLINE

    await _wait_for(offline)
    # The following health check must not log the same disconnect again.
    await pm.check_provider(await db.get_provider(pid))
    logs = [e for e in await db.get_request_logs(50, 0) if e.protocol == "system"]
    offline_logs = [e for e in logs if e.status == "error"]
    assert [(e.endpoint, e.error_detail) for e in offline_logs] == [
        ("disconnect", "worker is not connected")
    ]


async def test_disconnect_during_cycle_is_not_logged_twice(link) -> None:
    w = await _connected_worker(link)
    pm, pid, db = link["pm"], link["provider_id"], link["db"]

    async def idle():
        return (await db.get_provider(pid)).status == ProviderStatus.IDLE

    await _wait_for(idle)
    stale = await db.get_provider(pid)  # the cycle's snapshot: still idle
    await w.stop()

    async def offline():
        return (await db.get_provider(pid)).status == ProviderStatus.OFFLINE

    await _wait_for(offline)
    await pm.check_provider(stale)
    errors = [
        e
        for e in await db.get_request_logs(50, 0)
        if e.protocol == "system" and e.status == "error"
    ]
    assert [e.endpoint for e in errors] == ["disconnect"]


async def test_timeout_does_not_mark_reconnected_worker_offline(
    link, monkeypatch
) -> None:
    import llama_router.provider_manager as pm_mod

    await _connected_worker(link)
    pm, pid, db = link["pm"], link["provider_id"], link["db"]
    await db.update_provider_status(pid, ProviderStatus.IDLE)
    provider = await db.get_provider(pid)
    monkeypatch.setattr(pm_mod, "_WORKER_CHECK_TIMEOUT", 0.2)
    real_get = pm.worker_hub.get
    swapped = {"on": False}

    async def hung_probe(p, addr):
        swapped["on"] = True  # a new socket "arrives" while we hang
        await asyncio.sleep(30)
        return True

    def get(provider_id):
        return object() if swapped["on"] else real_get(provider_id)

    pm._probe_address = hung_probe
    monkeypatch.setattr(pm.worker_hub, "get", get)
    await pm.check_provider(provider)
    assert (await db.get_provider(pid)).status == ProviderStatus.IDLE

    # A manual refresh in the same situation still leaves a Log row.
    swapped["on"] = False
    monkeypatch.setattr(pm_mod, "_WORKER_MANUAL_TIMEOUT", 0.2)
    await pm.check_provider(await db.get_provider(pid), source="refresh")
    rows = [e for e in await db.get_request_logs(50, 0) if e.endpoint == "refresh"]
    assert rows
    assert "reconnected" in (rows[0].request_meta or "")


# --- decision requests through a worker ---

DECISION = {
    "model": "tiny:1b",
    "state": "The server is down and customers are calling.",
    "questions": {
        "is_urgent": {
            "type": "noul",
            "instructions": "The message conveys urgency",
        }
    },
}


async def _decide(ctx, path: str = "/v1/systemone") -> httpx.Response:
    async with httpx.AsyncClient(base_url=ctx["hub_url"], timeout=10) as c:
        return await c.post(path, json=DECISION)


async def _probed_worker(ctx) -> None:
    """Connect the worker, probe its upstream through the socket, re-route."""
    await _connected_worker(ctx)
    await ctx["pm"].probe_provider(ctx["provider_id"])
    await ctx["rt"].rebuild()


async def test_emulated_decision_through_worker(link) -> None:
    # The upstream behind the worker serves chat completions but no
    # decision route of its own, so the hub asks for logprobs.
    link["seen"]["serve_systemone"] = False
    await _probed_worker(link)
    prov = await link["db"].get_provider(link["provider_id"])
    assert prov.endpoint_state("/v1/decisions") == "missing"

    resp = await _decide(link, "/v1/decisions")
    assert resp.status_code == 200, resp.text
    answer = resp.json()["answers"]["is_urgent"]
    assert answer["type"] == "noul"
    assert answer["noul"] == pytest.approx(0.9)
    sent = link["seen"]["v1_chat_bodies"][-1]
    assert sent["logprobs"] is True
    assert sent["top_logprobs"] > 1
    assert sent["max_tokens"] == 1
    assert link["seen"]["systemone_bodies"] == []


async def test_native_decision_passes_through_worker(link) -> None:
    await _probed_worker(link)
    prov = await link["db"].get_provider(link["provider_id"])
    assert prov.endpoint_state("/v1/systemone") == "supported"

    resp = await _decide(link, "/v1/systemone")
    assert resp.status_code == 200, resp.text
    assert resp.json()["answers"]["is_urgent"] == {"type": "noul", "noul": 0.25}
    (sent,) = link["seen"]["systemone_bodies"]
    assert sent["questions"] == DECISION["questions"]
    assert link["seen"]["v1_chat_bodies"] == []


async def test_older_worker_gets_decisions_as_logprobs(link, monkeypatch) -> None:
    # A worker from before decision support refuses the routes with 403.
    decision_paths = {"/v1/decisions", "/decisions", "/v1/systemone", "/systemone"}
    old_paths = {k for k in worker_mod._INFERENCE_PATHS if k[1] not in decision_paths}
    monkeypatch.setattr(worker_mod, "_INFERENCE_PATHS", old_paths)
    await _probed_worker(link)
    pm, db, pid = link["pm"], link["db"], link["provider_id"]
    prov = await db.get_provider(pid)
    # The refusal reads as missing, not as a route behind a key.
    assert prov.endpoint_state("/v1/systemone") == "missing"

    resp = await _decide(link, "/v1/systemone")
    assert resp.status_code == 200, resp.text
    assert resp.json()["answers"]["is_urgent"]["noul"] == pytest.approx(0.9)
    assert link["seen"]["systemone_bodies"] == []

    # A probe stored before the hub knew about refusals still says the route
    # is there; the refusal sends the request to logprobs and re-probes.
    caps = dict(prov.capabilities or {})
    caps["endpoints"] = [
        {**e, "state": "needs_key"} if e["path"] == "/v1/systemone" else e
        for e in caps["endpoints"]
    ]
    await db.set_provider_capabilities(pid, caps)
    await link["rt"].rebuild()
    probes: list[str] = []
    monkeypatch.setattr(pm, "schedule_probe", lambda i, r="": probes.append(r))
    before = len(link["seen"]["v1_chat_bodies"])

    resp = await _decide(link, "/v1/systemone")
    assert resp.status_code == 200, resp.text
    assert resp.json()["answers"]["is_urgent"]["noul"] == pytest.approx(0.9)
    assert len(link["seen"]["v1_chat_bodies"]) == before + 1
    assert link["seen"]["systemone_bodies"] == []
    assert probes == ["worker refused decisions"]


async def test_worker_is_probed_when_it_connects_with_a_new_version(
    link, monkeypatch
) -> None:
    pm, pid = link["pm"], link["provider_id"]
    real_probe = pm.probe_provider
    probes: list[str] = []
    monkeypatch.setattr(pm, "schedule_probe", lambda i, r="": probes.append(r))
    handled: list[object] = []
    on_connect = pm.worker_hub._on_connect

    async def counted(conn) -> None:
        await on_connect(conn)
        handled.append(conn)

    monkeypatch.setattr(pm.worker_hub, "_on_connect", counted)
    await _connected_worker(link)
    await _wait_for(lambda: len(handled) == 1)
    assert probes == ["worker connected"]
    await real_probe(pid)

    async def reconnect() -> None:
        seen = len(handled)
        await pm.worker_hub.disconnect(pid, 1001, "restart")
        await _wait_for(lambda: len(handled) > seen, timeout=10)

    # Back as the same version: nothing new to relay.
    await reconnect()
    assert probes == ["worker connected"]

    # Back upgraded: what it relays may have changed.
    monkeypatch.setattr(worker_mod, "__version__", "99.0.0")
    await reconnect()
    assert probes == ["worker connected", "worker connected"]
