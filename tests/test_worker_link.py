"""Hub <-> worker links over a real WebSocket, with a fake Ollama upstream."""

from __future__ import annotations

import asyncio
import json
import math
import time
from typing import Any

import httpx
import pytest
import pytest_asyncio
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from llama_router.api import deps as api_deps
from llama_router.api.app import app as api_app
from llama_router.config import settings
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
from llama_router.worker_link.relay import RELAY_HEADER, relay_secret
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
        # X-Llama-Router-Relay of each tags request.
        "relay": [],
        # X-Llama-Router-Provider of each chat, and models per provider id.
        "chat_providers": [],
        "provider_models": {},
    }

    @app.get("/api/version")
    async def version(request: Request):
        seen["api_keys"].append(request.headers.get("api-key"))
        return {"version": "0.0-test"}

    @app.get("/api/tags")
    async def tags(request: Request):
        seen["via"].append(request.headers.get("x-llama-router-via"))
        seen["relay"].append(request.headers.get("x-llama-router-relay"))
        pid = request.headers.get("x-llama-router-provider")
        if pid is not None:
            names = seen["provider_models"].get(int(pid), [])
            return {"models": [{"name": n, "size": 1} for n in names]}
        return {"models": [{"name": "tiny:1b", "size": 123, "digest": "sha256:abc"}]}

    @app.get("/api/ps")
    async def ps():
        return {"models": [{"name": "tiny:1b", "size_vram": 1}]}

    @app.post("/api/chat")
    async def chat(request: Request):
        body = await request.json()
        seen["chat_bodies"].append(body)
        seen["chat_providers"].append(request.headers.get("x-llama-router-provider"))
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
    # With the secret that proves it, since the upstream is on this machine.
    assert link["seen"]["relay"][-1] == relay_secret()


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


# --- router IDs and loops ---

HUB_RID = "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa"
WORKER_RID = "bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb"


@pytest.fixture
def two_routers(monkeypatch):
    """The hub and worker share this process, so the hub gets its own ID."""
    from types import SimpleNamespace

    from llama_router import router_identity as ri
    from llama_router.worker_link import hub as hub_mod

    monkeypatch.setattr(ri, "_router_id", WORKER_RID)
    monkeypatch.setattr(ri, "_hub_id", None)
    monkeypatch.setattr(ri, "_hub_listeners", [])
    fake = SimpleNamespace(router_id=lambda: HUB_RID, parse_id=ri.parse_id)
    monkeypatch.setattr(hub_mod, "router_identity", fake)
    return fake


async def test_hub_and_worker_swap_router_ids(link, two_routers) -> None:
    from llama_router import router_identity as ri

    w = await _connected_worker(link)
    conn = link["pm"].worker_hub.get(link["provider_id"])
    assert conn.router_id == WORKER_RID
    assert conn.summary()["router_id"] == WORKER_RID
    assert w.hub_router_id == HUB_RID
    assert w.status()["hub_router_id"] == HUB_RID
    assert ri.hub_id() == HUB_RID


async def test_hub_refuses_a_worker_that_is_itself(link, two_routers) -> None:
    two_routers.router_id = lambda: WORKER_RID
    w = link["make_worker"]()
    link["workers"].append(w)
    w.start()
    await _wait_for(lambda: w.state == "rejected")
    assert "same router ID" in (w.last_error or "")
    assert w.next_retry_at - time.time() > 200
    assert not link["pm"].worker_hub.is_connected(link["provider_id"])


async def test_worker_refuses_a_hub_that_is_itself(link, monkeypatch) -> None:
    from llama_router import router_identity as ri

    # Both sides read the same ID. The hub's own check is blinded, as on a hub
    # that does not check, so the worker's check is the one that refuses.
    monkeypatch.setattr(ri, "_router_id", WORKER_RID)
    monkeypatch.setattr(ri, "_hub_id", None)
    real_parse = ri.parse_id
    calls = {"n": 0}

    def parse_once_blind(value):
        calls["n"] += 1
        # The hub reads the hello first.
        return None if calls["n"] == 1 else real_parse(value)

    monkeypatch.setattr(ri, "parse_id", parse_once_blind)
    w = link["make_worker"]()
    link["workers"].append(w)
    w.start()
    await _wait_for(lambda: w.state == "rejected")
    assert "same router ID" in (w.last_error or "")
    assert ri.hub_id() is None


async def test_remotes_from_our_hub_are_not_shared_back(link, two_routers) -> None:
    """This router serves hub WORKER_RID while alice-gpu, from that same
    router, dials in: alice-gpu must not appear in what the hub sees."""
    from llama_router import router_identity as ri
    from llama_router.model_listing import listed_models

    await _connected_worker(link)
    pm, rt, db, pid = link["pm"], link["rt"], link["db"], link["provider_id"]
    assert pm.hub_loopback_ids() == set()
    assert [m["name"] for m in await listed_models(db, rt, hub_only=True)] == [
        "tiny:1b"
    ]

    ri.set_hub_id(WORKER_RID)
    assert pm.hub_loopback_ids() == {pid}
    provider = await db.get_provider(pid)
    assert not pm.shares_with_hub(provider)
    assert await listed_models(db, rt, hub_only=True) == []
    assert [m["name"] for m in await listed_models(db, rt)] == ["tiny:1b"]

    await rt.rebuild()
    assert rt.table.ranked_for("tiny:1b", "/api/chat", "balanced", hub_only=True) == ()
    assert rt.table.ranked_for("tiny:1b", "/api/chat", "balanced")

    async with httpx.AsyncClient(base_url=link["hub_url"]) as c:
        via = {p.HUB_HEADER: p.HUB_HEADER_VALUE, RELAY_HEADER: relay_secret()}
        tags = await c.get("/api/tags", headers=via)
        ps = await c.get("/api/ps", headers=via)
    assert tags.json()["models"] == []
    assert ps.status_code == 200


# --- providers shared individually ---


async def _individual_worker(link, monkeypatch, catalog: list[dict]) -> WorkerClient:
    monkeypatch.setattr(worker_mod, "_CATALOG_INTERVAL", 0.05)
    monkeypatch.setattr(settings, "worker_catalog_min_interval_seconds", 0.0)
    link["seen"]["provider_models"].update({7: ["seven:1b"], 8: ["eight:1b"]})

    async def source() -> list[dict]:
        return [dict(e) for e in catalog]

    return await _connected_worker(link, catalog=source)


async def _children(link) -> dict[str, Any]:
    return {
        c.name: c for c in await link["db"].list_worker_children(link["provider_id"])
    }


async def test_providers_shared_individually_appear_on_the_hub(
    link, monkeypatch
) -> None:
    catalog = [
        {"id": 7, "name": "gpu", "locality": "local"},
        {"id": 8, "name": "api", "locality": "cloud"},
    ]
    w = await _individual_worker(link, monkeypatch, catalog)
    db, pm, pid = link["db"], link["pm"], link["provider_id"]
    assert w.expose == p.EXPOSE_INDIVIDUAL

    async def both_idle():
        kids = await _children(link)
        return len(kids) == 2 and all(
            c.status == ProviderStatus.IDLE for c in kids.values()
        )

    await _wait_for(both_idle)
    kids = await _children(link)
    gpu, api = kids["alice-gpu/gpu"], kids["alice-gpu/api"]
    assert (gpu.worker_remote_id, api.worker_remote_id) == (7, 8)
    assert gpu.worker_address == "worker://alice-gpu/7"
    assert (gpu.locality, api.locality) == ("local", "cloud")
    assert [m.name for m in await db.get_provider_models(gpu.id)] == ["seven:1b"]
    assert [m.name for m in await db.get_provider_models(api.id)] == ["eight:1b"]
    # The worker's own row has no models while it shares individually.
    assert await db.get_provider_models(pid) == []
    names = {m["name"] for m in await db.list_all_models()}
    assert names == {"seven:1b", "eight:1b"}
    conn = pm.worker_hub.get(pid)
    assert conn.summary()["expose"] == p.EXPOSE_INDIVIDUAL
    assert conn.summary()["shared_providers"] == 2
    assert pm.worker_hub.get(gpu.id) is conn
    assert w.status()["shared_providers"] == 2

    # A job for one provider names it to the worker's router.
    reply = await pm.get_ollama_client(gpu.id).chat(
        {"model": "seven:1b", "messages": [], "stream": False}
    )
    assert reply["message"]["content"] == "hi"
    assert link["seen"]["chat_providers"][-1] == "7"
    # Its jobs count toward the worker's socket, not a series of their own.
    before = pm.worker_hub.jobs_total(pid)
    await pm.get_ollama_client(pid).chat(
        {"model": "x", "messages": [], "stream": False}
    )
    assert link["seen"]["chat_providers"][-1] is None
    assert pm.worker_hub.jobs_total(pid) == before + 1
    assert pm.worker_hub.jobs_total(gpu.id) == 0

    # Health checks run per provider, through the same socket.
    assert await pm.check_provider(gpu, rediscover=True) == ProviderStatus.IDLE

    # One stops being shared: it goes offline, with a reason.
    catalog.pop()

    async def api_offline():
        got = await db.get_provider(api.id)
        return got.status == ProviderStatus.OFFLINE

    await _wait_for(api_offline)
    assert not pm.worker_hub.is_connected(api.id)
    # It steps aside from its name, keeping its history.
    infos = {i.provider.name: i for i in await pm.list_provider_infos()}
    assert infos["alice-gpu/api#8"].status_detail == "no longer shared by the worker"
    # A health check keeps that reason rather than blaming the socket.
    api_now = await db.get_provider(api.id)
    assert await pm.check_provider(api_now) == ProviderStatus.OFFLINE
    infos = {i.provider.name: i for i in await pm.list_provider_infos()}
    assert infos["alice-gpu/api#8"].status_detail == "no longer shared by the worker"

    # Renamed on the worker, renamed here; shared again, online again.
    catalog[0]["name"] = "big-gpu"
    catalog.append({"id": 8, "name": "api", "locality": "cloud"})

    async def renamed_and_back():
        kids = await _children(link)
        # The rename and the return can land in either order.
        back = kids.get("alice-gpu/api")
        return (
            "alice-gpu/big-gpu" in kids
            and back is not None
            and back.status == ProviderStatus.IDLE
        )

    await _wait_for(renamed_and_back)
    assert len(await _children(link)) == 2

    # The socket drops: every provider it served goes offline.
    await pm.worker_hub.disconnect(pid, p.CLOSE_REVOKED, "test")

    async def all_offline():
        kids = await _children(link)
        return all(c.status == ProviderStatus.OFFLINE for c in kids.values())

    await _wait_for(all_offline)


async def test_back_to_one_provider_takes_individual_ones_offline(
    link, monkeypatch
) -> None:
    w = await _individual_worker(
        link, monkeypatch, [{"id": 7, "name": "gpu", "locality": "local"}]
    )

    async def child_idle():
        kids = await _children(link)
        return kids and kids["alice-gpu/gpu"].status == ProviderStatus.IDLE

    await _wait_for(child_idle)
    await w.stop()
    link["workers"].remove(w)
    await _connected_worker(link)
    db, pid = link["db"], link["provider_id"]
    child = (await _children(link))["alice-gpu/gpu"]

    async def child_offline():
        got = await db.get_provider(child.id)
        return got.status == ProviderStatus.OFFLINE

    await _wait_for(child_offline)
    assert not link["pm"].worker_hub.is_connected(child.id)
    assert [m.name for m in await db.get_provider_models(pid)] == ["tiny:1b"]

    # Removing the worker removes the providers it shared.
    await link["pm"].remove_provider(pid)
    assert await db.get_provider(child.id) is None


async def test_catalog_frames_are_checked(link) -> None:
    from llama_router.worker_link.hub import WorkerConnection

    got: list[WorkerConnection] = []
    conn = WorkerConnection(
        None,  # never sends
        worker_id=1,
        name="w",
        provider_id=1,
        protocols=["ollama"],
        max_concurrency=1,
        software_version=None,
        remote_addr=None,
        expose=p.EXPOSE_INDIVIDUAL,
        on_catalog=got.append,
    )
    conn._on_text(
        {
            "type": "catalog",
            "providers": [
                {"id": 1, "name": " a ", "locality": "cloud"},
                {"id": 1, "name": "dup"},
                {"id": True, "name": "bool"},
                {"id": "2", "name": "str id"},
                {"id": 3, "name": ""},
                {"id": 4, "name": "b", "locality": "mars"},
                "junk",
            ],
        }
    )
    assert got == [conn]
    assert conn.catalog == [
        {"id": 1, "name": "a", "locality": "cloud"},
        {"id": 4, "name": "b", "locality": "local"},
    ]
    # A worker sharing as one provider has no catalog to send.
    conn.expose = p.EXPOSE_COMBINED
    conn._on_text({"type": "catalog", "providers": []})
    assert len(got) == 1


async def test_catalog_names_that_clash(link, monkeypatch) -> None:
    db, pm, pid = link["db"], link["pm"], link["provider_id"]
    # A provider on the hub already has the name the first entry wants.
    await db.add_provider("alice-gpu/gpu")
    await _individual_worker(
        link,
        monkeypatch,
        [
            {"id": 7, "name": "gpu", "locality": "local"},
            {"id": 8, "name": "other", "locality": "local"},
        ],
    )

    async def two():
        return len(await _children(link)) == 2

    await _wait_for(two)
    kids = await _children(link)
    assert set(kids) == {"alice-gpu/gpu#7", "alice-gpu/other"}

    # Two providers swap names in one catalog: both end up right.
    conn = pm.worker_hub.get(pid)
    conn.catalog = [
        {"id": 7, "name": "other", "locality": "local"},
        {"id": 8, "name": "gpu2", "locality": "local"},
    ]
    await pm._on_worker_catalog(conn)
    conn.catalog = [
        {"id": 7, "name": "gpu2", "locality": "local"},
        {"id": 8, "name": "other", "locality": "local"},
    ]
    await pm._on_worker_catalog(conn)
    by_rid = {c.worker_remote_id: c.name for c in (await _children(link)).values()}
    assert by_rid == {7: "alice-gpu/gpu2", 8: "alice-gpu/other"}


async def test_unshared_providers_give_up_their_names(link, monkeypatch) -> None:
    pm, pid = link["pm"], link["provider_id"]
    link["seen"]["provider_models"][9] = ["nine:1b"]
    await _individual_worker(
        link, monkeypatch, [{"id": 7, "name": "gpu", "locality": "local"}]
    )

    async def one():
        return "alice-gpu/gpu" in await _children(link)

    await _wait_for(one)
    conn = pm.worker_hub.get(pid)

    def by_rid(kids):
        return {c.worker_remote_id: c.name for c in kids.values()}

    # The worker replaces provider 7 with 9 under the same name.
    conn.catalog = [{"id": 9, "name": "gpu", "locality": "local"}]
    await pm._on_worker_catalog(conn)
    assert by_rid(await _children(link)) == {
        7: "alice-gpu/gpu#7",
        9: "alice-gpu/gpu",
    }
    # It stays that way on later catalogs, and 7 gets its name back when it
    # is shared again in 9's place.
    await pm._on_worker_catalog(conn)
    assert by_rid(await _children(link))[9] == "alice-gpu/gpu"
    conn.catalog = [{"id": 7, "name": "gpu", "locality": "local"}]
    await pm._on_worker_catalog(conn)
    assert by_rid(await _children(link)) == {
        7: "alice-gpu/gpu",
        9: "alice-gpu/gpu#9",
    }


async def test_overlapping_catalogs_end_on_the_latest(link, monkeypatch) -> None:
    """A newer catalog that arrives while an older one is taking providers
    offline wins: nothing it shares is left offline."""
    pm, pid = link["pm"], link["provider_id"]
    every = [
        {"id": 7, "name": "gpu", "locality": "local"},
        {"id": 8, "name": "api", "locality": "local"},
        {"id": 9, "name": "npu", "locality": "local"},
    ]
    link["seen"]["provider_models"][9] = ["nine:1b"]
    await _individual_worker(link, monkeypatch, list(every))

    async def all_idle():
        kids = await _children(link)
        return len(kids) == 3 and all(
            c.status == ProviderStatus.IDLE for c in kids.values()
        )

    await _wait_for(all_idle)
    conn = pm.worker_hub.get(pid)
    original = pm._take_child_offline
    newer: list[asyncio.Task] = []

    async def offline_with_a_newer_catalog_arriving(child, detail):
        if not newer:
            # The older sync is part way through its teardown.
            conn.catalog = list(every)
            newer.append(asyncio.create_task(pm._on_worker_catalog(conn)))
            for _ in range(50):
                await asyncio.sleep(0)
        await original(child, detail)

    monkeypatch.setattr(
        pm, "_take_child_offline", offline_with_a_newer_catalog_arriving
    )
    conn.catalog = every[:1]
    await pm._on_worker_catalog(conn)
    await newer[0]

    kids = await _children(link)
    for name in ("alice-gpu/api", "alice-gpu/npu"):
        assert pm.worker_hub.is_connected(kids[name].id), name

    async def back():
        kids = await _children(link)
        return all(c.status == ProviderStatus.IDLE for c in kids.values())

    await _wait_for(back)


# --- limits on what a hub or worker may send ---


def _bare_worker(**kw: Any) -> WorkerClient:
    return WorkerClient(
        hub_url="ws://127.0.0.1:1/worker/ws",
        token=generate_worker_token(),
        ollama_url="http://127.0.0.1:1",
        llamacpp_url=None,
        **kw,
    )


def _request_frame(req_id: int) -> dict:
    return {
        "type": "request",
        "id": req_id,
        "target": "ollama",
        "method": "POST",
        "path": "/api/chat",
        "headers": [],
    }


async def test_worker_refuses_jobs_past_four_times_its_concurrency(
    monkeypatch,
) -> None:
    w = _bare_worker(max_concurrency=1)
    replies: list[tuple[int, int]] = []

    async def reply(req_id: int, status: int, body: dict) -> None:
        replies.append((req_id, status))

    release = asyncio.Event()

    async def execute(job) -> None:
        await release.wait()

    monkeypatch.setattr(w, "_reply_json", reply)
    monkeypatch.setattr(w, "_execute", execute)
    for i in range(1, 7):
        w._on_text(_request_frame(i))
    await _wait_for(lambda: len(replies) == 2)
    assert replies == [(5, 503), (6, 503)]
    assert sorted(w._jobs) == [1, 2, 3, 4]
    # Body frames for a refused job are dropped.
    w._on_bytes(p.encode_binary(p.REQ_BODY, 5, b"x" * 10))
    assert w._body_bytes == 0
    release.set()
    await _wait_for(lambda: not w._jobs)
    await w.stop()


async def test_worker_caps_request_bodies_held_at_once(monkeypatch) -> None:
    w = _bare_worker(max_concurrency=4, max_body_bytes_in_flight=10)
    replies: list[tuple[int, int]] = []

    async def reply(req_id: int, status: int, body: dict) -> None:
        replies.append((req_id, status))

    monkeypatch.setattr(w, "_reply_json", reply)
    hold = asyncio.Event()
    real_execute = w._execute

    async def execute(job) -> None:
        if job.id == 1:
            await hold.wait()
            return
        await real_execute(job)

    monkeypatch.setattr(w, "_execute", execute)
    w._on_text(_request_frame(1))
    w._on_text(_request_frame(2))
    w._on_bytes(p.encode_binary(p.REQ_BODY, 1, b"x" * 8))
    w._on_bytes(p.encode_binary(p.REQ_BODY, 2, b"y" * 8))
    assert w._body_bytes == 8
    job2 = w._jobs[2]
    assert job2.over_budget
    assert job2.body == []
    w._on_bytes(p.encode_binary(p.REQ_END, 2))
    await _wait_for(lambda: replies == [(2, 503)])
    # Finishing a job frees its share of the budget.
    hold.set()
    w._on_bytes(p.encode_binary(p.REQ_END, 1))
    await _wait_for(lambda: not w._jobs)
    assert w._body_bytes == 0
    await w.stop()


class _SentFrames:
    """Stands in for the hub's side of a worker socket."""

    def __init__(self) -> None:
        self.text: list[dict] = []

    async def send_text(self, data: str) -> None:
        self.text.append(json.loads(data))

    async def send_bytes(self, data: bytes) -> None:
        pass


def _hub_conn(**kw: Any):
    from llama_router.worker_link.hub import WorkerConnection

    return WorkerConnection(
        _SentFrames(),
        worker_id=1,
        name="w",
        provider_id=1,
        protocols=["ollama"],
        max_concurrency=2,
        software_version=None,
        remote_addr=None,
        **kw,
    )


async def _open_response(conn, req_id: int = 1):
    """A started response on *conn*, as ``open`` leaves it."""
    from llama_router.worker_link.hub import _Pending, _WorkerStream

    await conn._slots.acquire()
    pend = _Pending(req_id)
    conn._pending[req_id] = pend
    request = httpx.Request("POST", "http://worker/api/chat")
    conn._requests[req_id] = request
    pend.started.set_result((200, []))
    return pend, _WorkerStream(conn, pend, 5.0, request).__aiter__()


async def test_hub_cuts_off_a_response_over_the_ceiling(monkeypatch) -> None:
    monkeypatch.setattr(settings, "worker_max_response_bytes", 10)
    conn = _hub_conn()
    _, stream = await _open_response(conn)
    await conn._on_bytes(p.encode_binary(p.RESP_BODY, 1, b"x" * 8))
    assert await stream.__anext__() == b"x" * 8
    await conn._on_bytes(p.encode_binary(p.RESP_BODY, 1, b"x" * 8))
    with pytest.raises(httpx.ReadError, match="over 10 bytes"):
        await stream.__anext__()
    assert conn.in_flight == 0
    assert conn._ws.text[-1] == {"type": "cancel", "id": 1}


async def test_hub_stops_reading_while_a_client_falls_behind(monkeypatch) -> None:
    from llama_router.worker_link import hub as hub_mod

    monkeypatch.setattr(settings, "worker_response_buffer_bytes", 10)
    monkeypatch.setattr(hub_mod, "_STALL_TIMEOUT", 0.3)
    conn = _hub_conn()
    pend, stream = await _open_response(conn)
    await conn._on_bytes(p.encode_binary(p.RESP_BODY, 1, b"a" * 6))
    await conn._on_bytes(p.encode_binary(p.RESP_BODY, 1, b"b" * 6))
    # Full: the next frame waits for the client to read.
    blocked = asyncio.create_task(
        conn._on_bytes(p.encode_binary(p.RESP_BODY, 1, b"c" * 6))
    )
    await asyncio.sleep(0.05)
    assert not blocked.done()
    assert await stream.__anext__() == b"a" * 6
    await asyncio.wait_for(blocked, 1)
    assert pend.buffered == 12

    # A client that stops reading altogether loses its response.
    await conn._on_bytes(p.encode_binary(p.RESP_BODY, 1, b"d" * 6))
    with pytest.raises(httpx.ReadError, match="stopped reading"):
        await stream.__anext__()
    assert conn.in_flight == 0
    assert pend.buffered == 0


async def test_catalogs_are_applied_at_a_limited_rate_latest_first(
    monkeypatch,
) -> None:
    from llama_router.worker_link.hub import WorkerHub

    monkeypatch.setattr(settings, "worker_catalog_min_interval_seconds", 0.3)
    hub = WorkerHub()
    applied: list[tuple[float, list[int]]] = []

    async def on_catalog(conn) -> None:
        applied.append((time.monotonic(), [e["id"] for e in conn.catalog]))

    hub.set_callbacks(None, None, on_catalog)
    conn = _hub_conn(expose=p.EXPOSE_INDIVIDUAL, on_catalog=hub._catalog_received)

    def send(*ids: int) -> None:
        conn._on_text(
            {"type": "catalog", "providers": [{"id": i, "name": str(i)} for i in ids]}
        )

    send(1)
    await _wait_for(lambda: len(applied) == 1)
    send(1, 2)
    send(1, 2, 3)
    send(1, 2, 3, 4)
    await _wait_for(lambda: len(applied) == 2)
    await asyncio.sleep(0.4)
    # Only the newest of the three, and not before the interval was up.
    assert [ids for _, ids in applied] == [[1], [1, 2, 3, 4]]
    assert applied[1][0] - applied[0][0] >= 0.25


async def test_catalog_is_capped(monkeypatch) -> None:
    monkeypatch.setattr(settings, "worker_max_shared_providers", 2)
    conn = _hub_conn(expose=p.EXPOSE_INDIVIDUAL)
    conn._on_text(
        {
            "type": "catalog",
            "providers": [{"id": i, "name": f"p{i}"} for i in range(1, 6)],
        }
    )
    assert [e["id"] for e in conn.catalog] == [1, 2]


async def test_unshared_providers_are_deleted_after_a_while(link, monkeypatch) -> None:
    db, pm, pid = link["db"], link["pm"], link["provider_id"]
    catalog = [
        {"id": 7, "name": "gpu", "locality": "local"},
        {"id": 8, "name": "api", "locality": "local"},
    ]
    await _individual_worker(link, monkeypatch, catalog)

    async def two():
        return len(await _children(link)) == 2

    await _wait_for(two)
    conn = pm.worker_hub.get(pid)
    # About a third of a second.
    monkeypatch.setattr(settings, "worker_unshared_purge_hours", 0.0001)
    conn.catalog = [catalog[0]]
    await pm._on_worker_catalog(conn)
    kids = await _children(link)
    assert set(kids) == {"alice-gpu/gpu", "alice-gpu/api#8"}
    api = kids["alice-gpu/api#8"]

    async def purged():
        return await db.get_provider(api.id) is None

    # The purge timer syncs again once the grace period is over.
    await _wait_for(purged)
    assert set(await _children(link)) == {"alice-gpu/gpu"}


async def test_unshared_providers_beyond_the_cap_go_at_once(link, monkeypatch) -> None:
    pm, pid = link["pm"], link["provider_id"]
    link["seen"]["provider_models"][9] = ["nine:1b"]
    catalog = [
        {"id": 7, "name": "gpu", "locality": "local"},
        {"id": 8, "name": "api", "locality": "local"},
    ]
    await _individual_worker(link, monkeypatch, catalog)

    async def two():
        return len(await _children(link)) == 2

    await _wait_for(two)
    monkeypatch.setattr(settings, "worker_max_shared_providers", 2)
    conn = pm.worker_hub.get(pid)
    conn.catalog = [catalog[0], {"id": 9, "name": "npu", "locality": "local"}]
    await pm._on_worker_catalog(conn)
    by_rid = {c.worker_remote_id for c in (await _children(link)).values()}
    assert by_rid == {7, 9}
