"""OllamaClient, OpenAIClient and V1Client against a mock transport."""

from __future__ import annotations

import json

import httpx
import pytest

from llama_router.llamacpp_client import OpenAIClient, _chat_metrics
from llama_router.ollama_client import OllamaClient, _human_bytes
from llama_router.v1_client import V1Client


class Recorder:
    """Transport handler: canned responses by path, remembers requests."""

    def __init__(self, routes: dict[str, httpx.Response | Exception]) -> None:
        self.routes = routes
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        out = self.routes.get(request.url.path, httpx.Response(404))
        if isinstance(out, Exception):
            raise out
        return out

    @property
    def transport(self) -> httpx.MockTransport:
        return httpx.MockTransport(self)


def _json(data, status=200) -> httpx.Response:
    return httpx.Response(status, json=data)


# --- Ollama ---------------------------------------------------------------


async def test_ollama_discovery_endpoints():
    rec = Recorder(
        {
            "/api/version": _json({"version": "0.5"}),
            "/api/tags": _json(
                {"models": [{"name": "a:latest", "size": 5, "digest": "d"}]}
            ),
            "/api/ps": _json({"models": [{"name": "a"}]}),
            "/api/show": _json({"details": {"family": "x"}}),
        }
    )
    c = OllamaClient("http://o:11434/", transport=rec.transport)
    assert await c.is_reachable()
    assert await c.get_version() == {"version": "0.5"}
    (m,) = await c.get_tags()
    assert (m.name, m.size, m.digest) == ("a:latest", 5, "d")
    assert await c.get_ps() == [{"name": "a"}]
    assert (await c.show("a"))["details"] == {"family": "x"}
    assert json.loads(rec.requests[-1].content) == {"model": "a"}
    await c.close()


async def test_ollama_unreachable_and_http_errors_logged(caplog):
    rec = Recorder(
        {"/api/version": httpx.ConnectError("no"), "/api/tags": _json({}, 500)}
    )
    c = OllamaClient("http://o:11434", transport=rec.transport)
    assert not await c.is_reachable()
    assert await c.get_version() is None
    with pytest.raises(httpx.HTTPStatusError):
        await c.get_tags()
    rec.routes["/api/ps"] = httpx.ConnectError("boom")
    with pytest.raises(httpx.ConnectError):
        await c.get_ps()
    assert "http://o:11434/api/ps" in caplog.text
    await c.close()


async def test_ollama_non_stream_calls_force_stream_flag():
    ok = _json({"ok": True})
    rec = Recorder(
        dict.fromkeys(
            (
                "/api/chat",
                "/api/generate",
                "/api/embeddings",
                "/api/embed",
                "/v1/chat/completions",
                "/v1/completions",
                "/v1/embeddings",
                "/v1/responses",
                "/v1/images/generations",
                "/v1/audio/voices",
            ),
            ok,
        )
    )
    c = OllamaClient("http://o", transport=rec.transport)
    for call in (
        c.chat,
        c.generate,
        c.v1_chat_completions,
        c.v1_completions,
        c.v1_responses,
    ):
        body = {"stream": True}
        assert await call(body) == {"ok": True}
        assert body["stream"] is False
    assert await c.embeddings({}) == {"ok": True}
    assert await c.embed({}) == {"ok": True}
    assert await c.v1_embeddings({}) == {"ok": True}
    assert await c.v1_images_generations({}) == {"ok": True}
    assert await c.v1_audio_voices() == {"ok": True}
    await c.close()


async def test_ollama_streams_and_binary_proxies():
    rec = Recorder(
        {
            "/api/chat": httpx.Response(200, content=b"a\nb\n"),
            "/api/generate": httpx.Response(200, content=b"g"),
            "/v1/chat/completions": httpx.Response(200, content=b"c"),
            "/v1/completions": httpx.Response(200, content=b"d"),
            "/v1/responses": httpx.Response(200, content=b"r"),
            "/v1/audio/speech": httpx.Response(200, content=b"mp3"),
            "/v1/images/edits": httpx.Response(200, content=b"img"),
            "/v1/audio/transcriptions": _json({"text": "hi"}),
        }
    )
    c = OllamaClient("http://o", transport=rec.transport)

    async def collect(gen):
        return b"".join([x async for x in gen])

    assert await collect(c.chat_stream({})) == b"a\nb\n"
    assert json.loads(rec.requests[-1].content)["stream"] is True
    assert await collect(c.generate_stream({})) == b"g"
    assert await collect(c.v1_chat_completions_stream({})) == b"c"
    assert await collect(c.v1_completions_stream({})) == b"d"
    assert await collect(c.v1_responses_stream({})) == b"r"
    assert await collect(c.v1_audio_speech({})) == b"mp3"
    resp = await c.v1_images_edits(b"data", "multipart/form-data; boundary=x")
    assert resp.content == b"img"
    assert rec.requests[-1].headers["content-type"].startswith("multipart")
    assert (await c.v1_audio_transcriptions(b"d", "audio/wav")).json() == {"text": "hi"}
    await c.close()


@pytest.mark.parametrize(
    "method",
    [
        "chat_stream",
        "generate_stream",
        "v1_chat_completions_stream",
        "v1_completions_stream",
        "v1_responses_stream",
        "v1_audio_speech",
    ],
)
async def test_ollama_stream_errors_raise(method):
    rec = Recorder({})  # every path 404s
    c = OllamaClient("http://o", transport=rec.transport)
    with pytest.raises(httpx.HTTPStatusError):
        async for _ in getattr(c, method)({}):
            pass
    await c.close()


async def test_ollama_pull_rewrites_for_cache_and_reports_progress():
    lines = [
        {"status": "pulling manifest"},
        {"status": "downloading", "total": 100, "completed": 50},
        {"status": "downloading", "total": 100, "completed": 100},
        {"status": "success"},
    ]
    payload = b"".join(json.dumps(x).encode() + b"\n" for x in lines) + b"junk\n\n"
    rec = Recorder({"/api/pull": httpx.Response(200, content=payload)})
    c = OllamaClient("http://o", transport=rec.transport)
    seen: list[dict] = []
    final = await c.pull_model("llama3", "http://cache:9000/", seen.append)
    assert final == {"status": "success"}
    sent = json.loads(rec.requests[0].content)
    assert sent == {
        "model": "cache:9000/library/llama3",
        "stream": True,
        "insecure": True,
    }
    assert [s.get("percent") for s in seen] == [None, 50, 100, None]

    await c.pull_model("org/m@sha256:x", "http://cache:9000")
    assert json.loads(rec.requests[1].content)["model"] == "cache:9000/org/m"
    await c.pull_model("plain")
    assert json.loads(rec.requests[2].content) == {
        "model": "plain",
        "stream": True,
        "insecure": False,
    }
    await c.close()


async def test_ollama_pull_error_in_stream_and_http_failure():
    rec = Recorder(
        {"/api/pull": httpx.Response(200, content=b'{"error": "no such model"}\n')}
    )
    c = OllamaClient("http://o", transport=rec.transport)
    with pytest.raises(RuntimeError, match="no such model"):
        await c.pull_model("x")
    rec.routes["/api/pull"] = httpx.Response(500)
    with pytest.raises(httpx.HTTPStatusError):
        await c.pull_model("x")
    await c.close()


async def test_ollama_blob_create_delete(tmp_path):
    f = tmp_path / "w.gguf"
    f.write_bytes(b"z" * 100)
    rec = Recorder(
        {
            "/api/blobs/sha256:abc": httpx.Response(200),
            "/api/create": _json({"status": "success"}),
            "/api/delete": httpx.Response(200),
        }
    )
    c = OllamaClient("http://o", transport=rec.transport)
    assert await c.has_blob("sha256:abc")
    progress: list[tuple[int, int]] = []
    await c.push_blob("sha256:abc", str(f), lambda s, t: progress.append((s, t)))
    assert progress == [(100, 100)]
    assert rec.requests[-1].content == b"z" * 100
    assert await c.create_from_files("m", {"w.gguf": "sha256:abc"}) == {
        "status": "success"
    }
    await c.delete_model("m")
    assert rec.requests[-1].method == "DELETE"
    rec.routes["/api/create"] = httpx.Response(200)
    assert await c.create_from_files("m", {}) == {}
    await c.close()


async def test_ollama_benchmarks():
    rec = Recorder(
        {
            "/api/chat": _json(
                {
                    "load_duration": 2_000_000_000,
                    "eval_count": 50,
                    "eval_duration": 1_000_000_000,
                }
            ),
            "/api/embed": _json(
                {
                    "load_duration": 1_000_000_000,
                    "prompt_eval_count": 10,
                    "total_duration": 2_000_000_000,
                }
            ),
        }
    )
    c = OllamaClient("http://o", transport=rec.transport)
    assert await c.benchmark_chat("m", "hi") == {
        "startup_time_ms": 2000.0,
        "tokens_per_second": 50.0,
    }
    assert await c.benchmark_embed("m", "hi") == {
        "startup_time_ms": 1000.0,
        "tokens_per_second": 10.0,
    }
    rec.routes["/api/embed"] = _json({})
    assert (await c.benchmark_embed("m", "hi"))["tokens_per_second"] == 0
    await c.close()


def test_human_bytes():
    assert _human_bytes(1 << 10) == "1 KB"
    assert _human_bytes(1 << 20) == "1.0 MB"
    assert _human_bytes(1 << 30) == "1.0 GB"


# --- OpenAI-style ----------------------------------------------------------


def test_chat_metrics_prefers_server_timings():
    timings = {"predicted_per_second": 42, "prompt_ms": 100, "predicted_ms": 900}
    cold = _chat_metrics({"timings": timings}, 5.0)
    assert cold == {"startup_time_ms": 4000.0, "tokens_per_second": 42.0}
    warm = _chat_metrics({"timings": timings}, 1.5)
    assert warm["startup_time_ms"] is None
    fallback = _chat_metrics({"usage": {"completion_tokens": 20}}, 2.0)
    assert fallback == {"startup_time_ms": None, "tokens_per_second": 10.0}
    assert _chat_metrics({}, 0)["tokens_per_second"] == 0


async def test_openai_reachability_and_models():
    rec = Recorder(
        {
            "/health": httpx.Response(503),
            "/v1/models": _json(
                {
                    "data": [
                        {
                            "id": "m1",
                            "pricing": {"prompt": "1"},
                            "owned_by": "x",
                            "meta": {"n_ctx": 4},
                        },
                        {"id": ""},
                        "junk",
                    ]
                }
            ),
            "/props": _json({"model_path": "x"}),
        }
    )
    c = OpenAIClient("http://l:8080", transport=rec.transport)
    assert await c.is_reachable()  # falls back to /v1/models
    (m,) = await c.get_models()
    assert m.name == "m1"
    assert m.details == {"n_ctx": 4, "pricing": {"prompt": "1"}, "owned_by": "x"}
    assert await c.get_props() == {"model_path": "x"}
    rec.routes["/props"] = httpx.Response(500)
    assert await c.get_props() is None
    rec.routes["/health"] = httpx.ConnectError("x")
    assert not await c.is_reachable()
    await c.close()


async def test_openai_model_list_shapes():
    rec = Recorder({"/v1/models": _json([{"id": "bare"}])})
    c = OpenAIClient("http://l", transport=rec.transport)
    assert [m.name for m in await c.get_models()] == ["bare"]
    rec.routes["/v1/models"] = _json({"models": [{"name": "ts"}, "x"]})
    assert [m.name for m in await c.get_models()] == ["ts"]
    await c.close()


async def test_openai_api_base_prefix_is_mapped():
    rec = Recorder({"/openai/v1/chat/completions": _json({"ok": 1})})
    c = OpenAIClient("http://cloud/openai/v1", transport=rec.transport)
    assert await c.chat_completions({}) == {"ok": 1}
    assert c._upstream_url("/v1/embeddings") == "http://cloud/openai/v1/embeddings"
    assert c._upstream_url("health") == "http://cloud/health"
    await c.close()


async def test_openai_calls_and_streams():
    ok = _json({"ok": True})
    rec = Recorder(
        {
            "/v1/chat/completions": ok,
            "/v1/completions": ok,
            "/v1/responses": ok,
            "/v1/embeddings": ok,
            "/v1/images/generations": ok,
            "/v1/audio/voices": ok,
            "/v1/audio/speech": httpx.Response(200, content=b"wav"),
            "/v1/images/edits": httpx.Response(200, content=b"img"),
            "/v1/audio/transcriptions": ok,
        }
    )
    c = OpenAIClient("http://l", transport=rec.transport)
    for call in (c.chat_completions, c.completions, c.responses):
        body = {"stream": True}
        assert await call(body) == {"ok": True}
        assert body["stream"] is False
    assert await c.embeddings({}) == {"ok": True}
    assert await c.images_generations({}) == {"ok": True}
    assert await c.audio_voices() == {"ok": True}
    assert (await c.images_edits(b"d", "multipart/form-data")).content == b"img"
    assert (await c.audio_transcriptions(b"d", "multipart/form-data")).json() == {
        "ok": True
    }
    assert b"".join([x async for x in c.audio_speech({})]) == b"wav"
    for stream in (c.chat_completions_stream, c.completions_stream, c.responses_stream):
        body = {}
        chunks = [x async for x in stream(body)]
        assert body["stream"] is True and chunks
    await c.close()


@pytest.mark.parametrize(
    "method",
    [
        "chat_completions_stream",
        "completions_stream",
        "responses_stream",
        "audio_speech",
    ],
)
async def test_openai_stream_errors_raise(method):
    c = OpenAIClient("http://l", transport=Recorder({}).transport)
    with pytest.raises(httpx.HTTPStatusError):
        async for _ in getattr(c, method)({}):
            pass
    await c.close()


async def test_openai_benchmarks():
    rec = Recorder(
        {
            "/v1/chat/completions": _json({"usage": {"completion_tokens": 10}}),
            "/v1/embeddings": _json({"usage": {"prompt_tokens": 8}}),
        }
    )
    c = OpenAIClient("http://l", transport=rec.transport)
    chat = await c.benchmark_chat("m", "hi")
    assert chat["tokens_per_second"] > 0 and chat["startup_time_ms"] is None
    emb = await c.benchmark_embed("m", "hi")
    assert emb["tokens_per_second"] > 0
    await c.close()


# --- V1Client dispatch -------------------------------------------------------


async def test_v1client_dispatches_to_openai_client_first():
    rec = Recorder({"/v1/chat/completions": _json({"via": "lcpp"})})
    lc = OpenAIClient("http://l", transport=rec.transport)
    oc = OllamaClient("http://o", transport=Recorder({}).transport)
    v1 = V1Client(ollama=oc, llamacpp=lc)
    assert await v1.chat_completions({}) == {"via": "lcpp"}
    assert await v1.post_json("/v1/chat/completions", {}, op="t") == {"via": "lcpp"}
    await lc.close()
    await oc.close()


async def test_v1client_falls_back_to_ollama():
    ok = _json({"via": "ollama"})
    rec = Recorder(
        {
            "/v1/chat/completions": ok,
            "/v1/completions": ok,
            "/v1/embeddings": ok,
            "/v1/responses": ok,
            "/v1/images/generations": ok,
            "/v1/audio/voices": ok,
            "/v1/audio/speech": httpx.Response(200, content=b"s"),
            "/v1/images/edits": httpx.Response(200, content=b"i"),
            "/v1/audio/transcriptions": ok,
        }
    )
    oc = OllamaClient("http://o", transport=rec.transport)
    v1 = V1Client(ollama=oc)
    want = {"via": "ollama"}
    assert await v1.chat_completions({}) == want
    assert await v1.completions({}) == want
    assert await v1.embeddings({}) == want
    assert await v1.responses({}) == want
    assert await v1.images_generations({}) == want
    assert await v1.audio_voices() == want
    assert (await v1.images_edits(b"", "x/y")).content == b"i"
    assert (await v1.audio_transcriptions(b"", "x/y")).json() == want
    for stream in (
        v1.chat_completions_stream,
        v1.completions_stream,
        v1.responses_stream,
        v1.audio_speech,
    ):
        assert [x async for x in stream({})]
    await oc.close()


async def test_v1client_llamacpp_paths():
    ok = _json({"via": "lcpp"})
    rec = Recorder(
        dict.fromkeys(
            (
                "/v1/completions",
                "/v1/embeddings",
                "/v1/responses",
                "/v1/images/generations",
                "/v1/audio/voices",
                "/v1/audio/transcriptions",
            ),
            ok,
        )
        | {
            "/v1/chat/completions": httpx.Response(200, content=b"s"),
            "/v1/audio/speech": httpx.Response(200, content=b"s"),
            "/v1/images/edits": httpx.Response(200, content=b"i"),
        }
    )
    lc = OpenAIClient("http://l", transport=rec.transport)
    v1 = V1Client(llamacpp=lc)
    want = {"via": "lcpp"}
    assert await v1.completions({}) == want
    assert await v1.embeddings({}) == want
    assert await v1.responses({}) == want
    assert await v1.images_generations({}) == want
    assert await v1.audio_voices() == want
    assert (await v1.images_edits(b"", "x/y")).content == b"i"
    assert (await v1.audio_transcriptions(b"", "x/y")).json() == want
    for stream in (
        v1.chat_completions_stream,
        v1.completions_stream,
        v1.responses_stream,
        v1.audio_speech,
    ):
        assert [x async for x in stream({})]
    await lc.close()
