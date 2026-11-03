"""Ollama API requests served by OpenAI-style providers."""

from __future__ import annotations

import itertools
import json

import httpx
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from llama_router.api import deps as api_deps
from llama_router.api.app import app as api_app
from llama_router.background import drain
from llama_router.models import ProviderModel, ProviderProtocol, ProviderStatus
from llama_router.provider_manager import ProviderManager
from llama_router.router import Router
from llama_router.translate import (
    TranslationError,
    _StreamTranslator,
    chat_request,
    chat_response,
    generate_request,
)

pytestmark = pytest.mark.httpx_mock(
    assert_all_responses_were_requested=False,
    can_send_already_matched_responses=True,
)

UP = "http://vllm.test:8000"


def test_chat_request_translation() -> None:
    out = chat_request(
        {
            "model": "m",
            "messages": [
                {"role": "system", "content": "be brief"},
                {"role": "user", "content": "what is this?", "images": ["iVBORw0KGgo"]},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {"function": {"name": "look", "arguments": {"q": 1}}}
                    ],
                },
                {"role": "tool", "content": "a cat"},
            ],
            "tools": [{"type": "function", "function": {"name": "look"}}],
            "format": {"type": "object"},
            "options": {"temperature": 0.2, "num_predict": 64, "top_k": 40},
            "keep_alive": "5m",
        }
    )
    msgs = out["messages"]
    assert msgs[1]["content"][1] == {
        "type": "image_url",
        "image_url": {"url": "data:image/png;base64,iVBORw0KGgo"},
    }
    call = msgs[2]["tool_calls"][0]
    assert call["type"] == "function"
    assert json.loads(call["function"]["arguments"]) == {"q": 1}
    assert msgs[2]["content"] is None
    assert msgs[3]["tool_call_id"] == call["id"]
    assert out["tools"][0]["function"]["name"] == "look"
    assert out["response_format"]["type"] == "json_schema"
    assert (out["temperature"], out["max_tokens"]) == (0.2, 64)
    assert "top_k" not in out
    assert "keep_alive" not in out


def test_generate_request_translation() -> None:
    out = generate_request(
        {"model": "m", "system": "s", "prompt": "p", "format": "json"}
    )
    assert out["messages"] == [
        {"role": "system", "content": "s"},
        {"role": "user", "content": "p"},
    ]
    assert out["response_format"] == {"type": "json_object"}
    with pytest.raises(TranslationError):
        generate_request({"model": "m", "prompt": "def f(", "suffix": "return"})


def test_chat_response_translation() -> None:
    out = chat_response(
        {
            "choices": [
                {
                    "message": {
                        "content": None,
                        "tool_calls": [
                            {"function": {"name": "f", "arguments": '{"x": 2}'}}
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
            "usage": {"prompt_tokens": 9, "completion_tokens": 4},
        },
        "m",
        0.0,
    )
    assert out["message"]["tool_calls"] == [
        {"function": {"name": "f", "arguments": {"x": 2}}}
    ]
    assert (out["done"], out["done_reason"]) == (True, "stop")
    assert (out["prompt_eval_count"], out["eval_count"]) == (9, 4)
    assert out["eval_duration"] == out["total_duration"]


def test_chat_response_durations_from_llamacpp_timings() -> None:
    out = chat_response(
        {
            "choices": [{"message": {"content": "hi"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 9, "completion_tokens": 40},
            "timings": {"prompt_ms": 12.5, "predicted_ms": 500.0},
        },
        "m",
        0.0,
    )
    assert out["prompt_eval_duration"] == 12_500_000
    assert out["eval_duration"] == 500_000_000


@pytest.fixture
async def api(db, httpx_mock):
    await db.set_allow_unauthenticated(True)
    p = await db.add_provider("vllm", ProviderProtocol.OPENAI)
    await db.add_address(p.id, UP, is_preferred=True)
    await db.update_provider_status(p.id, ProviderStatus.IDLE)
    await db.set_provider_models(p.id, [ProviderModel(provider_id=p.id, name="m")])
    pm = ProviderManager(db)
    await pm._rebuild_clients(await db.get_provider(p.id))
    rt = Router(db, pm)
    api_deps.init(db, pm, rt)
    seen: list[dict] = []

    def upstream(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content or b"{}")
        seen.append({"path": request.url.path, "body": body})
        if request.url.path == "/v1/embeddings":
            n = len(body["input"]) if isinstance(body["input"], list) else 1
            return httpx.Response(
                200,
                json={
                    "data": [{"embedding": [0.1 * i, 0.2]} for i in range(n)],
                    "usage": {"prompt_tokens": 3},
                },
            )
        if body.get("stream"):
            events = [
                {"choices": [{"delta": {"content": "Hel"}}]},
                {"choices": [{"delta": {"content": "lo"}, "finish_reason": "stop"}]},
                {"choices": [], "usage": {"prompt_tokens": 5, "completion_tokens": 2}},
            ]
            raw = b"".join(b"data: " + json.dumps(e).encode() + b"\n\n" for e in events)
            return httpx.Response(
                200, stream=httpx.ByteStream(raw + b"data: [DONE]\n\n")
            )
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": "Hello"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 5, "completion_tokens": 2},
            },
        )

    httpx_mock.add_callback(upstream, is_reusable=True, is_optional=True)
    transport = httpx.ASGITransport(app=api_app)
    async with httpx.AsyncClient(transport=transport, base_url="http://router") as c:
        yield c, seen, db
    await drain()
    await pm.stop()


async def test_api_chat_non_streamed(api) -> None:
    c, seen, db = api
    r = await c.post(
        "/api/chat",
        json={
            "model": "m",
            "stream": False,
            "messages": [{"role": "user", "content": "hi"}],
        },
    )
    assert r.status_code == 200
    data = r.json()
    assert data["message"] == {"role": "assistant", "content": "Hello"}
    assert data["done"]
    assert data["eval_count"] == 2
    assert seen[-1]["path"] == "/v1/chat/completions"
    assert seen[-1]["body"]["messages"] == [{"role": "user", "content": "hi"}]
    await drain()
    entry = next(
        e for e in await db.get_request_logs(10, 0) if e.endpoint == "/api/chat"
    )
    assert (entry.prompt_tokens, entry.completion_tokens) == (5, 2)


async def test_api_chat_streamed(api) -> None:
    c, seen, _ = api
    r = await c.post(
        "/api/chat",
        json={"model": "m", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert r.status_code == 200
    lines = [json.loads(line) for line in r.text.splitlines() if line.strip()]
    text = "".join(line["message"]["content"] for line in lines if not line["done"])
    assert text == "Hello"
    final = lines[-1]
    assert final["done"]
    assert final["done_reason"] == "stop"
    assert (final["prompt_eval_count"], final["eval_count"]) == (5, 2)
    assert seen[-1]["body"]["stream_options"] == {"include_usage": True}


async def test_api_generate(api) -> None:
    c, seen, _ = api
    r = await c.post(
        "/api/generate", json={"model": "m", "prompt": "hi", "stream": False}
    )
    assert r.json()["response"] == "Hello"
    r = await c.post("/api/generate", json={"model": "m", "prompt": "hi"})
    lines = [json.loads(x) for x in r.text.splitlines() if x.strip()]
    assert "".join(x["response"] for x in lines) == "Hello"
    calls = len(seen)
    r = await c.post("/api/generate", json={"model": "m", "stream": False})
    assert r.json()["done_reason"] == "load"
    assert len(seen) == calls
    r = await c.post("/api/generate", json={"model": "m", "prompt": "a", "suffix": "b"})
    assert r.status_code == 400


async def test_api_embed_and_embeddings(api) -> None:
    c, seen, _ = api
    r = await c.post("/api/embed", json={"model": "m", "input": ["a", "b"]})
    data = r.json()
    assert len(data["embeddings"]) == 2
    assert data["prompt_eval_count"] == 3
    r = await c.post("/api/embeddings", json={"model": "m", "prompt": "a"})
    assert r.json() == {"embedding": [0.0, 0.2]}
    assert seen[-1]["path"] == "/v1/embeddings"


def test_stream_split_anywhere_and_tool_calls() -> None:
    from llama_router.translate import _StreamTranslator

    events = [
        {
            "choices": [
                {
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "function": {"name": "get", "arguments": '{"ci'},
                            }
                        ]
                    }
                }
            ]
        },
        {
            "choices": [
                {
                    "delta": {
                        "tool_calls": [
                            {"index": 0, "function": {"arguments": 'ty": "Oslo"}'}}
                        ]
                    },
                    "finish_reason": "tool_calls",
                }
            ]
        },
    ]
    raw = b"".join(b"data: " + json.dumps(e).encode() + b"\n\n" for e in events)
    tr = _StreamTranslator("m", "chat", 0.0)
    out: list[bytes] = []
    for i in range(len(raw)):
        out += tr.feed(raw[i : i + 1])
    out += tr.finish()
    lines = [json.loads(x) for x in out]
    calls = [x for x in lines if x["message"].get("tool_calls")]
    assert calls[0]["message"]["tool_calls"] == [
        {"function": {"name": "get", "arguments": {"city": "Oslo"}}}
    ]
    assert lines[-1]["done"]
    assert lines[-1]["done_reason"] == "stop"


def test_stream_final_line_carries_timings() -> None:
    from llama_router.translate import _StreamTranslator

    events = [
        {"choices": [{"delta": {"content": "hi"}, "finish_reason": "stop"}]},
        {
            "choices": [],
            "usage": {"prompt_tokens": 3, "completion_tokens": 20},
            "timings": {"prompt_ms": 5.0, "predicted_ms": 250.0},
        },
    ]
    raw = b"".join(b"data: " + json.dumps(e).encode() + b"\n\n" for e in events)
    tr = _StreamTranslator("m", "chat", 0.0)
    out = tr.feed(raw) + tr.finish()
    final = json.loads(out[-1])
    assert (final["eval_count"], final["eval_duration"]) == (20, 250_000_000)
    assert final["prompt_eval_duration"] == 5_000_000


# --- property: OpenAI -> Ollama -> OpenAI keeps what both sides can express ---

_json_value = st.recursive(
    st.none()
    | st.booleans()
    | st.integers(-(2**53), 2**53)
    | st.floats(allow_nan=False, allow_infinity=False)
    | st.text(),
    lambda inner: (
        st.lists(inner, max_size=3)
        | st.dictionaries(st.text(max_size=8), inner, max_size=3)
    ),
    max_leaves=8,
)
_arguments = st.dictionaries(st.text(max_size=8), _json_value, max_size=4)
_tool_call = st.fixed_dictionaries(
    {"name": st.text(min_size=1, max_size=16), "arguments": _arguments}
)
_reply = st.fixed_dictionaries(
    {
        "content": st.one_of(st.none(), st.text()),
        "tool_calls": st.lists(_tool_call, max_size=3),
    }
)


def _openai_message(reply: dict) -> dict:
    message: dict = {"role": "assistant", "content": reply["content"]}
    if reply["tool_calls"]:
        message["tool_calls"] = [
            {
                "id": f"call_{i}",
                "type": "function",
                "function": {
                    "name": c["name"],
                    "arguments": json.dumps(c["arguments"]),
                },
            }
            for i, c in enumerate(reply["tool_calls"])
        ]
    return message


def _comparable(message: dict) -> tuple:
    calls = [
        (c["function"]["name"], json.loads(c["function"]["arguments"]))
        for c in message.get("tool_calls") or []
    ]
    return (message.get("content") or "", calls)


@settings(max_examples=100, deadline=None)
@given(reply=_reply, model=st.text(max_size=20))
def test_reply_survives_openai_to_ollama_and_back(reply: dict, model: str) -> None:
    sent = _openai_message(reply)
    ollama = chat_response(
        {"choices": [{"message": sent, "finish_reason": "stop"}]}, model, 0.0
    )
    assert ollama["model"] == model
    # The Ollama client sends the reply back as history on its next turn.
    back = chat_request(
        {"model": model, "messages": [ollama["message"], {"role": "tool"}]}
    )
    assistant, tool = back["messages"]
    assert _comparable(assistant) == _comparable(sent)
    if sent.get("tool_calls"):
        assert tool["tool_call_id"] == assistant["tool_calls"][0]["id"]


@settings(max_examples=100, deadline=None)
@given(
    reply=_reply,
    cuts=st.lists(st.integers(min_value=0, max_value=10_000), max_size=8),
)
def test_streamed_reply_survives_any_chunking(reply: dict, cuts: list[int]) -> None:
    sent = _openai_message(reply)
    events = [
        {"choices": [{"delta": {"content": ch}}]} for ch in (sent["content"] or "")
    ]
    for i, call in enumerate(sent.get("tool_calls") or []):
        args = call["function"]["arguments"]
        half = len(args) // 2
        events.append(
            {
                "choices": [
                    {
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": i,
                                    "function": {
                                        "name": call["function"]["name"],
                                        "arguments": args[:half],
                                    },
                                }
                            ]
                        }
                    }
                ]
            }
        )
        events.append(
            {
                "choices": [
                    {
                        "delta": {
                            "tool_calls": [
                                {"index": i, "function": {"arguments": args[half:]}}
                            ]
                        }
                    }
                ]
            }
        )
    events.append({"choices": [{"delta": {}, "finish_reason": "stop"}]})
    sse = b"".join(f"data: {json.dumps(e)}\n\n".encode() for e in events)
    sse += b"data: [DONE]\n\n"
    bounds = sorted({c % (len(sse) + 1) for c in cuts} | {0, len(sse)})

    tr = _StreamTranslator("m", "chat", 0.0)
    lines: list[bytes] = []
    for start, end in itertools.pairwise(bounds):
        lines += tr.feed(sse[start:end])
    lines += tr.feed(b"\n") + tr.finish()

    parsed = [json.loads(line) for line in lines]
    assert parsed[-1]["done"] is True
    content = "".join(p["message"]["content"] for p in parsed)
    calls = [c for p in parsed for c in p["message"].get("tool_calls", [])]
    rebuilt = {
        "content": content,
        "tool_calls": [
            {
                "function": {
                    "name": c["function"]["name"],
                    "arguments": json.dumps(c["function"]["arguments"]),
                }
            }
            for c in calls
        ],
    }
    assert _comparable(rebuilt) == _comparable(sent)
