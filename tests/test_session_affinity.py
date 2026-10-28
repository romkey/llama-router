from __future__ import annotations

import json

import httpx
import pytest
from starlette.requests import Request

from llama_router.affinity import (
    PROMPT_PREFIX_CHARS,
    SessionAffinity,
    session_key,
)
from llama_router.auth import key_hash, routing_preferences_from_request
from llama_router.database import Database
from llama_router.models import ProviderModel, ProviderProtocol, ProviderStatus
from llama_router.provider_manager import ProviderManager
from llama_router.router import Router, RoutingPreferences
from llama_router.routing_stats import ProviderStats

CHAT = "/v1/chat/completions"
SYSTEM = "You are a coding agent. " * 2000  # ~48 KB, longer than any prefix
MODEL = "qwen-coder"


class FakeClock:
    def __init__(self) -> None:
        self.t = 1000.0

    def __call__(self) -> float:
        return self.t


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock()


@pytest.fixture
def pm(db: Database, clock: FakeClock) -> ProviderManager:
    manager = ProviderManager(db)
    manager.stats = ProviderStats(clock=clock)
    return manager


def _router(db, pm, clock, **kw) -> Router:
    return Router(
        db,
        pm,
        affinity=SessionAffinity(
            kw.pop("ttl_s", 1800), kw.pop("max_sessions", 1000), clock
        ),
        session_affinity=kw.pop("session_affinity", True),
    )


async def _provider(db: Database, name: str, models: list[str]):
    p = await db.add_provider(name, ProviderProtocol.OLLAMA)
    await db.update_provider_status(p.id, ProviderStatus.IDLE)
    await db.set_provider_models(
        p.id, [ProviderModel(provider_id=p.id, name=m) for m in models]
    )
    return p


def _chat(first_user: str = "fix the bug in main.py", turns: int = 0, **extra):
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": first_user},
    ]
    for i in range(turns):
        messages.append({"role": "assistant", "content": f"answer {i}"})
        messages.append({"role": "user", "content": f"follow-up {i}"})
    return {"model": MODEL, "messages": messages, **extra}


# --- Key derivation ---


def test_same_conversation_keeps_its_key_as_it_grows():
    assert session_key({}, _chat()) == session_key({}, _chat(turns=12))


def test_different_conversations_get_different_keys():
    # The system prompt is far longer than any prefix; the first user message
    # still tells sessions of the same client apart.
    assert session_key({}, _chat("task one")) != session_key({}, _chat("task two"))


def test_key_is_scoped_by_api_key_and_model():
    body = _chat()
    other_model = {**body, "model": "other"}
    assert session_key({}, body, 1) != session_key({}, body, 2)
    assert session_key({}, body) != session_key({}, other_model)


def test_explicit_ids_take_precedence_in_order():
    body = _chat(prompt_cache_key="pck", user="alice")
    by_header = session_key({"x-session-id": "s1"}, body)
    assert by_header == session_key({"x-session-id": "s1"}, _chat("other"))
    assert by_header != session_key({}, body)
    # prompt_cache_key beats user
    assert session_key({}, body) == session_key({}, _chat("x", prompt_cache_key="pck"))
    assert session_key({}, body) != session_key({}, _chat(user="alice"))
    # user beats the fingerprint
    assert session_key({}, _chat("a", user="bob")) == session_key(
        {}, _chat("b", user="bob")
    )


def test_content_parts_are_read():
    parts = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": [{"type": "text", "text": "sys"}]},
            {
                "role": "user",
                "content": [{"type": "text", "text": "hi"}, {"type": "image_url"}],
            },
        ],
    }
    plain = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "hi"},
        ],
    }
    assert session_key({}, parts) is not None
    assert session_key({}, parts) == session_key({}, plain)


def test_responses_and_prompt_shapes():
    r1 = {
        "model": MODEL,
        "instructions": "be terse",
        "input": [{"role": "user", "content": "go"}],
    }
    r2 = {**r1, "input": r1["input"] + [{"role": "assistant", "content": "ok"}]}
    assert session_key({}, r1) == session_key({}, r2) is not None

    long_prompt = "x" * PROMPT_PREFIX_CHARS
    a = session_key({}, {"model": MODEL, "prompt": long_prompt})
    b = session_key({}, {"model": MODEL, "prompt": long_prompt + " more text"})
    assert a == b is not None
    # Short prompts (autocomplete, one-off calls) get no session.
    assert session_key({}, {"model": MODEL, "prompt": "def f():"}) is None


def test_nothing_stable_means_no_key():
    assert session_key({}, {"model": MODEL}) is None
    assert session_key({}, {"model": MODEL, "messages": []}) is None
    assert session_key({}, {"messages": [{"role": "user", "content": "x"}]}) is None


# --- Store ---


def test_store_expires_and_caps(clock):
    store = SessionAffinity(ttl_s=60, max_sessions=2, clock=clock)
    store.bind("a", 1)
    store.bind("b", 2)
    store.bind("c", 3)
    assert store.get("a") is None  # evicted, least recently used
    assert store.get("c") == 3
    clock.t += 61
    assert store.peek("c") is None
    assert store.get("c") is None
    store.bind("d", 4)
    assert len(store) == 1  # "b" was swept as expired


# --- Routing ---


async def _two_equal(db, pm, clock):
    a = await _provider(db, "a", [MODEL])
    b = await _provider(db, "b", [MODEL])
    return a, b, _router(db, pm, clock)


async def test_session_sticks_even_when_its_provider_is_busier(db, pm, clock):
    a, b, rt = await _two_equal(db, pm, clock)
    session = rt.session_for({}, _chat(), None)
    first = await rt.route(MODEL, CHAT, session=session)
    assert first is not None
    bound = first.provider.id
    other = b.id if bound == a.id else a.id
    pm._active_requests[bound] = 5
    pm._active_requests[other] = 0
    for _ in range(5):
        r = await rt.route(MODEL, CHAT, session=session)
        assert r.provider.id == bound
    # Without a session the idle provider wins, as before.
    assert (await rt.route(MODEL, CHAT)).provider.id == other
    stats = rt.affinity.stats()
    assert stats["hits"] == 5
    assert stats["misses"] == 1
    assert stats["sessions"] == 1


async def test_session_moves_when_its_provider_goes_offline(db, pm, clock):
    a, b, rt = await _two_equal(db, pm, clock)
    session = rt.session_for({}, _chat(), None)
    bound = (await rt.route(MODEL, CHAT, session=session)).provider.id
    other = b.id if bound == a.id else a.id

    await db.update_provider_status(bound, ProviderStatus.OFFLINE)
    r = await rt.route(MODEL, CHAT, session=session)
    assert r.provider.id == other
    assert rt.affinity.rebinds == 1

    # Coming back does not pull the session back.
    await db.update_provider_status(bound, ProviderStatus.IDLE)
    pm._active_requests[other] = 3
    assert (await rt.route(MODEL, CHAT, session=session)).provider.id == other


async def test_pin_and_chaos_override_affinity(db, pm, clock):
    a, b, rt = await _two_equal(db, pm, clock)
    session = rt.session_for({}, _chat(), None)
    bound = (await rt.route(MODEL, CHAT, session=session)).provider.id
    other = b.id if bound == a.id else a.id

    pinned = RoutingPreferences(pinned_providers={MODEL: other})
    assert (await rt.route(MODEL, CHAT, pinned, session=session)).provider.id == other
    # The pin did not move the binding.
    assert rt.affinity.peek(session) == bound

    assert rt.session_for({}, _chat(), RoutingPreferences(mode="chaos")) is None


async def test_per_key_override_beats_router_default(db, pm, clock):
    await _provider(db, "a", [MODEL])
    rt_on = _router(db, pm, clock, session_affinity=True)
    off = RoutingPreferences(session_affinity=False)
    assert rt_on.session_for({}, _chat(), off) is None
    # Even a session passed in by hand is ignored when affinity is off.
    await rt_on.route(MODEL, CHAT, off, session="0:x:y")
    assert len(rt_on.affinity) == 0

    rt_off = _router(db, pm, clock, session_affinity=False)
    assert rt_off.session_for({}, _chat(), None) is None
    on = RoutingPreferences(session_affinity=True, key_id=7)
    session = rt_off.session_for({}, _chat(), on)
    assert session is not None
    assert session.startswith("7:")
    await rt_off.route(MODEL, CHAT, on, session=session)
    assert len(rt_off.affinity) == 1


async def test_explain_reports_binding_without_creating_one(db, pm, clock):
    a, b, rt = await _two_equal(db, pm, clock)
    session = rt.session_for({}, _chat(), None)

    out = await rt.explain(MODEL, CHAT, session=session)
    assert out["session"]["bound_provider_id"] is None
    assert len(rt.affinity) == 0

    bound = (await rt.route(MODEL, CHAT, session=session)).provider.id
    other = b.id if bound == a.id else a.id
    pm._active_requests[bound] = 4
    out = await rt.explain(MODEL, CHAT, session=session)
    assert out["session"]["honored"] is True
    assert out["chosen"]["provider_id"] == bound
    assert "for this session" in out["summary"]
    rows = {c["provider_id"]: c for c in out["steps"][0]["candidates"]}
    assert rows[bound]["session_bound"]
    assert not rows[other]["session_bound"]
    assert rt.affinity.hits == 0  # explaining is not routing


async def test_removed_provider_bindings_are_dropped(db, pm, clock):
    _a, _b, rt = await _two_equal(db, pm, clock)
    session = rt.session_for({}, _chat(), None)
    bound = (await rt.route(MODEL, CHAT, session=session)).provider.id
    await db.remove_provider(bound)
    await rt.rebuild()
    assert rt.affinity.peek(session) is None


# --- API key flag ---


def _key_request(key: str) -> Request:
    return Request(
        {
            "type": "http",
            "method": "POST",
            "path": "/api/chat",
            "headers": [(b"api-key", key.encode())],
        }
    )


async def test_api_key_carries_affinity_flag_and_id(db: Database):
    await db.set_allow_unauthenticated(False)
    for flag in (None, True, False):
        key = f"lrk_affinity-{flag}"
        key_id = await db.create_api_key(
            key[:10], key_hash(key), "balanced", True, flag
        )
        prefs = await routing_preferences_from_request(db, _key_request(key))
        assert prefs.session_affinity is flag
        assert prefs.key_id == key_id
    listed = {k["id"]: k["session_affinity"] for k in await db.list_api_keys()}
    assert sorted(listed.values(), key=str) == sorted([None, True, False], key=str)


# --- Handler ---


def _json_request(path: str, payload: dict) -> Request:
    raw = json.dumps(payload).encode()

    async def receive():
        return {"type": "http.request", "body": raw, "more_body": False}

    return Request(
        {
            "type": "http",
            "method": "POST",
            "scheme": "http",
            "server": ("test", 8080),
            "path": path,
            "root_path": "",
            "query_string": b"",
            "headers": [(b"content-type", b"application/json")],
        },
        receive,
    )


async def test_upstream_failure_forgets_the_session(db, pm, clock, monkeypatch):
    from llama_router.llamacpp_api import chat as chat_api
    from llama_router.llamacpp_api import deps

    await _provider(db, "a", [MODEL])
    await db.set_allow_unauthenticated(True)
    rt = _router(db, pm, clock)
    deps.init(db, pm, rt)

    class Down:
        async def chat_completions(self, body):
            raise httpx.ConnectError(
                "refused", request=httpx.Request("POST", "http://a/v1/chat/completions")
            )

    monkeypatch.setattr(chat_api, "get_v1_client", lambda pm, pid: Down())
    monkeypatch.setattr(chat_api, "log_request_background", lambda *a, **k: None)

    body = _chat()
    session = rt.session_for({}, body, None)
    with pytest.raises(httpx.ConnectError):
        await chat_api.chat_completions(_json_request(CHAT, body))
    assert rt.affinity.peek(session) is None
    assert rt.affinity.misses == 1  # it was bound, then forgotten
