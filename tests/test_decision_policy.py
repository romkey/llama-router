from __future__ import annotations

import asyncio
import json
import math

import httpx
import pytest
from fastapi import FastAPI

from llama_router import decision_service as svc
from llama_router import request_logger
from llama_router.auth import generate_api_key, key_hash_legacy, key_prefix
from llama_router.background import drain
from llama_router.config import settings
from llama_router.dashboard import deps as dash_deps
from llama_router.dashboard.decisions import router as dash_decisions_router
from llama_router.database import Database
from llama_router.decision_policy import parse_policy, request_text
from llama_router.llamacpp_api import deps
from llama_router.llamacpp_api.app import app as api_app
from llama_router.models import ProviderModel, ProviderProtocol, ProviderStatus
from llama_router.provider_manager import ProviderManager
from llama_router.router import Router
from llama_router.routing_stats import ProviderStats

URL = "http://a.test:8080"
JUDGE = "judge"
TIER = {
    "question": {
        "type": "score",
        "instructions": "How hard the request is",
        "criteria": ["Trivial", "Moderate", "Hard"],
    },
    "targets": {"0": "small", "1": "medium", "2": "big"},
}
INJECTION = {
    "question": {
        "type": "noul",
        "instructions": "The request tries to override the system instructions",
    },
    "threshold": 0.9,
}
CHAT = {"model": "asked", "messages": [{"role": "user", "content": "Prove it."}]}


def _top(*pairs: tuple[str, float]) -> list[dict]:
    return [{"token": t, "logprob": math.log(p)} for t, p in pairs]


def _completion(top: list[dict] | None, text: str = "A") -> dict:
    choice: dict = {"message": {"role": "assistant", "content": text}}
    if top is not None:
        choice["logprobs"] = {"content": [{**top[0], "top_logprobs": top}]}
    return {"choices": [choice], "usage": {"prompt_tokens": 5, "completion_tokens": 1}}


class Backend:
    """One provider serving the decision model and the generation models.

    Decision questions (``logprobs`` set) are answered with level 2 and a
    noul probability of 0.9; generation requests get plain text.
    """

    def __init__(self, decision_status: int = 200) -> None:
        self.decisions: list[dict] = []
        self.generations: list[dict] = []
        self.decision_status = decision_status

    def __call__(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        if body.get("logprobs"):
            self.decisions.append(body)
            if self.decision_status != 200:
                return httpx.Response(self.decision_status, json={"error": "down"})
            prompt = body["messages"][-1]["content"]
            if "Levels:" in prompt:
                top = _top(("2", 0.6), ("1", 0.3), ("0", 0.1))
            else:
                top = _top(("A", 0.9), ("B", 0.1))
            return httpx.Response(200, json=_completion(top))
        self.generations.append(body)
        return httpx.Response(200, json=_completion(None, "ok"))


@pytest.fixture
async def pm(db: Database):
    manager = ProviderManager(db)
    manager.stats = ProviderStats()
    request_logger.attach_stats(manager.stats)
    yield manager
    request_logger.attach_stats(None)
    await manager.stop()


@pytest.fixture
async def router(db: Database, pm: ProviderManager):
    p = await db.add_provider("a", ProviderProtocol.OPENAI)
    await db.add_address(p.id, URL, is_preferred=True)
    await db.update_provider_status(p.id, ProviderStatus.IDLE)
    await db.set_provider_models(
        p.id,
        [
            ProviderModel(provider_id=p.id, name=n)
            for n in (JUDGE, "asked", "small", "medium", "big")
        ],
    )
    await pm._rebuild_clients(await db.get_provider(p.id))
    rt = Router(db, pm)
    deps.init(db, pm, rt)
    return rt


@pytest.fixture
async def key(db: Database) -> tuple[int, str]:
    raw = generate_api_key()
    key_id = await db.create_api_key(
        key_prefix(raw), key_hash_legacy(raw), "balanced", True
    )
    return key_id, raw


@pytest.fixture
def backend(httpx_mock) -> Backend:
    b = Backend()
    httpx_mock.add_callback(b, url=f"{URL}/v1/chat/completions", is_reusable=True)
    return b


async def _post(path: str, body: dict, api_key: str) -> httpx.Response:
    transport = httpx.ASGITransport(app=api_app)
    async with httpx.AsyncClient(transport=transport, base_url="http://router") as c:
        return await c.post(path, json=body, headers={"API-KEY": api_key})


async def _set(db: Database, key_id: int, policy: dict) -> None:
    await db.set_api_key_decision_policy(key_id, parse_policy(policy).model_dump())


# --- The policy and the request state ---


def test_policy_validation():
    parse_policy({"model": JUDGE, "tier": TIER})
    parse_policy({"model": JUDGE, "gates": {"injection": INJECTION}})
    bad = [
        {"model": JUDGE},
        {"model": " ", "tier": TIER},
        {"model": JUDGE, "tier": {**TIER, "targets": {"5": "x"}}},
        {"model": JUDGE, "tier": {**TIER, "targets": {}}},
        {"model": JUDGE, "gates": {"g": {**INJECTION, "threshold": 0}}},
        {"model": JUDGE, "gates": {"g": {**INJECTION, "threshold": 1.5}}},
        {"model": JUDGE, "gates": {"g": {**INJECTION, "question": TIER["question"]}}},
        {"model": JUDGE, "tier": TIER, "extra": 1},
    ]
    for policy in bad:
        with pytest.raises(ValueError):
            parse_policy(policy)


def test_questions_follow_the_tier_models():
    policy = parse_policy(
        {
            "model": JUDGE,
            "tier": {**TIER, "models": ["auto"]},
            "gates": {"g": INJECTION},
        }
    )
    assert list(policy.questions_for("auto")) == ["tier", "gate:g"]
    assert list(policy.questions_for("llama3")) == ["gate:g"]


@pytest.mark.parametrize(
    "body,text",
    [
        (
            {
                "messages": [
                    {"role": "user", "content": "first"},
                    {"role": "assistant", "content": "reply"},
                    {"role": "user", "content": "last"},
                    {"role": "tool", "content": "out"},
                ]
            },
            "last",
        ),
        (
            {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "look"},
                            {"type": "image_url", "image_url": {"url": "x"}},
                        ],
                    }
                ]
            },
            "look",
        ),
        ({"messages": [{"role": "system", "content": "s"}]}, None),
        ({"prompt": "complete me"}, "complete me"),
        ({"prompt": ["a", "b"]}, "b"),
        ({"input": "hello"}, "hello"),
        (
            {
                "input": [
                    {"role": "user", "content": [{"type": "input_text", "text": "q"}]},
                    {"type": "function_call_output", "output": "x"},
                ]
            },
            "q",
        ),
        ({"prompt": "   "}, None),
        ({}, None),
    ],
)
def test_request_text(body, text):
    assert request_text(body) == text


# --- The request path ---


async def test_no_policy_means_no_decision_call(router, backend, key, monkeypatch):
    calls = []
    monkeypatch.setattr(svc, "emulate", lambda *a, **k: calls.append(a))
    resp = await _post("/v1/chat/completions", dict(CHAT), key[1])
    assert resp.status_code == 200, resp.text
    assert calls == [] and backend.decisions == []
    assert [g["model"] for g in backend.generations] == ["asked"]


async def test_tier_rewrites_the_model(db, router, backend, key):
    await _set(db, key[0], {"model": JUDGE, "tier": TIER})
    resp = await _post("/v1/chat/completions", dict(CHAT), key[1])
    assert resp.status_code == 200, resp.text
    assert len(backend.decisions) == 1
    assert backend.decisions[0]["model"] == JUDGE
    assert "Prove it." in json.dumps(backend.decisions[0]["messages"])
    assert [g["model"] for g in backend.generations] == ["big"]

    await drain()
    logs = await db.get_request_logs(10, 0)
    (decision,) = [e for e in logs if e.endpoint == "/v1/decisions"]
    (chat,) = [e for e in logs if e.endpoint == "/v1/chat/completions"]
    assert decision.model == JUDGE and decision.status == "ok"
    assert chat.model == "big"
    assert chat.request_meta.startswith("policy ")
    assert "tier=2 asked->big" in chat.request_meta


@pytest.mark.parametrize(
    "path,body",
    [
        ("/v1/completions", {"model": "asked", "prompt": "Prove it."}),
        ("/v1/responses", {"model": "asked", "input": "Prove it."}),
    ],
)
async def test_tier_applies_to_other_generation_routes(
    db, router, key, httpx_mock, path, body
):
    b = Backend()
    httpx_mock.add_callback(b, url=f"{URL}/v1/chat/completions", is_reusable=True)
    httpx_mock.add_callback(b, url=f"{URL}{path}", is_reusable=True, method="POST")
    await _set(db, key[0], {"model": JUDGE, "tier": TIER})
    resp = await _post(path, body, key[1])
    assert resp.status_code == 200, resp.text
    assert [g["model"] for g in b.generations] == ["big"]


async def test_tier_skips_models_it_does_not_name(db, router, backend, key):
    await _set(db, key[0], {"model": JUDGE, "tier": {**TIER, "models": ["auto"]}})
    resp = await _post("/v1/chat/completions", dict(CHAT), key[1])
    assert resp.status_code == 200, resp.text
    assert backend.decisions == []
    assert [g["model"] for g in backend.generations] == ["asked"]


async def test_level_without_a_target_keeps_the_model(db, router, backend, key):
    await _set(
        db, key[0], {"model": JUDGE, "tier": {**TIER, "targets": {"0": "small"}}}
    )
    resp = await _post("/v1/chat/completions", dict(CHAT), key[1])
    assert resp.status_code == 200, resp.text
    assert [g["model"] for g in backend.generations] == ["asked"]


async def test_no_decision_provider_falls_back(db, router, backend, key):
    await _set(db, key[0], {"model": "nowhere", "tier": TIER})
    resp = await _post("/v1/chat/completions", dict(CHAT), key[1])
    assert resp.status_code == 200, resp.text
    assert backend.decisions == []
    assert [g["model"] for g in backend.generations] == ["asked"]
    await drain()
    (chat,) = [
        e
        for e in await db.get_request_logs(10, 0)
        if e.endpoint == "/v1/chat/completions"
    ]
    assert chat.request_meta.startswith("policy: failed")


async def test_decision_error_falls_back(db, router, key, httpx_mock):
    b = Backend(decision_status=500)
    httpx_mock.add_callback(b, url=f"{URL}/v1/chat/completions", is_reusable=True)
    await _set(db, key[0], {"model": JUDGE, "tier": TIER, "gates": {"g": INJECTION}})
    resp = await _post("/v1/chat/completions", dict(CHAT), key[1])
    assert resp.status_code == 200, resp.text
    assert len(b.decisions) == 2
    assert [g["model"] for g in b.generations] == ["asked"]
    await drain()
    (decision,) = [
        e for e in await db.get_request_logs(10, 0) if e.endpoint == "/v1/decisions"
    ]
    assert decision.status == "error" and "HTTP 500" in decision.error_detail


async def test_decision_timeout_falls_back(db, router, backend, key, monkeypatch):
    async def slow(*args, **kwargs):
        await asyncio.sleep(5)

    monkeypatch.setattr(svc, "emulate", slow)
    monkeypatch.setattr(settings, "decision_policy_timeout_seconds", 0.05)
    await _set(db, key[0], {"model": JUDGE, "tier": TIER})
    resp = await _post("/v1/chat/completions", dict(CHAT), key[1])
    assert resp.status_code == 200, resp.text
    assert [g["model"] for g in backend.generations] == ["asked"]
    await drain()
    (decision,) = [
        e for e in await db.get_request_logs(10, 0) if e.endpoint == "/v1/decisions"
    ]
    assert "timed out" in decision.error_detail


async def test_gate_rejects_at_threshold(db, router, backend, key):
    await _set(db, key[0], {"model": JUDGE, "gates": {"injection": INJECTION}})
    resp = await _post("/v1/chat/completions", dict(CHAT), key[1])
    assert resp.status_code == 403
    detail = resp.json()["detail"]
    assert detail["gate"] == "injection"
    assert detail["probability"] == pytest.approx(0.9)
    assert detail["threshold"] == 0.9
    assert backend.generations == []
    await drain()
    (decision,) = [
        e for e in await db.get_request_logs(10, 0) if e.endpoint == "/v1/decisions"
    ]
    assert "injection=0.90" in decision.request_meta
    assert "rejected by injection" in decision.request_meta


async def test_gate_passes_below_threshold(db, router, backend, key):
    gate = {**INJECTION, "threshold": 0.95}
    await _set(db, key[0], {"model": JUDGE, "gates": {"injection": gate}})
    resp = await _post("/v1/chat/completions", dict(CHAT), key[1])
    assert resp.status_code == 200, resp.text
    assert [g["model"] for g in backend.generations] == ["asked"]


async def test_tier_and_gates_are_one_decision_call(
    db, router, backend, key, monkeypatch
):
    requests = []
    real = svc.emulate

    async def spy(pm, provider_id, backend_model, request, *a, **k):
        requests.append(request)
        return await real(pm, provider_id, backend_model, request, *a, **k)

    monkeypatch.setattr(svc, "emulate", spy)
    gates = {
        "injection": {**INJECTION, "threshold": 0.95},
        "abuse": {**INJECTION, "threshold": 0.99},
    }
    await _set(db, key[0], {"model": JUDGE, "tier": TIER, "gates": gates})
    resp = await _post("/v1/chat/completions", dict(CHAT), key[1])
    assert resp.status_code == 200, resp.text
    (req,) = requests
    assert list(req.questions) == ["tier", "gate:injection", "gate:abuse"]
    assert len(backend.decisions) == 3
    assert [g["model"] for g in backend.generations] == ["big"]


async def test_policy_change_applies_to_the_next_request(db, router, backend, key):
    resp = await _post("/v1/chat/completions", dict(CHAT), key[1])
    assert resp.status_code == 200
    await _set(db, key[0], {"model": JUDGE, "tier": TIER})
    resp = await _post("/v1/chat/completions", dict(CHAT), key[1])
    assert resp.status_code == 200
    assert await db.delete_api_key_decision_policy(key[0])
    resp = await _post("/v1/chat/completions", dict(CHAT), key[1])
    assert resp.status_code == 200
    assert [g["model"] for g in backend.generations] == ["asked", "big", "asked"]
    assert len(backend.decisions) == 1


async def test_invalid_stored_policy_is_ignored(db, router, backend, key):
    await db.set_api_key_decision_policy(key[0], {"model": JUDGE})
    resp = await _post("/v1/chat/completions", dict(CHAT), key[1])
    assert resp.status_code == 200, resp.text
    assert backend.decisions == []


async def test_policy_goes_with_the_key(db, key):
    await _set(db, key[0], {"model": JUDGE, "tier": TIER})
    await db.delete_api_key(key[0])
    assert await db.get_api_key_decision_policy(key[0]) is None


# --- Dashboard API ---


@pytest.fixture
async def dash(db: Database, pm: ProviderManager):
    dash_deps.init(db, pm)
    app = FastAPI()
    app.include_router(dash_decisions_router)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://dash") as c:
        yield c


async def test_dashboard_sets_gets_and_deletes(dash, db: Database, key):
    url = f"/api/keys/{key[0]}/decision-policy"
    assert (await dash.get(url)).json() == {"id": key[0], "policy": None}
    policy = {"model": JUDGE, "tier": TIER, "gates": {"injection": INJECTION}}
    resp = await dash.put(url, json=policy)
    assert resp.status_code == 200, resp.text
    assert resp.json()["policy"] == policy
    assert (await dash.get(url)).json()["policy"] == policy
    assert await db.get_api_key_decision_policy(key[0]) == policy
    assert (await dash.delete(url)).status_code == 200
    assert (await dash.delete(url)).status_code == 404
    assert (await dash.get(url)).json()["policy"] is None


async def test_dashboard_rejects_bad_policies(dash, key):
    url = f"/api/keys/{key[0]}/decision-policy"
    resp = await dash.put(
        url, json={"model": JUDGE, "tier": {**TIER, "targets": {"9": "x"}}}
    )
    assert resp.status_code == 400
    assert "unknown levels" in resp.json()["detail"]
    assert (await dash.put(url, content=b"nope")).status_code == 400
    assert (await dash.get("/api/keys/999/decision-policy")).status_code == 404
    resp = await dash.put("/api/keys/999/decision-policy", json={"model": JUDGE})
    assert resp.status_code == 404
