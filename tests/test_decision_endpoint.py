from __future__ import annotations

import json
import math

import httpx
import pytest

from llama_router import decisions as d
from llama_router.auto_benchmark import AutoBenchmarker
from llama_router.background import drain
from llama_router.database import Database
from llama_router.dashboard import deps as dash_deps
from llama_router.dashboard.decisions import router as dash_decisions_router
from llama_router.endpoints import DECIDE, DECISION_PATHS, endpoint_class
from llama_router.llamacpp_api import deps
from llama_router.llamacpp_api.app import app as api_app
from llama_router.models import ProviderModel, ProviderProtocol, ProviderStatus
from llama_router.provider_manager import ProviderManager
from llama_router import request_logger
from llama_router.request_logger import _request_meta_summary
from llama_router.router import Router
from llama_router.routing_stats import NO_LOGPROBS_TTL_S, ProviderStats
from llama_router.routing_table import TIER_MISSING, TIER_OK

MODEL = "m"
QUESTIONS = {
    "department": {
        "type": "choice",
        "instructions": "Which team should handle this",
        "criteria": {
            "billing": "Payment or subscription issues",
            "technical": "Bugs or integration problems",
            "sales": "Pricing or account questions",
        },
    },
    "frustration": {
        "type": "score",
        "instructions": "How frustrated the customer appears",
        "criteria": ["Calm", "Frustrated but civil", "Very angry"],
    },
    "is_urgent": {
        "type": "noul",
        "instructions": "The message conveys urgency or time-sensitivity",
    },
}
BODY = {
    "model": MODEL,
    "state": "Every API request returns 500.",
    "questions": QUESTIONS,
}


class FakeClock:
    def __init__(self) -> None:
        self.t = 1000.0

    def __call__(self) -> float:
        return self.t


def _top(*pairs: tuple[str, float]) -> list[dict]:
    return [{"token": t, "logprob": math.log(p)} for t, p in pairs]


def _completion(top: list[dict] | None) -> dict:
    choice: dict = {"message": {"role": "assistant", "content": "A"}}
    if top is not None:
        choice["logprobs"] = {"content": [{**top[0], "top_logprobs": top}]}
    return {
        "choices": [choice],
        "usage": {"prompt_tokens": 50, "completion_tokens": 1},
    }


def _answer_for(prompt: str) -> list[dict]:
    """What a sensible model says to each of the test questions."""
    if "Which team" in prompt:
        return _top(("B", 0.8), ("A", 0.1), ("C", 0.1))
    if "Levels:" in prompt:
        return _top(("2", 0.6), ("1", 0.3), ("0", 0.1))
    return _top(("A", 0.9), ("B", 0.1))


def _backend(with_logprobs: bool = True):
    seen: list[dict] = []

    def callback(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        seen.append(body)
        prompt = body["messages"][-1]["content"]
        top = _answer_for(prompt) if with_logprobs else None
        return httpx.Response(200, json=_completion(top))

    return callback, seen


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock()


@pytest.fixture
async def pm(db: Database, clock: FakeClock):
    manager = ProviderManager(db)
    manager.stats = ProviderStats(clock=clock)
    request_logger.attach_stats(manager.stats)
    yield manager
    request_logger.attach_stats(None)
    await manager.stop()


async def _provider(
    db: Database, pm: ProviderManager, name: str, url: str, caps: dict | None = None
):
    p = await db.add_provider(name, ProviderProtocol.OPENAI)
    await db.add_address(p.id, url, is_preferred=True)
    await db.update_provider_status(p.id, ProviderStatus.IDLE)
    await db.set_provider_models(p.id, [ProviderModel(provider_id=p.id, name=MODEL)])
    if caps is not None:
        await db.set_provider_capabilities(p.id, caps)
    await pm._rebuild_clients(await db.get_provider(p.id))
    return await db.get_provider(p.id)


def _caps(*supported: str, missing: tuple[str, ...] = ()) -> dict:
    eps = [{"path": p, "state": "available"} for p in supported]
    eps += [{"path": p, "state": "missing"} for p in missing]
    return {"endpoints": eps}


async def _post(path: str, body: dict) -> httpx.Response:
    transport = httpx.ASGITransport(app=api_app)
    async with httpx.AsyncClient(transport=transport, base_url="http://router") as c:
        return await c.post(path, json=body)


@pytest.fixture
async def router(db: Database, pm: ProviderManager):
    await db.set_allow_unauthenticated(True)
    rt = Router(db, pm)
    deps.init(db, pm, rt)
    return rt


# --- Endpoint registry and routing ---


def test_decision_routes_are_their_own_class():
    assert set(DECISION_PATHS) == {"/v1/decisions", "/v1/systemone"}
    for path in ("/v1/decisions", "/decisions", "/v1/systemone", "/systemone"):
        assert endpoint_class(path) == DECIDE


async def test_decisions_route_to_providers_with_chat_completions(
    db: Database, pm: ProviderManager, router: Router
):
    chat = await _provider(
        db, pm, "chat", "http://chat.test:8080", _caps("/v1/chat/completions")
    )
    nochat = await _provider(
        db,
        pm,
        "nochat",
        "http://nochat.test:8080",
        _caps(missing=("/v1/chat/completions",)),
    )
    table = await router.rebuild()
    by_name = {
        c.provider.name: c for c in table.all_candidates[(MODEL, "/v1/systemone")]
    }
    assert by_name["chat"].endpoint_support == "supported"
    assert by_name["chat"].tier == TIER_OK
    assert by_name["nochat"].tier == TIER_MISSING
    assert (await router.route(MODEL, "/v1/systemone")).provider.id == chat.id
    assert nochat.id is not None


async def test_no_logprobs_demotes_a_provider_for_an_hour(
    db: Database, pm: ProviderManager, router: Router, clock: FakeClock
):
    p = await _provider(
        db, pm, "a", "http://a.test:8080", _caps("/v1/chat/completions")
    )
    pm.stats.record_logprobs(p.id, ok=False)
    table = await router.rebuild()
    (cand,) = table.all_candidates[(MODEL, "/v1/decisions")]
    assert cand.endpoint_support == "missing"
    # Chat itself is untouched.
    (chat,) = table.all_candidates[(MODEL, "/v1/chat/completions")]
    assert chat.tier == TIER_OK
    clock.t += NO_LOGPROBS_TTL_S + 1
    table = await router.rebuild()
    (cand,) = table.all_candidates[(MODEL, "/v1/decisions")]
    assert cand.endpoint_support == "supported"


async def test_native_route_is_supported_even_without_logprobs(
    db: Database, pm: ProviderManager, router: Router
):
    p = await _provider(db, pm, "lite", "http://lite.test:4000", _caps("/v1/systemone"))
    pm.stats.record_logprobs(p.id, ok=False)
    table = await router.rebuild()
    (cand,) = table.all_candidates[(MODEL, "/v1/systemone")]
    assert cand.endpoint_support == "supported"


# --- The endpoint ---


async def test_emulated_request_answers_every_question(
    db: Database, pm: ProviderManager, router: Router, httpx_mock
):
    p = await _provider(
        db, pm, "a", "http://a.test:8080", _caps("/v1/chat/completions")
    )
    callback, seen = _backend()
    httpx_mock.add_callback(
        callback, url="http://a.test:8080/v1/chat/completions", is_reusable=True
    )
    resp = await _post("/v1/systemone", BODY)
    assert resp.status_code == 200, resp.text
    out = resp.json()
    assert out["model"] == MODEL
    a = out["answers"]
    assert list(a) == ["department", "frustration", "is_urgent"]
    assert a["department"]["choice"] == "technical"
    assert a["department"]["probabilities"]["technical"] == pytest.approx(0.8)
    assert a["frustration"]["score"] == pytest.approx(1.5)
    assert a["is_urgent"] == {"type": "noul", "noul": 0.9}
    assert out["usage"] == {"input_tokens": 150, "output_tokens": 3}
    assert len(seen) == 3
    assert all(b["max_tokens"] == 1 and b["logprobs"] for b in seen)
    assert pm.active_requests(p.id) == 0

    await drain()
    (entry,) = [
        e for e in await db.get_request_logs(10, 0) if e.endpoint == "/v1/systemone"
    ]
    assert entry.status == "ok" and entry.provider_id == p.id
    assert entry.request_meta.startswith("questions=3; department=technical")
    stats = pm.stats.model(p.id, MODEL, DECIDE)
    assert stats is not None and stats.duration_ms.value is not None


async def test_calibration_is_applied_by_model_and_question(
    db: Database, pm: ProviderManager, router: Router, httpx_mock
):
    await _provider(db, pm, "a", "http://a.test:8080", _caps("/v1/chat/completions"))
    callback, _ = _backend()
    httpx_mock.add_callback(
        callback, url="http://a.test:8080/v1/chat/completions", is_reusable=True
    )
    q = d.NoulQuestion.model_validate(QUESTIONS["is_urgent"])
    # Undo a 9:1 yes bias: p(true) 0.9 becomes 0.5.
    await db.set_decision_calibration(
        MODEL,
        d.question_key(q),
        "noul",
        q.model_dump(),
        1.0,
        {d.NOUL_TRUE: -math.log(3), d.NOUL_FALSE: math.log(3)},
    )
    resp = await _post("/v1/decisions", BODY)
    a = resp.json()["answers"]
    assert a["is_urgent"]["noul"] == pytest.approx(0.5)
    # Other questions and other models are untouched.
    assert a["department"]["probabilities"]["technical"] == pytest.approx(0.8)
    await drain()
    (entry,) = [
        e for e in await db.get_request_logs(10, 0) if e.endpoint == "/v1/decisions"
    ]
    assert "calibrated=1" in entry.request_meta

    # Deleting it takes effect at once (the cache follows the table).
    (row,) = await db.list_decision_calibrations()
    await db.delete_decision_calibration(row["id"])
    resp = await _post("/v1/decisions", BODY)
    assert resp.json()["answers"]["is_urgent"]["noul"] == pytest.approx(0.9)


async def test_native_upstream_gets_the_request_as_sent(
    db: Database, pm: ProviderManager, router: Router, httpx_mock
):
    await _provider(db, pm, "lite", "http://lite.test:4000", _caps("/v1/systemone"))
    native = {
        "model": "jev-1.13.0",
        "answers": {"is_urgent": {"type": "noul", "noul": 0.99}},
        "usage": {"input_tokens": 10, "output_tokens": 1},
    }
    httpx_mock.add_response(
        url="http://lite.test:4000/v1/systemone", method="POST", json=native
    )
    body = {**BODY, "questions": {"is_urgent": QUESTIONS["is_urgent"]}}
    resp = await _post("/v1/systemone", body)
    assert resp.status_code == 200
    assert resp.json() == native
    (sent,) = httpx_mock.get_requests()
    assert json.loads(sent.content) == body


async def test_no_logprobs_reroutes_to_a_provider_that_has_them(
    db: Database, pm: ProviderManager, router: Router, httpx_mock
):
    bad = await _provider(
        db, pm, "bad", "http://bad.test:8080", _caps("/v1/chat/completions")
    )
    good = await _provider(
        db, pm, "good", "http://good.test:8080", _caps("/v1/chat/completions")
    )
    # Make "bad" the first choice: "good" is busy.
    pm.acquire(good.id)
    pm.acquire(good.id)
    bad_cb, bad_seen = _backend(with_logprobs=False)
    good_cb, good_seen = _backend()
    httpx_mock.add_callback(
        bad_cb, url="http://bad.test:8080/v1/chat/completions", is_reusable=True
    )
    httpx_mock.add_callback(
        good_cb, url="http://good.test:8080/v1/chat/completions", is_reusable=True
    )
    resp = await _post("/v1/systemone", BODY)
    assert resp.status_code == 200, resp.text
    assert bad_seen and len(good_seen) == 3
    assert pm.stats.lacks_logprobs(bad.id)
    assert not pm.stats.lacks_logprobs(good.id)
    pm.release(good.id)
    pm.release(good.id)


async def test_no_provider_with_logprobs_is_a_502(
    db: Database, pm: ProviderManager, router: Router, httpx_mock
):
    await _provider(
        db, pm, "bad", "http://bad.test:8080", _caps("/v1/chat/completions")
    )
    cb, _ = _backend(with_logprobs=False)
    httpx_mock.add_callback(
        cb, url="http://bad.test:8080/v1/chat/completions", is_reusable=True
    )
    resp = await _post("/v1/systemone", BODY)
    assert resp.status_code == 502
    assert "logprobs" in resp.json()["detail"]


async def test_off_label_answer_is_a_502_naming_the_question(
    db: Database, pm: ProviderManager, router: Router, httpx_mock
):
    await _provider(db, pm, "a", "http://a.test:8080", _caps("/v1/chat/completions"))
    httpx_mock.add_response(
        url="http://a.test:8080/v1/chat/completions",
        json=_completion(_top(("Hello", 0.9))),
    )
    body = {**BODY, "questions": {"is_urgent": QUESTIONS["is_urgent"]}}
    resp = await _post("/v1/systemone", body)
    assert resp.status_code == 502
    assert "is_urgent" in resp.json()["detail"]


async def test_upstream_http_error_is_forwarded(
    db: Database, pm: ProviderManager, router: Router, httpx_mock
):
    await _provider(db, pm, "a", "http://a.test:8080", _caps("/v1/chat/completions"))
    httpx_mock.add_response(
        url="http://a.test:8080/v1/chat/completions",
        status_code=503,
        json={"error": "loading model"},
    )
    body = {**BODY, "questions": {"is_urgent": QUESTIONS["is_urgent"]}}
    resp = await _post("/v1/systemone", body)
    assert resp.status_code == 503
    assert resp.json() == {"error": "loading model"}


@pytest.mark.parametrize(
    "body",
    [
        {"state": "x", "questions": QUESTIONS},
        {"model": MODEL, "state": "x", "questions": {}},
        {"model": MODEL, "state": "x", "questions": {"q": {"type": "maybe"}}},
    ],
)
async def test_invalid_requests_are_400(router: Router, body):
    resp = await _post("/v1/systemone", body)
    assert resp.status_code == 400
    assert isinstance(resp.json()["detail"], list)


async def test_unknown_model_is_404(router: Router):
    resp = await _post("/v1/systemone", BODY)
    assert resp.status_code == 404


def test_request_meta_counts_questions():
    assert _request_meta_summary(BODY) == "questions=3"


# --- Calibrations in the database and the dashboard API ---


async def test_calibration_upsert_and_delete(db: Database):
    await db.set_decision_calibration("m", "k", "noul", {"x": 1}, 2.0, {"true": 1.0})
    await db.set_decision_calibration("m", "k", "noul", {"x": 2}, 0.5, {"true": -1.0})
    (row,) = await db.list_decision_calibrations()
    assert row["temperature"] == 0.5 and row["offsets"] == {"true": -1.0}
    assert row["question"] == {"x": 2}
    assert await db.delete_decision_calibration(row["id"])
    assert not await db.delete_decision_calibration(row["id"])
    assert await db.list_decision_calibrations() == []


@pytest.fixture
async def dash(db: Database, pm: ProviderManager):
    from fastapi import FastAPI

    dash_deps.init(db, pm)
    app = FastAPI()
    app.include_router(dash_decisions_router)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://dash") as c:
        yield c


async def test_dashboard_sets_lists_and_deletes(dash, db: Database):
    q = QUESTIONS["is_urgent"]
    r = await dash.post(
        "/api/decisions/calibrations",
        json={"model": MODEL, "question": q, "offsets": {"true": -1.0}},
    )
    assert r.status_code == 200, r.text
    assert r.json()["question_key"] == d.question_key(d.NoulQuestion.model_validate(q))
    (row,) = (await dash.get("/api/decisions/calibrations")).json()["calibrations"]
    assert row["model_name"] == MODEL and row["temperature"] == 1.0
    assert (
        await dash.delete(f"/api/decisions/calibrations/{row['id']}")
    ).status_code == 200
    assert (
        await dash.delete(f"/api/decisions/calibrations/{row['id']}")
    ).status_code == 404


async def test_dashboard_fits_from_examples(dash, db: Database):
    pos = [0.9, 0.95, 0.99]
    neg = [0.6, 0.7, 0.8, 0.85]
    examples = [
        {"probabilities": {"true": p, "false": 1 - p}, "answer": "true"} for p in pos
    ] + [{"probabilities": {"true": p, "false": 1 - p}, "answer": "false"} for p in neg]
    r = await dash.post(
        "/api/decisions/calibrations",
        json={"model": MODEL, "question": QUESTIONS["is_urgent"], "examples": examples},
    )
    assert r.status_code == 200, r.text
    out = r.json()
    assert out["examples"] == 7
    cal = d.Calibration.from_dict(out)
    assert cal.apply({"true": 0.8, "false": 0.2})["true"] < 0.5
    assert cal.apply({"true": 0.97, "false": 0.03})["true"] > 0.5


@pytest.mark.parametrize(
    "body",
    [
        {"question": QUESTIONS["is_urgent"]},
        {"model": MODEL, "question": {"type": "noul"}},
        {"model": MODEL, "question": QUESTIONS["is_urgent"], "offsets": {"maybe": 1}},
        {"model": MODEL, "question": QUESTIONS["is_urgent"], "temperature": 0},
        {"model": MODEL, "question": QUESTIONS["is_urgent"], "examples": []},
        {
            "model": MODEL,
            "question": QUESTIONS["is_urgent"],
            "examples": [{"probabilities": {"true": 1.0}, "answer": "true"}],
        },
        {
            "model": MODEL,
            "question": QUESTIONS["is_urgent"],
            "examples": [
                {"probabilities": {"true": 0.5, "false": 0.5}, "answer": "maybe"}
            ],
        },
    ],
)
async def test_dashboard_rejects_bad_calibrations(dash, body):
    r = await dash.post("/api/decisions/calibrations", json=body)
    assert r.status_code == 400


# --- Auto-benchmark decision check ---


async def test_benchmark_check_learns_logprobs_support(
    db: Database, pm: ProviderManager, httpx_mock
):
    good = await _provider(
        db, pm, "good", "http://good.test:8080", _caps("/v1/chat/completions")
    )
    bad = await _provider(
        db, pm, "bad", "http://bad.test:8080", _caps("/v1/chat/completions")
    )
    native = await _provider(
        db, pm, "lite", "http://lite.test:4000", _caps("/v1/systemone")
    )
    good_cb, _ = _backend()
    bad_cb, _ = _backend(with_logprobs=False)
    httpx_mock.add_callback(good_cb, url="http://good.test:8080/v1/chat/completions")
    httpx_mock.add_callback(bad_cb, url="http://bad.test:8080/v1/chat/completions")
    ab = AutoBenchmarker(db, pm)
    assert await ab.check_decisions(good.id, MODEL) == "ok"
    assert await ab.check_decisions(bad.id, MODEL) == "no_logprobs"
    assert await ab.check_decisions(native.id, MODEL) == "skipped"
    assert pm.stats.lacks_logprobs(bad.id) and not pm.stats.lacks_logprobs(good.id)
    assert pm.stats.model(good.id, MODEL, DECIDE) is not None
    # The check is not traffic: it never touches the request counters.
    assert pm.idle_seconds(good.id, 10**9) > 0


async def test_calibration_written_during_a_cache_load_is_not_lost(db: Database):
    from llama_router import decision_service as svc

    store = svc.CalibrationStore(db)
    q = d.NoulQuestion.model_validate(QUESTIONS["is_urgent"])
    req = d.DecisionRequest.model_validate(
        {**BODY, "questions": {"is_urgent": QUESTIONS["is_urgent"]}}
    )
    real_list = db.list_decision_calibrations

    async def list_then_write():
        rows = await real_list()  # the snapshot, taken before the write
        await db.set_decision_calibration(
            MODEL, d.question_key(q), "noul", q.model_dump(), 1.0, {"true": -1.0}
        )
        return rows

    db.list_decision_calibrations = list_then_write
    try:
        assert await store.for_request(MODEL, req) == {}  # read before the write
    finally:
        db.list_decision_calibrations = real_list
    # The stale snapshot was not cached: the next request sees the write.
    assert set(await store.for_request(MODEL, req)) == {"is_urgent"}


def _rejects_logprobs(status: int = 400):
    """A server that refuses the logprobs fields but answers without them."""
    seen: list[dict] = []

    def callback(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        seen.append(body)
        if "logprobs" in body or "top_logprobs" in body:
            return httpx.Response(status, json={"error": "logprobs not supported"})
        return httpx.Response(200, json=_completion(None))

    return callback, seen


async def test_backend_rejecting_logprobs_is_demoted_and_rerouted(
    db: Database, pm: ProviderManager, router: Router, httpx_mock
):
    bad = await _provider(
        db, pm, "bad", "http://bad.test:8080", _caps("/v1/chat/completions")
    )
    good = await _provider(
        db, pm, "good", "http://good.test:8080", _caps("/v1/chat/completions")
    )
    pm.acquire(good.id)  # "bad" is the first choice
    pm.acquire(good.id)
    bad_cb, bad_seen = _rejects_logprobs(422)
    good_cb, good_seen = _backend()
    httpx_mock.add_callback(
        bad_cb, url="http://bad.test:8080/v1/chat/completions", is_reusable=True
    )
    httpx_mock.add_callback(
        good_cb, url="http://good.test:8080/v1/chat/completions", is_reusable=True
    )
    resp = await _post("/v1/systemone", BODY)
    pm.release(good.id)
    pm.release(good.id)
    assert resp.status_code == 200, resp.text
    assert len(good_seen) == 3
    assert pm.stats.lacks_logprobs(bad.id)
    # Three questions failed on "bad", but they shared one probe.
    assert sum("logprobs" not in b for b in bad_seen) == 1


async def test_genuine_bad_request_is_forwarded_and_not_demoted(
    db: Database, pm: ProviderManager, router: Router, httpx_mock
):
    p = await _provider(
        db, pm, "a", "http://a.test:8080", _caps("/v1/chat/completions")
    )
    httpx_mock.add_response(
        url="http://a.test:8080/v1/chat/completions",
        status_code=400,
        json={"error": "the prompt is longer than the context"},
        is_reusable=True,
    )
    body = {**BODY, "questions": {"is_urgent": QUESTIONS["is_urgent"]}}
    resp = await _post("/v1/systemone", body)
    assert resp.status_code == 400
    assert resp.json() == {"error": "the prompt is longer than the context"}
    assert not pm.stats.lacks_logprobs(p.id)


async def test_calibration_saved_under_an_alias_applies(
    db: Database, pm: ProviderManager, router: Router, httpx_mock
):
    await _provider(db, pm, "a", "http://a.test:8080", _caps("/v1/chat/completions"))
    await db.set_model_alias("urgent-model", MODEL)
    callback, _ = _backend()
    httpx_mock.add_callback(
        callback, url="http://a.test:8080/v1/chat/completions", is_reusable=True
    )
    q = d.NoulQuestion.model_validate(QUESTIONS["is_urgent"])
    await db.set_decision_calibration(
        "urgent-model",
        d.question_key(q),
        "noul",
        q.model_dump(),
        1.0,
        {d.NOUL_TRUE: -math.log(3), d.NOUL_FALSE: math.log(3)},
    )
    await router.rebuild()
    body = {**BODY, "questions": {"is_urgent": QUESTIONS["is_urgent"]}}
    resp = await _post("/v1/systemone", body)
    assert resp.json()["answers"]["is_urgent"]["noul"] == pytest.approx(0.5)


def test_model_names_put_given_names_first_then_aliases():
    from llama_router.decision_service import model_names

    aliases = {"fast": "m", "quick": "fast"}
    assert model_names(["m", None, "m"], aliases) == ["m", "fast", "quick"]
    assert model_names(["x"], aliases) == ["x"]
    assert model_names(["m"], None) == ["m"]
