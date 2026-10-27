"""The dashboard's Decisions page: fitting calibrations from labelled states
and trying questions."""

from __future__ import annotations

import json

import httpx
import pytest

from llama_router import decisions as d
from llama_router.dashboard import decisions as dash_decisions
from llama_router.dashboard.auth_core import hash_password
from llama_router.dashboard.middleware import invalidate_dashboard_user_count_cache
from llama_router.database import Database
from llama_router.models import ProviderModel, ProviderProtocol
from llama_router.provider_manager import ProviderManager
from llama_router.router import Router
from tests.test_dashboard_auth import _session_cookie_header
from tests.test_dashboard_auth import dash_client  # noqa: F401  (fixture)
from tests.test_decision_endpoint import (  # noqa: F401  (fixtures)
    MODEL,
    QUESTIONS,
    _caps,
    _completion,
    _provider,
    _top,
    clock,
    dash,
    pm,
    router,
)

FRUSTRATION = QUESTIONS["frustration"]
CHAT = "http://chat.test:8080/v1/chat/completions"

# A model with the middle-level bias Phase 0 found: it ranks the states
# right but puts most of its mass on level 1.
_BIASED = {
    "calm": _top(("1", 0.55), ("0", 0.4), ("2", 0.05)),
    "civil": _top(("1", 0.8), ("0", 0.1), ("2", 0.1)),
    "angry": _top(("1", 0.55), ("2", 0.4), ("0", 0.05)),
}
_LEVEL = {"calm": "0", "civil": "1", "angry": "2"}


def _labelled(n_each: int = 4) -> list[dict]:
    return [
        {"state": f"ticket {i}: the customer is {mood}", "answer": _LEVEL[mood]}
        for mood in _LEVEL
        for i in range(n_each)
    ]


def _biased_backend():
    seen: list[dict] = []

    def callback(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        seen.append(body)
        prompt = body["messages"][-1]["content"]
        mood = next(m for m in _BIASED if f"is {m}" in prompt)
        return httpx.Response(200, json=_completion(_BIASED[mood]))

    return callback, seen


async def test_fit_stores_a_calibration_and_reports_held_out_scores(
    dash, db: Database, pm: ProviderManager, router: Router, httpx_mock
):
    await _provider(
        db, pm, "chat", "http://chat.test:8080", _caps("/v1/chat/completions")
    )
    callback, seen = _biased_backend()
    httpx_mock.add_callback(callback, url=CHAT, is_reusable=True)

    r = await dash.post(
        "/api/decisions/calibrations/fit",
        json={"model": MODEL, "question": FRUSTRATION, "examples": _labelled()},
    )
    assert r.status_code == 200, r.text
    out = r.json()
    assert len(seen) == 12
    # Requests carry no calibration: they are the raw one-token prompts.
    assert all(b["max_tokens"] == 1 and b["logprobs"] for b in seen)
    assert out["model"] == MODEL and out["provider"] == "chat"
    assert out["examples"] == 12
    assert out["method"] == "leave-one-out"
    assert out["before"]["accuracy"] == pytest.approx(4 / 12, abs=1e-3)
    assert out["after"]["accuracy"] == 1.0
    assert out["after"]["log_loss"] < out["before"]["log_loss"]
    assert out["after"]["mae"] < out["before"]["mae"]
    assert out["label_mass"] == pytest.approx(1.0)

    (row,) = await db.list_decision_calibrations()
    assert row["model_name"] == MODEL
    assert row["question_type"] == "score"
    assert row["question_key"] == d.question_key(
        d.ScoreQuestion.model_validate(FRUSTRATION)
    )
    assert row["examples"] == 12
    cal = d.Calibration.from_dict(row)
    calm = d.distribution(_BIASED["calm"], {"0": "0", "1": "1", "2": "2"})
    probs = cal.apply(calm.probabilities)
    assert max(probs, key=probs.__getitem__) == "0"


async def test_fit_accepts_any_case_choice_keys_and_bool_noul(
    dash, db: Database, pm: ProviderManager, router: Router, httpx_mock
):
    await _provider(
        db, pm, "chat", "http://chat.test:8080", _caps("/v1/chat/completions")
    )
    httpx_mock.add_response(
        url=CHAT, json=_completion(_top(("A", 0.9), ("B", 0.1))), is_reusable=True
    )
    examples = [{"state": f"s{i}", "answer": i % 2 == 0} for i in range(4)]
    r = await dash.post(
        "/api/decisions/calibrations/fit",
        json={"model": MODEL, "question": QUESTIONS["is_urgent"], "examples": examples},
    )
    assert r.status_code == 200, r.text
    assert set(r.json()["offsets"]) == {"true", "false"}

    examples = [{"state": f"s{i}", "answer": "BILLING"} for i in range(4)]
    r = await dash.post(
        "/api/decisions/calibrations/fit",
        json={
            "model": MODEL,
            "question": QUESTIONS["department"],
            "examples": examples,
        },
    )
    assert r.status_code == 200, r.text
    assert len(await db.list_decision_calibrations()) == 2


async def test_fit_switches_to_k_fold_for_many_examples(
    dash, db: Database, pm: ProviderManager, router: Router, httpx_mock, monkeypatch
):
    monkeypatch.setattr(dash_decisions, "LOO_MAX", 6)
    monkeypatch.setattr(dash_decisions, "FIT_FOLDS", 3)
    await _provider(
        db, pm, "chat", "http://chat.test:8080", _caps("/v1/chat/completions")
    )
    callback, _seen = _biased_backend()
    httpx_mock.add_callback(callback, url=CHAT, is_reusable=True)
    r = await dash.post(
        "/api/decisions/calibrations/fit",
        json={"model": MODEL, "question": FRUSTRATION, "examples": _labelled()},
    )
    assert r.status_code == 200, r.text
    assert r.json()["method"] == "3-fold"


async def test_fit_rejects_a_native_provider(
    dash, db: Database, pm: ProviderManager, router: Router, httpx_mock
):
    await _provider(db, pm, "lite", "http://lite.test:4000", _caps("/v1/systemone"))
    r = await dash.post(
        "/api/decisions/calibrations/fit",
        json={"model": MODEL, "question": FRUSTRATION, "examples": _labelled()},
    )
    assert r.status_code == 400
    assert "lite" in r.json()["detail"]
    assert httpx_mock.get_requests() == []
    assert await db.list_decision_calibrations() == []


async def test_fit_without_logprobs_is_a_502_and_stores_nothing(
    dash, db: Database, pm: ProviderManager, router: Router, httpx_mock
):
    await _provider(
        db, pm, "chat", "http://chat.test:8080", _caps("/v1/chat/completions")
    )
    httpx_mock.add_response(url=CHAT, json=_completion(None), is_reusable=True)
    r = await dash.post(
        "/api/decisions/calibrations/fit",
        json={"model": MODEL, "question": FRUSTRATION, "examples": _labelled()},
    )
    assert r.status_code == 502
    assert "logprobs" in r.json()["detail"]
    assert await db.list_decision_calibrations() == []


async def test_fit_unknown_model_is_404(dash, router: Router):
    r = await dash.post(
        "/api/decisions/calibrations/fit",
        json={"model": "nope", "question": FRUSTRATION, "examples": _labelled()},
    )
    assert r.status_code == 404


@pytest.mark.parametrize(
    "body",
    [
        {"question": FRUSTRATION, "examples": _labelled()},
        {"model": MODEL, "question": {"type": "score"}, "examples": _labelled()},
        {"model": MODEL, "question": FRUSTRATION},
        {"model": MODEL, "question": FRUSTRATION, "examples": _labelled(1)},
        {"model": MODEL, "question": FRUSTRATION, "examples": "not a list"},
        {
            "model": MODEL,
            "question": FRUSTRATION,
            "examples": _labelled() + [{"state": "x", "answer": "3"}],
        },
        {
            "model": MODEL,
            "question": FRUSTRATION,
            "examples": _labelled() + [{"answer": "1"}],
        },
        {
            "model": MODEL,
            "question": FRUSTRATION,
            "examples": _labelled() + [{"state": 42, "answer": "1"}],
        },
        {
            "model": MODEL,
            "question": FRUSTRATION,
            "examples": _labelled(dash_decisions.MAX_FIT_STATES),
        },
    ],
)
async def test_fit_rejects_bad_input(dash, router: Router, httpx_mock, body):
    r = await dash.post("/api/decisions/calibrations/fit", json=body)
    assert r.status_code == 400, r.text
    assert httpx_mock.get_requests() == []


async def test_try_answers_through_the_decision_endpoint(
    dash, db: Database, pm: ProviderManager, router: Router, httpx_mock
):
    await _provider(
        db, pm, "chat", "http://chat.test:8080", _caps("/v1/chat/completions")
    )
    callback, _seen = _biased_backend()
    httpx_mock.add_callback(callback, url=CHAT, is_reusable=True)
    body = {
        "model": MODEL,
        "state": "the customer is calm",
        "questions": {"f": FRUSTRATION},
    }
    r = await dash.post("/api/decisions/try", json=body)
    assert r.status_code == 200, r.text
    (a,) = r.json()["answers"].values()
    assert a["type"] == "score" and a["probabilities"]["1"] == pytest.approx(0.55)

    # A stored calibration applies, as it does on /v1/systemone.
    fit = await dash.post(
        "/api/decisions/calibrations/fit",
        json={"model": MODEL, "question": FRUSTRATION, "examples": _labelled()},
    )
    assert fit.status_code == 200, fit.text
    r = await dash.post("/api/decisions/try", json=body)
    probs = r.json()["answers"]["f"]["probabilities"]
    assert max(probs, key=probs.__getitem__) == "0"


async def test_try_rejects_bad_requests(dash, router: Router):
    r = await dash.post("/api/decisions/try", json={"model": MODEL})
    assert r.status_code == 400
    r = await dash.post("/api/decisions/try", content=b"not json")
    assert r.status_code == 400


async def test_decisions_page_renders_for_a_logged_in_user(
    dash_client,  # noqa: F811
) -> None:
    client, db = dash_client
    uid = await db.create_dashboard_user("adm", hash_password("password-12345"), True)
    invalidate_dashboard_user_count_cache()
    p = await db.add_provider("gpu-1", ProviderProtocol.OPENAI)
    await db.set_provider_models(p.id, [ProviderModel(provider_id=p.id, name="m<1>")])

    assert (await client.get("/decisions")).status_code == 302
    r = await client.get("/decisions", headers=_session_cookie_header(uid))
    assert r.status_code == 200
    assert "Fit a calibration" in r.text
    assert 'href="/decisions"' in r.text
    # Model names are escaped into the picker.
    assert '<option value="m&lt;1&gt;">' in r.text
    assert "m<1>" not in r.text


async def test_decisions_page_is_read_only_for_viewers(
    dash_client,  # noqa: F811
) -> None:
    client, db = dash_client
    await db.create_dashboard_user("adm", hash_password("password-12345"), True)
    uid = await db.create_dashboard_user(
        "viewer", hash_password("password-12345"), False
    )
    invalidate_dashboard_user_count_cache()
    r = await client.get("/decisions", headers=_session_cookie_header(uid))
    assert r.status_code == 200
    assert "Calibrations" in r.text
    assert "Fit a calibration" not in r.text
    r = await client.post(
        "/api/decisions/try", json={}, headers=_session_cookie_header(uid)
    )
    assert r.status_code == 403
