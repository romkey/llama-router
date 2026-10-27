from __future__ import annotations

import math

import pytest
from pydantic import ValidationError

from llama_router import decisions as d


def _lp(*pairs: tuple[str, float]) -> list[dict]:
    """top_logprobs entries from (token, probability) pairs."""
    return [{"token": t, "logprob": math.log(p)} for t, p in pairs]


def _resp(top: list[dict], usage: dict | None = None) -> dict:
    """An OpenAI-style chat completion as llama.cpp returns it."""
    return {
        "choices": [
            {
                "message": {"role": "assistant", "content": top[0]["token"]},
                "logprobs": {
                    "content": [
                        {**top[0], "top_logprobs": top},
                    ]
                },
            }
        ],
        "usage": usage or {"prompt_tokens": 100, "completion_tokens": 1},
    }


QUICKSTART = {
    "model": "jev-latest",
    "state": "Our API integration started returning 500 errors on every request.",
    "questions": {
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
            "criteria": [
                "Calm, just stating facts",
                "Frustrated but civil",
                "Very angry, strong language",
            ],
        },
        "is_urgent": {
            "type": "noul",
            "instructions": "The message conveys urgency or time-sensitivity",
        },
    },
}


# --- Request validation ---


def test_parses_typesafe_quickstart_request():
    req = d.DecisionRequest.model_validate(QUICKSTART)
    assert isinstance(req.questions["department"], d.ChoiceQuestion)
    assert isinstance(req.questions["frustration"], d.ScoreQuestion)
    assert isinstance(req.questions["is_urgent"], d.NoulQuestion)


@pytest.mark.parametrize(
    "state", ["plain text", {"ticket": {"id": 7, "body": "hi"}}, ["one", "two"]]
)
def test_accepts_each_state_shape(state):
    d.DecisionRequest.model_validate({**QUICKSTART, "state": state})


@pytest.mark.parametrize(
    "question",
    [
        {"type": "choice", "instructions": "x", "criteria": {"only": "one"}},
        {"type": "choice", "instructions": "x", "criteria": {"A": "1", "a": "2"}},
        {"type": "choice", "instructions": "x", "criteria": {" ": "1", "b": "2"}},
        {
            "type": "choice",
            "instructions": "x",
            "criteria": {f"k{i}": "v" for i in range(27)},
        },
        {"type": "score", "instructions": "x", "criteria": ["one"]},
        {"type": "score", "instructions": "x", "criteria": [str(i) for i in range(11)]},
        {"type": "noul", "instructions": "   "},
        {"type": "maybe", "instructions": "x"},
    ],
)
def test_rejects_malformed_questions(question):
    with pytest.raises(ValidationError):
        d.DecisionRequest.model_validate({**QUICKSTART, "questions": {"q": question}})


def test_rejects_empty_and_too_many_questions():
    q = {"type": "noul", "instructions": "x"}
    with pytest.raises(ValidationError):
        d.DecisionRequest.model_validate({**QUICKSTART, "questions": {}})
    many = {f"q{i}": q for i in range(d.MAX_QUESTIONS + 1)}
    with pytest.raises(ValidationError):
        d.DecisionRequest.model_validate({**QUICKSTART, "questions": many})


def test_rejects_oversized_state():
    with pytest.raises(ValidationError, match="state is longer"):
        d.DecisionRequest.model_validate(
            {**QUICKSTART, "state": "x" * (d.MAX_STATE_CHARS + 1)}
        )


# --- Prompts ---


def _req() -> d.DecisionRequest:
    return d.DecisionRequest.model_validate(QUICKSTART)


def test_choice_prompt_uses_letter_labels_and_keeps_key_names():
    req = _req()
    p = d.build_prompt(req.state, req.questions["department"])
    user = p.messages[-1]["content"]
    assert p.messages[0]["role"] == "system"
    assert user.startswith("State:\n" + req.state)
    assert "A) billing: Payment or subscription issues" in user
    assert "C) sales: Pricing or account questions" in user
    assert p.labels == {"a": "billing", "b": "technical", "c": "sales"}


def test_choice_prompt_name_scheme():
    req = _req()
    p = d.build_prompt(req.state, req.questions["department"], scheme="names")
    assert "technical: Bugs or integration problems" in p.messages[-1]["content"]
    assert p.labels == {
        "billing": "billing",
        "technical": "technical",
        "sales": "sales",
    }


def test_noul_is_asked_as_two_options_naming_the_false_case():
    req = _req()
    p = d.build_prompt(req.state, req.questions["is_urgent"])
    user = p.messages[-1]["content"]
    assert "Statement: The message conveys urgency" in user
    assert "B) False: " + d.DEFAULT_NOUL_NEGATIVE in user
    assert p.labels == {"a": d.NOUL_TRUE, "b": d.NOUL_FALSE}


def test_noul_with_negative_lists_both_situations_without_true_false():
    q = d.NoulQuestion(type="noul", instructions="It is urgent", negative="It can wait")
    user = d.build_prompt("s", q).messages[-1]["content"]
    assert "A) It is urgent\nB) It can wait" in user
    assert "True" not in user and "False" not in user


def test_score_prompt_numbers_levels_from_zero():
    req = _req()
    p = d.build_prompt(req.state, req.questions["frustration"])
    assert "0: Calm, just stating facts\n1: Frustrated" in p.messages[-1]["content"]
    assert p.labels == {"0": "0", "1": "1", "2": "2"}


def test_all_questions_on_a_state_share_the_prompt_prefix():
    req = _req()
    users = [
        d.build_prompt(req.state, q).messages[-1]["content"]
        for q in req.questions.values()
    ]
    prefix = "State:\n" + req.state + "\n\n"
    assert all(u.startswith(prefix) for u in users)


def test_json_and_list_states_render_readably():
    assert d.render_state({"a": 1}) == '{\n  "a": 1\n}'
    assert d.render_state(["hi", "there"]) == "[1] hi\n\n[2] there"


def test_upstream_body_asks_for_one_token_with_logprobs():
    p = d.build_prompt("s", d.NoulQuestion(type="noul", instructions="x"))
    body = d.upstream_body("llama3.1:8b", p)
    assert body["model"] == "llama3.1:8b"
    assert body["max_tokens"] == 1
    assert body["temperature"] == 0
    assert body["logprobs"] is True
    assert body["top_logprobs"] == d.DEFAULT_TOP_LOGPROBS
    assert body["stream"] is False
    assert "grammar" not in body


# --- Distributions ---


def test_renormalises_over_labels_and_merges_token_variants():
    # Shape seen from llama.cpp in Phase 0: pre-grammar top tokens include
    # labels with and without a leading space and non-label words.
    top = _lp(("B", 0.5), (" B", 0.1), ("A", 0.2), ("Technical", 0.1), ("C)", 0.05))
    dist = d.distribution(top, {"a": "billing", "b": "technical", "c": "sales"})
    assert dist.label_mass == pytest.approx(0.85)
    assert dist.probabilities == pytest.approx(
        {"billing": 0.2 / 0.85, "technical": 0.6 / 0.85, "sales": 0.05 / 0.85}
    )
    assert list(dist.probabilities) == ["billing", "technical", "sales"]


def test_label_missing_from_top_logprobs_gets_zero():
    dist = d.distribution(_lp(("0", 0.9), ("1", 0.1)), {"0": "0", "1": "1", "2": "2"})
    assert dist.probabilities["2"] == 0.0


def test_no_label_tokens_is_an_error():
    with pytest.raises(d.DecisionError, match="no answer label"):
        d.distribution(_lp(("Hello", 0.9)), {"a": "x", "b": "y"})


def test_skips_non_finite_logprobs():
    top = [{"token": "A", "logprob": float("-inf")}, *_lp(("B", 0.5))]
    dist = d.distribution(top, {"a": "x", "b": "y"})
    assert dist.probabilities == {"x": 0.0, "y": 1.0}


def test_reads_top_logprobs_from_chat_response():
    top = _lp(("A", 0.7), ("B", 0.3))
    assert d.top_logprobs_from_response(_resp(top)) == top


@pytest.mark.parametrize(
    "resp",
    [
        {},
        {"choices": []},
        {"choices": [{"logprobs": None}]},
        {"choices": [{"logprobs": {"content": []}}]},
    ],
)
def test_missing_logprobs_is_an_error(resp):
    with pytest.raises(d.DecisionError, match="no first-token logprobs"):
        d.top_logprobs_from_response(resp)


# --- Answers ---


def test_choice_answer_shape():
    req = _req()
    q = req.questions["department"]
    dist = d.Distribution({"billing": 0.0, "technical": 1.0, "sales": 0.0}, 1.0)
    assert d.answer(q, dist) == {
        "type": "choice",
        "choice": "technical",
        "confidence": 1.0,
        "probabilities": {"technical": 1.0, "billing": 0.0, "sales": 0.0},
    }


def test_score_answer_matches_typesafe_example():
    # TypeSafe quickstart: probabilities 0.33 / 0.67 / 0 give score 0.67 and
    # confidence 0.5.
    req = _req()
    q = req.questions["frustration"]
    dist = d.Distribution({"0": 0.33, "1": 0.67, "2": 0.0}, 1.0)
    a = d.answer(q, dist)
    assert a["type"] == "score"
    assert a["score"] == pytest.approx(0.67)
    assert a["confidence"] == pytest.approx(0.5, abs=0.01)
    assert a["legend"] == {
        "0": "Calm, just stating facts",
        "1": "Frustrated but civil",
        "2": "Very angry, strong language",
    }
    assert list(a["probabilities"]) == ["0", "1", "2"]


def test_noul_answer_is_probability_of_true():
    q = d.NoulQuestion(type="noul", instructions="x")
    dist = d.Distribution({d.NOUL_TRUE: 0.99, d.NOUL_FALSE: 0.01}, 1.0)
    assert d.answer(q, dist) == {"type": "noul", "noul": 0.99}


def test_confidence_is_zero_for_uniform_and_one_for_certain():
    assert d.confidence({"a": 1 / 3, "b": 1 / 3, "c": 1 / 3}) == pytest.approx(0)
    assert d.confidence({"a": 1.0, "b": 0.0}) == 1.0
    assert d.confidence({"a": 0.75, "b": 0.25}) == pytest.approx(0.5)


def test_end_to_end_from_backend_responses():
    req = _req()
    fixtures = {
        "department": _lp(("B", 0.9), ("A", 0.05), ("C", 0.05)),
        "frustration": _lp(("1", 0.6), ("0", 0.3), ("2", 0.1)),
        "is_urgent": _lp(("A", 0.8), ("B", 0.2)),
    }
    answers, responses = {}, []
    for qid, q in req.questions.items():
        prompt = d.build_prompt(req.state, q)
        resp = _resp(fixtures[qid])
        responses.append(resp)
        top = d.top_logprobs_from_response(resp)
        answers[qid] = d.answer(q, d.distribution(top, prompt.labels))
    assert answers["department"]["choice"] == "technical"
    assert answers["frustration"]["score"] == pytest.approx(0.8)
    assert answers["is_urgent"]["noul"] == pytest.approx(0.8)
    assert d.usage_from_responses(responses) == {
        "input_tokens": 300,
        "output_tokens": 3,
    }


# --- Calibration ---


def test_identity_calibration_changes_nothing():
    probs = {"a": 0.7, "b": 0.2, "c": 0.1}
    assert d.Calibration().apply(probs) == pytest.approx(probs)


def test_offsets_shift_the_prior_and_temperature_flattens():
    probs = {d.NOUL_TRUE: 0.9, d.NOUL_FALSE: 0.1}
    shifted = d.Calibration(offsets={d.NOUL_TRUE: -math.log(9)}).apply(probs)
    assert shifted[d.NOUL_TRUE] == pytest.approx(0.5)
    flat = d.Calibration(temperature=1e6).apply(probs)
    assert flat[d.NOUL_TRUE] == pytest.approx(0.5, abs=1e-4)


def test_calibration_handles_zero_probability():
    out = d.Calibration().apply({"0": 1.0, "1": 0.0})
    assert out["1"] > 0 and out["0"] == pytest.approx(1.0)


def test_calibration_round_trips_through_dict():
    c = d.Calibration(0.5, {"a": 1.0, "b": -1.0})
    assert d.Calibration.from_dict(c.to_dict()) == c
    with pytest.raises(ValueError):
        d.Calibration.from_dict({"temperature": 0})


def _log_loss(examples, cal=None):
    total = 0.0
    for probs, y in examples:
        p = cal.apply(probs) if cal else probs
        total -= math.log(max(p[y], 1e-12))
    return total / len(examples)


def test_fit_removes_a_yes_bias():
    # The Phase 0 pattern: urgent items score >= 0.89, but so do many
    # non-urgent ones. Ranking is good; the prior is shifted.
    pos = [0.9, 0.92, 0.95, 0.97, 0.99]
    neg = [0.05, 0.2, 0.4, 0.55, 0.62, 0.7, 0.75, 0.8, 0.85, 0.88]
    examples = [({d.NOUL_TRUE: p, d.NOUL_FALSE: 1 - p}, d.NOUL_TRUE) for p in pos]
    examples += [({d.NOUL_TRUE: p, d.NOUL_FALSE: 1 - p}, d.NOUL_FALSE) for p in neg]
    cal = d.fit_calibration(examples)
    assert _log_loss(examples, cal) < _log_loss(examples)

    def accuracy(c=None):
        return sum(
            max(c.apply(pr) if c else pr, key=(c.apply(pr) if c else pr).get) == y
            for pr, y in examples
        ) / len(examples)

    assert accuracy() == pytest.approx(8 / 15)  # 7 of 10 negatives read as yes
    assert accuracy(cal) >= 13 / 15
    assert sum(cal.offsets.values()) == pytest.approx(0, abs=1e-9)


def test_fit_corrects_middle_level_bias():
    def ex(p0, p1, p2, y):
        return ({"0": p0, "1": p1, "2": p2}, y)

    examples = [ex(0.3, 0.65, 0.05, "0")] * 6 + [ex(0.05, 0.9, 0.05, "1")] * 4
    examples += [ex(0.05, 0.6, 0.35, "2")] * 4
    cal = d.fit_calibration(examples, fit_temperature=False)
    assert cal.temperature == 1.0
    picks = [max(cal.apply(p), key=cal.apply(p).get) for p, _ in examples]
    assert picks == [y for _, y in examples]


def test_fit_rejects_inconsistent_examples():
    with pytest.raises(ValueError):
        d.fit_calibration([])
    with pytest.raises(ValueError):
        d.fit_calibration([({"a": 1.0, "b": 0.0}, "c")])
    with pytest.raises(ValueError):
        d.fit_calibration([({"a": 1.0, "b": 0.0}, "a"), ({"a": 1.0}, "a")])


# --- Keys ---


def test_question_key_follows_the_wording_only():
    a = d.NoulQuestion(type="noul", instructions="It is urgent")
    b = d.NoulQuestion(type="noul", instructions="It is urgent")
    c = d.NoulQuestion(type="noul", instructions="It is urgent", negative="It can wait")
    assert d.question_key(a) == d.question_key(b)
    assert d.question_key(a) != d.question_key(c)
    assert len(d.question_key(a)) == 32


def test_answer_keys_per_type():
    req = _req()
    assert d.answer_keys(req.questions["department"]) == [
        "billing",
        "technical",
        "sales",
    ]
    assert d.answer_keys(req.questions["frustration"]) == ["0", "1", "2"]
    assert d.answer_keys(req.questions["is_urgent"]) == [d.NOUL_TRUE, d.NOUL_FALSE]


def test_missing_logprobs_is_its_own_error():
    with pytest.raises(d.NoLogprobsError):
        d.top_logprobs_from_response({"choices": [{"message": {}}]})
    assert issubclass(d.NoLogprobsError, d.DecisionError)
