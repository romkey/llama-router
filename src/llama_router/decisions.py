"""Decision questions answered by a chat model's first-token probabilities.

A decision request, in the shape of TypeSafe's Jev API, carries a *state*
(text, a JSON object, or a list of strings) and named, typed questions:

- ``noul``: a statement; the answer is the probability it is true.
- ``choice``: unordered options; the answer is a distribution over them.
- ``score``: an ordered rubric; the answer is a distribution over levels and
  its expected level.

Each question becomes one chat prompt that lists the options under
single-token labels (``A``, ``B``...; ``0``, ``1``...) and asks for the label
alone. The backend is asked for one token with its top logprobs, and the
probability of each answer is read off the label tokens and renormalised over
the label set. llama.cpp reports logprobs from before any grammar is applied,
so constraining the output would not change the distribution; the request
asks for the label in words and relies on the renormalisation.

Measured on local Llama models (``docs/decision-models-plan.md``, Phase 0):

- choice questions with letter labels are accurate and well calibrated;
- a plain yes/no question is strongly biased toward "yes", so ``noul`` is
  asked as a two-option choice that names the false case;
- score questions drift to the middle level and need per-model calibration
  (:class:`Calibration`, fit with :func:`fit_calibration`).

Everything here is pure: no I/O, no settings, no clock. The endpoint does the
network calls and passes the upstream responses in.
"""

from __future__ import annotations

import hashlib
import json
import math
import string
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Annotated, Any, Literal, Union

from pydantic import BaseModel, Field, field_validator, model_validator

MAX_QUESTIONS = 32
MAX_CHOICE_OPTIONS = 26  # letters A..Z
MAX_SCORE_LEVELS = 10  # digits 0..9
# About 32k tokens at four characters a token, matching Jev's state limit.
MAX_STATE_CHARS = 128_000
DEFAULT_TOP_LOGPROBS = 20

SYSTEM_PROMPT = (
    "You are a precise classifier. You read a state and answer one question "
    "about it. Reply with exactly one option label and nothing else."
)
NOUL_TRUE = "true"
NOUL_FALSE = "false"
DEFAULT_NOUL_NEGATIVE = "the statement does not hold for this state"

LabelScheme = Literal["letters", "names"]


class DecisionError(ValueError):
    """A question could not be answered from the backend's response."""


class NoLogprobs(DecisionError):
    """The backend answered without logprobs, so it cannot serve decisions."""


# --- Request models ---


def _nonblank(value: str) -> str:
    if not value.strip():
        raise ValueError("must not be blank")
    return value


class NoulQuestion(BaseModel):
    type: Literal["noul"]
    instructions: str
    # Extension: a description of the situation in which the statement is
    # false. Asking yes/no as two labelled options removes most of the yes
    # bias on its own; this lets a caller name the alternative concretely.
    negative: str | None = None

    _check = field_validator("instructions")(_nonblank)


class ChoiceQuestion(BaseModel):
    type: Literal["choice"]
    instructions: str
    criteria: dict[str, str]

    _check = field_validator("instructions")(_nonblank)

    @field_validator("criteria")
    @classmethod
    def _criteria(cls, v: dict[str, str]) -> dict[str, str]:
        if not 2 <= len(v) <= MAX_CHOICE_OPTIONS:
            raise ValueError(f"needs 2 to {MAX_CHOICE_OPTIONS} options")
        if any(not k.strip() for k in v):
            raise ValueError("option keys must not be blank")
        if len({k.strip().lower() for k in v}) != len(v):
            raise ValueError("option keys must differ ignoring case")
        return v


class ScoreQuestion(BaseModel):
    type: Literal["score"]
    instructions: str
    criteria: list[str]

    _check = field_validator("instructions")(_nonblank)

    @field_validator("criteria")
    @classmethod
    def _criteria(cls, v: list[str]) -> list[str]:
        if not 2 <= len(v) <= MAX_SCORE_LEVELS:
            raise ValueError(f"needs 2 to {MAX_SCORE_LEVELS} levels")
        return v


Question = Annotated[
    NoulQuestion | ChoiceQuestion | ScoreQuestion, Field(discriminator="type")
]
State = Union[str, dict[str, Any], list[str]]


class DecisionRequest(BaseModel):
    model: str
    state: State
    questions: dict[str, Question]

    @field_validator("model")
    @classmethod
    def _model(cls, v: str) -> str:
        return _nonblank(v)

    @field_validator("questions")
    @classmethod
    def _questions(cls, v: dict[str, Any]) -> dict[str, Any]:
        if not 1 <= len(v) <= MAX_QUESTIONS:
            raise ValueError(f"needs 1 to {MAX_QUESTIONS} questions")
        return v

    @model_validator(mode="after")
    def _state_size(self) -> DecisionRequest:
        if len(render_state(self.state)) > MAX_STATE_CHARS:
            raise ValueError(f"state is longer than {MAX_STATE_CHARS} characters")
        return self


def question_key(question: Any) -> str:
    """A stable key for a question's wording, which calibrations are kept
    under: the same question text and options always give the same key."""
    data = question.model_dump(exclude_none=True)
    canon = json.dumps(data, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(canon.encode()).hexdigest()[:32]


# --- Prompts ---


@dataclass(frozen=True, slots=True)
class Prompt:
    messages: list[dict[str, str]]
    # Normalised label token text -> answer key.
    labels: dict[str, str]


def render_state(state: State) -> str:
    if isinstance(state, str):
        return state
    if isinstance(state, dict):
        return json.dumps(state, indent=2, ensure_ascii=False)
    return "\n\n".join(f"[{i}] {s}" for i, s in enumerate(state, 1))


def normalize_label(token: str) -> str:
    """Token text as a label: Llama tokenises ``A`` and `` A`` differently."""
    return token.strip().lower().rstrip(").:")


def build_prompt(
    state: State, question: Any, *, scheme: LabelScheme = "letters"
) -> Prompt:
    """The chat prompt for one question. The state comes first so questions
    on the same state share a prompt prefix in the backend's cache."""
    head = f"State:\n{render_state(state)}\n\n"
    if isinstance(question, NoulQuestion):
        if question.negative:
            # Two situations side by side. Prefixing the negative with
            # "False:" reads as a double negation and inverted 8B answers.
            body = (
                "Question: Which of these describes the state?\n"
                "Options:\n"
                f"A) {question.instructions}\n"
                f"B) {question.negative}\n"
                "Answer with the option letter only."
            )
        else:
            body = (
                "Question: Is the following statement true of the state?\n"
                f"Statement: {question.instructions}\n"
                "Options:\n"
                "A) True: the statement holds for this state\n"
                f"B) False: {DEFAULT_NOUL_NEGATIVE}\n"
                "Answer with the option letter only."
            )
        labels = {"a": NOUL_TRUE, "b": NOUL_FALSE}
    elif isinstance(question, ChoiceQuestion):
        keys = list(question.criteria)
        if scheme == "names":
            opts = "\n".join(f"{k}: {d}" for k, d in question.criteria.items())
            labels = {normalize_label(k): k for k in keys}
            tail = "Answer with the option name only."
        else:
            letters = string.ascii_uppercase[: len(keys)]
            opts = "\n".join(
                f"{letter}) {k}: {question.criteria[k]}"
                for letter, k in zip(letters, keys)
            )
            labels = {letter.lower(): k for letter, k in zip(letters, keys)}
            tail = "Answer with the option letter only."
        body = f"Question: {question.instructions}\nOptions:\n{opts}\n{tail}"
    elif isinstance(question, ScoreQuestion):
        opts = "\n".join(f"{i}: {d}" for i, d in enumerate(question.criteria))
        labels = {str(i): str(i) for i in range(len(question.criteria))}
        body = (
            f"Question: {question.instructions}\nLevels:\n{opts}\n"
            "Answer with the level number only."
        )
    else:
        raise TypeError(f"not a decision question: {question!r}")
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": head + body},
    ]
    return Prompt(messages, labels)


def upstream_body(
    backend_model: str, prompt: Prompt, *, top_logprobs: int = DEFAULT_TOP_LOGPROBS
) -> dict[str, Any]:
    """An OpenAI-style chat completion asking for one token and its logprobs."""
    return {
        "model": backend_model,
        "messages": prompt.messages,
        "max_tokens": 1,
        "temperature": 0,
        "stream": False,
        "logprobs": True,
        "top_logprobs": top_logprobs,
    }


# --- Distributions ---


@dataclass(frozen=True, slots=True)
class Distribution:
    probabilities: dict[str, float]
    # Raw probability on the label tokens before renormalising: how much of
    # the model's answer was one of the offered labels.
    label_mass: float


def top_logprobs_from_response(resp: Mapping[str, Any]) -> list[dict[str, Any]]:
    try:
        content = resp["choices"][0]["logprobs"]["content"]
        return list(content[0]["top_logprobs"])
    except KeyError, IndexError, TypeError:
        raise NoLogprobs("backend response has no first-token logprobs") from None


def distribution(
    top_logprobs: Sequence[Mapping[str, Any]], labels: Mapping[str, str]
) -> Distribution:
    """Probability of each answer key, renormalised over the label tokens.

    Several tokens may spell one label (``A``, `` A``, ``A)``); their
    probabilities add. A label missing from the top logprobs gets zero.
    """
    raw: dict[str, float] = {}
    for t in top_logprobs:
        key = labels.get(normalize_label(str(t.get("token", ""))))
        if key is None:
            continue
        lp = t.get("logprob")
        if lp is None or not math.isfinite(lp):
            continue
        raw[key] = raw.get(key, 0.0) + math.exp(lp)
    mass = sum(raw.values())
    if mass <= 0.0:
        raise DecisionError("no answer label among the backend's top tokens")
    keys = dict.fromkeys(labels.values())
    return Distribution({k: raw.get(k, 0.0) / mass for k in keys}, mass)


# --- Calibration ---

_LOG_FLOOR = math.log(1e-6)


@dataclass(frozen=True, slots=True)
class Calibration:
    """``p'_k ∝ exp(log p_k / temperature + offsets[k])``.

    For a two-way question this is Platt scaling on the log odds. The
    offsets correct a shifted prior (yes bias, middle-level bias), which a
    temperature alone cannot.
    """

    temperature: float = 1.0
    offsets: Mapping[str, float] = field(default_factory=dict)

    def apply(self, probs: Mapping[str, float]) -> dict[str, float]:
        z = {
            k: _log(p) / self.temperature + self.offsets.get(k, 0.0)
            for k, p in probs.items()
        }
        return _softmax(z)

    def to_dict(self) -> dict[str, Any]:
        return {"temperature": self.temperature, "offsets": dict(self.offsets)}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Calibration:
        t = float(data.get("temperature", 1.0))
        if not t > 0:
            raise ValueError("temperature must be positive")
        offsets = {str(k): float(v) for k, v in (data.get("offsets") or {}).items()}
        return cls(t, offsets)


def _log(p: float) -> float:
    return math.log(p) if p > 0 else _LOG_FLOOR


def _softmax(z: Mapping[str, float]) -> dict[str, float]:
    m = max(z.values())
    e = {k: math.exp(v - m) for k, v in z.items()}
    s = sum(e.values())
    return {k: v / s for k, v in e.items()}


def fit_calibration(
    examples: Sequence[tuple[Mapping[str, float], str]],
    *,
    fit_temperature: bool = True,
    iterations: int = 3000,
    learning_rate: float = 0.05,
) -> Calibration:
    """Fit a :class:`Calibration` to (uncalibrated distribution, true key)
    pairs by gradient descent on log loss. All distributions must share keys.
    """
    if not examples:
        raise ValueError("no examples")
    keys = list(examples[0][0])
    if any(list(d) != keys for d, _ in examples) or any(
        y not in keys for _, y in examples
    ):
        raise ValueError("examples must share keys and name one of them as truth")
    logs = [[max(_log(d[k]), _LOG_FLOOR) for k in keys] for d, _ in examples]
    truth = [keys.index(y) for _, y in examples]
    n, scale, w = len(examples), 1.0, [0.0] * len(keys)
    for _ in range(iterations):
        g_scale, g_w = 0.0, [0.0] * len(keys)
        for row, y in zip(logs, truth):
            z = [scale * x + wk for x, wk in zip(row, w)]
            m = max(z)
            e = [math.exp(v - m) for v in z]
            s = sum(e)
            for k, ek in enumerate(e):
                err = ek / s - (1.0 if k == y else 0.0)
                g_w[k] += err
                g_scale += err * row[k]
        w = [wk - learning_rate * gk / n for wk, gk in zip(w, g_w)]
        if fit_temperature:
            scale = max(0.01, scale - learning_rate * g_scale / n)
    mean = sum(w) / len(w)  # offsets are only defined up to a constant
    return Calibration(1.0 / scale, {k: wk - mean for k, wk in zip(keys, w)})


def held_out(
    examples: Sequence[tuple[Mapping[str, float], str]], *, folds: int
) -> list[dict[str, float]]:
    """Each example's distribution calibrated by a fit that did not see it.

    Example ``i`` is held out in fold ``i % folds``; *folds* equal to the
    number of examples is leave-one-out.
    """
    n = len(examples)
    folds = max(2, min(folds, n))
    out: list[dict[str, float]] = [{} for _ in examples]
    for f in range(folds):
        cal = fit_calibration([ex for i, ex in enumerate(examples) if i % folds != f])
        for i in range(f, n, folds):
            out[i] = cal.apply(examples[i][0])
    return out


def scores(
    predictions: Sequence[Mapping[str, float]],
    truth: Sequence[str],
    *,
    ordinal: bool = False,
) -> dict[str, float]:
    """Accuracy of the top answer and mean log loss of the true one; with
    *ordinal* (score questions, keys ``"0"``, ``"1"``...) also the mean
    absolute error of the expected level."""
    n = len(truth)
    acc = sum(max(p, key=p.__getitem__) == y for p, y in zip(predictions, truth))
    loss = -sum(math.log(max(p[y], 1e-6)) for p, y in zip(predictions, truth))
    out = {"accuracy": _r(acc / n), "log_loss": _r(loss / n)}
    if ordinal:
        err = sum(
            abs(sum(int(k) * pk for k, pk in p.items()) - int(y))
            for p, y in zip(predictions, truth)
        )
        out["mae"] = _r(err / n)
    return out


# --- Answers ---


def confidence(probs: Mapping[str, float]) -> float:
    """The top probability rescaled so a uniform answer is 0 and a certain
    one is 1. Reproduces the values in TypeSafe's published examples (1.0 at
    p = 1; 0.5 at p = 0.67 over three levels); their formula is unpublished.
    """
    n = len(probs)
    if n < 2:
        return 1.0
    return max(0.0, (max(probs.values()) - 1.0 / n) / (1.0 - 1.0 / n))


def _r(x: float) -> float:
    return round(x, 4)


def answer(
    question: Any, dist: Distribution, calibration: Calibration | None = None
) -> dict[str, Any]:
    """The answer object for one question, in Jev's response shape."""
    probs = (
        calibration.apply(dist.probabilities)
        if calibration
        else dict(dist.probabilities)
    )
    if isinstance(question, NoulQuestion):
        return {"type": "noul", "noul": _r(probs[NOUL_TRUE])}
    if isinstance(question, ChoiceQuestion):
        ranked = sorted(probs.items(), key=lambda kv: -kv[1])
        return {
            "type": "choice",
            "choice": ranked[0][0],
            "confidence": _r(confidence(probs)),
            "probabilities": {k: _r(p) for k, p in ranked},
        }
    if isinstance(question, ScoreQuestion):
        levels = [str(i) for i in range(len(question.criteria))]
        return {
            "type": "score",
            "score": _r(sum(i * probs[k] for i, k in enumerate(levels))),
            "confidence": _r(confidence(probs)),
            "legend": dict(zip(levels, question.criteria)),
            "probabilities": {k: _r(probs[k]) for k in levels},
        }
    raise TypeError(f"not a decision question: {question!r}")


def answer_keys(question: Any) -> list[str]:
    """The keys of a question's probability distribution, in order."""
    if isinstance(question, NoulQuestion):
        return [NOUL_TRUE, NOUL_FALSE]
    if isinstance(question, ChoiceQuestion):
        return list(question.criteria)
    if isinstance(question, ScoreQuestion):
        return [str(i) for i in range(len(question.criteria))]
    raise TypeError(f"not a decision question: {question!r}")


def usage_from_responses(responses: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    """Sum the upstream chat usage into Jev's ``usage`` shape."""
    inp = out = 0
    for r in responses:
        u = r.get("usage") or {}
        inp += int(u.get("prompt_tokens") or 0)
        out += int(u.get("completion_tokens") or 0)
    return {"input_tokens": inp, "output_tokens": out}
