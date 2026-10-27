"""Dashboard API for decision-question calibrations and key policies.

A calibration belongs to one model and one question wording (see
:func:`llama_router.decisions.question_key`). It is either given directly
(``temperature`` and ``offsets``) or fit here from labelled examples: the
uncalibrated distribution the model gave for a state, and the right answer.
``/api/decisions/calibrations/fit`` takes labelled *states* instead, asks the
routed model about each, and fits from its answers.

``/api/decisions/try`` answers a decision request as ``/v1/systemone`` does,
for the Decisions page.
"""

from __future__ import annotations

import asyncio
import logging
import math
from typing import Any

import httpx
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import TypeAdapter, ValidationError

from .. import decision_service as svc
from .. import decisions as d
from ..decision_policy import parse_policy
from ..httpx_errors import describe_httpx_error
from ..llamacpp_api import deps as api_deps
from ..llamacpp_api.decisions import serve as serve_decision
from ..routing_table import decides_natively
from . import deps
from .routes import merge_dash_template_ctx, templates

logger = logging.getLogger(__name__)

router = APIRouter()

_QUESTION = TypeAdapter(d.Question)
MAX_EXAMPLES = 5000
# Labelled states for a fit: each is one backend request, and the held-out
# scores refit the calibration up to LOO_MAX times.
MAX_FIT_STATES = 500
MIN_FIT_STATES = 4
LOO_MAX = 50
FIT_FOLDS = 10
FIT_CONCURRENCY = 4


def parse_question(raw: Any) -> Any:
    try:
        return _QUESTION.validate_python(raw)
    except ValidationError as exc:
        msg = "; ".join(e.get("msg", "") for e in exc.errors(include_url=False))
        raise HTTPException(status_code=400, detail=f"invalid question: {msg}") from exc


def _finite(x: Any, what: str) -> float:
    try:
        v = float(x)
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=f"{what} must be a number") from exc
    if not math.isfinite(v):
        raise HTTPException(status_code=400, detail=f"{what} must be finite")
    return v


def fit_from_examples(question: Any, raw: Any) -> d.Calibration:
    """Fit from ``[{"probabilities": {key: p}, "answer": key}, ...]``."""
    if not isinstance(raw, list) or not raw:
        raise HTTPException(status_code=400, detail="examples must be a list")
    if len(raw) > MAX_EXAMPLES:
        raise HTTPException(status_code=400, detail=f"at most {MAX_EXAMPLES} examples")
    keys = d.answer_keys(question)
    examples = []
    for i, ex in enumerate(raw):
        if not isinstance(ex, dict) or not isinstance(ex.get("probabilities"), dict):
            raise HTTPException(
                status_code=400, detail=f"example {i}: probabilities required"
            )
        probs = ex["probabilities"]
        if set(probs) != set(keys):
            raise HTTPException(
                status_code=400,
                detail=f"example {i}: probabilities must have keys {keys}",
            )
        answer = str(ex.get("answer"))
        if answer not in keys:
            raise HTTPException(
                status_code=400, detail=f"example {i}: answer must be one of {keys}"
            )
        examples.append(({k: _finite(probs[k], "probability") for k in keys}, answer))
    return d.fit_calibration(examples)


def calibration_from_body(
    question: Any, body: dict[str, Any]
) -> tuple[d.Calibration, int | None]:
    if "examples" in body:
        examples = body["examples"]
        return fit_from_examples(question, examples), len(examples)
    keys = set(d.answer_keys(question))
    offsets_raw = body.get("offsets") or {}
    if not isinstance(offsets_raw, dict) or not set(offsets_raw) <= keys:
        raise HTTPException(
            status_code=400, detail=f"offsets keys must be among {sorted(keys)}"
        )
    temperature = _finite(body.get("temperature", 1.0), "temperature")
    if temperature <= 0:
        raise HTTPException(status_code=400, detail="temperature must be positive")
    offsets = {str(k): _finite(v, "offset") for k, v in offsets_raw.items()}
    return d.Calibration(temperature, offsets), None


@router.get("/decisions", response_class=HTMLResponse)
async def decisions_page(request: Request):
    """Calibrate models for decision questions and try questions out."""
    models = [m["name"] for m in await deps.get_db().list_all_models()]
    return templates.TemplateResponse(
        request,
        "decisions.html",
        merge_dash_template_ctx(
            request,
            {
                "models": models,
                "min_fit_states": MIN_FIT_STATES,
                "max_fit_states": MAX_FIT_STATES,
            },
        ),
    )


@router.get("/api/decisions/calibrations")
async def api_list_calibrations():
    rows = await deps.get_db().list_decision_calibrations()
    return JSONResponse({"calibrations": rows})


@router.post("/api/decisions/calibrations")
async def api_set_calibration(request: Request):
    """Store a calibration for ``model`` and ``question``.

    The body carries either ``temperature`` and ``offsets``, or
    ``examples`` to fit them from.
    """
    try:
        body = await request.json()
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="JSON body required") from exc
    if not isinstance(body, dict):
        raise HTTPException(status_code=400, detail="JSON object required")
    model = str(body.get("model") or "").strip()
    if not model:
        raise HTTPException(status_code=400, detail="model is required")
    question = parse_question(body.get("question"))
    cal, examples = calibration_from_body(question, body)
    key = d.question_key(question)
    await deps.get_db().set_decision_calibration(
        model,
        key,
        question.type,
        question.model_dump(exclude_none=True),
        cal.temperature,
        dict(cal.offsets),
        examples,
    )
    return JSONResponse(
        {"model": model, "question_key": key, **cal.to_dict(), "examples": examples}
    )


def _label(question: Any, raw: Any, i: int) -> str:
    """A labelled answer as the question's answer key: ``true``/``false``
    for noul, an option key (any case) for choice, a level number for score."""
    keys = d.answer_keys(question)
    if isinstance(question, d.NoulQuestion) and isinstance(raw, bool):
        return d.NOUL_TRUE if raw else d.NOUL_FALSE
    if isinstance(question, d.ScoreQuestion) and isinstance(raw, int):
        raw = str(raw)
    if isinstance(raw, str):
        by_lower = {k.lower(): k for k in keys}
        key = by_lower.get(raw.strip().lower())
        if key is not None:
            return key
    raise HTTPException(
        status_code=400, detail=f"example {i}: answer must be one of {keys}"
    )


def _labelled_states(question: Any, raw: Any) -> list[tuple[Any, str]]:
    if not isinstance(raw, list) or len(raw) < MIN_FIT_STATES:
        raise HTTPException(
            status_code=400,
            detail=f"examples must be a list of at least {MIN_FIT_STATES}",
        )
    if len(raw) > MAX_FIT_STATES:
        raise HTTPException(
            status_code=400, detail=f"at most {MAX_FIT_STATES} examples"
        )
    out = []
    for i, ex in enumerate(raw):
        if not isinstance(ex, dict) or "state" not in ex:
            raise HTTPException(status_code=400, detail=f"example {i}: state required")
        try:
            # Validates the state's type and size as a request would.
            req = d.DecisionRequest(
                model="-", state=ex["state"], questions={"q": question}
            )
        except ValidationError as exc:
            msg = "; ".join(e.get("msg", "") for e in exc.errors(include_url=False))
            raise HTTPException(status_code=400, detail=f"example {i}: {msg}") from exc
        out.append((req.state, _label(question, ex.get("answer"), i)))
    return out


async def _distributions(
    provider_id: int, backend_model: str, question: Any, states: list[Any]
) -> list[d.Distribution]:
    asker = svc.Asker(api_deps.get_pm(), provider_id, backend_model)
    gate = asyncio.Semaphore(FIT_CONCURRENCY)

    async def one(i: int, state: Any) -> d.Distribution:
        async with gate:
            try:
                dist, _resp = await asker.ask(state, question)
                return dist
            except d.NoLogprobsError:
                raise
            except d.DecisionError as exc:
                raise d.DecisionError(f"example {i}: {exc}") from None

    results = await asyncio.gather(
        *(one(i, s) for i, s in enumerate(states)), return_exceptions=True
    )
    errors = [r for r in results if isinstance(r, BaseException)]
    if errors:
        raise next((e for e in errors if isinstance(e, d.NoLogprobsError)), errors[0])
    return results  # type: ignore[return-value]


def _fit_and_score(
    examples: list[tuple[dict[str, float], str]], ordinal: bool
) -> tuple[d.Calibration, dict[str, Any]]:
    """The calibration fit on every example, and before/after scores with
    each example scored by a fit that did not see it."""
    n = len(examples)
    folds = n if n <= LOO_MAX else FIT_FOLDS
    truth = [y for _p, y in examples]
    before = d.scores([p for p, _y in examples], truth, ordinal=ordinal)
    after = d.scores(d.held_out(examples, folds=folds), truth, ordinal=ordinal)
    report = {
        "method": "leave-one-out" if folds == n else f"{folds}-fold",
        "before": before,
        "after": after,
    }
    return d.fit_calibration(examples), report


@router.post("/api/decisions/calibrations/fit")
async def api_fit_calibration(request: Request):
    """Fit and store a calibration from labelled states.

    The body is ``{model, question, examples: [{state, answer}]}``. Each
    state is asked of the provider the model routes to, uncalibrated; the
    response reports accuracy and log loss before and after calibration,
    cross-validated so the after numbers are not scored on their own fit.
    """
    try:
        body = await request.json()
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="JSON body required") from exc
    if not isinstance(body, dict):
        raise HTTPException(status_code=400, detail="JSON object required")
    model = str(body.get("model") or "").strip()
    if not model:
        raise HTTPException(status_code=400, detail="model is required")
    question = parse_question(body.get("question"))
    labelled = _labelled_states(question, body.get("examples"))

    result = await api_deps.get_router().route(model, "/v1/systemone")
    if not result:
        raise HTTPException(
            status_code=404, detail=f"No available provider for model '{model}'"
        )
    provider = result.provider
    assert provider.id is not None
    # A provider that answers decisions itself never applies a calibration.
    if decides_natively(provider, "/v1/systemone"):
        raise HTTPException(
            status_code=400,
            detail=(
                f"Provider '{provider.name}' answers decisions itself, so its "
                "answers cannot be calibrated here"
            ),
        )
    try:
        dists = await _distributions(
            provider.id, result.backend_model, question, [s for s, _y in labelled]
        )
    except d.NoLogprobsError as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Provider '{provider.name}' returned no logprobs",
        ) from exc
    except d.DecisionError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except httpx.HTTPError as exc:
        detail = describe_httpx_error(exc)
        logger.error("Calibration fit on %s failed: %s", provider.name, detail)
        raise HTTPException(
            status_code=502, detail=f"upstream request failed: {detail}"
        ) from exc

    examples = [
        (dist.probabilities, y) for dist, (_s, y) in zip(dists, labelled, strict=False)
    ]
    cal, report = await asyncio.to_thread(
        _fit_and_score, examples, isinstance(question, d.ScoreQuestion)
    )
    key = d.question_key(question)
    await deps.get_db().set_decision_calibration(
        result.model_key,
        key,
        question.type,
        question.model_dump(exclude_none=True),
        cal.temperature,
        dict(cal.offsets),
        len(examples),
    )
    return JSONResponse(
        {
            "model": result.model_key,
            "provider": provider.name,
            "question_key": key,
            **cal.to_dict(),
            "examples": len(examples),
            "label_mass": round(sum(dist.label_mass for dist in dists) / len(dists), 4),
            **report,
        }
    )


@router.post("/api/decisions/try")
async def api_try_decision(request: Request):
    """Answer a decision request as ``/v1/systemone`` would."""
    try:
        body = await request.json()
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="JSON body required") from exc
    return await serve_decision(request, body, "/v1/systemone", None)


@router.delete("/api/decisions/calibrations/{calibration_id}")
async def api_delete_calibration(calibration_id: int):
    if not await deps.get_db().delete_decision_calibration(calibration_id):
        raise HTTPException(status_code=404, detail="calibration not found")
    return JSONResponse({"ok": True})


# --- Decision policies per API key (see llama_router.decision_policy) ---


async def _require_key(db: Any, key_id: int) -> None:
    if not any(int(k["id"]) == key_id for k in await db.list_api_keys()):
        raise HTTPException(status_code=404, detail="API key not found")


@router.get("/api/keys/{key_id}/decision-policy")
async def api_get_key_policy(key_id: int):
    db = deps.get_db()
    await _require_key(db, key_id)
    return JSONResponse(
        {"id": key_id, "policy": await db.get_api_key_decision_policy(key_id)}
    )


@router.put("/api/keys/{key_id}/decision-policy")
async def api_set_key_policy(key_id: int, request: Request):
    """Set the key's policy: ``{"model", "tier"?, "gates"?}``.

    ``tier`` is ``{"question": <score question>, "targets": {level: model},
    "models"?: [requested model, ...]}``; ``gates`` maps a name to
    ``{"question": <noul question>, "threshold": p}``.
    """
    db = deps.get_db()
    await _require_key(db, key_id)
    try:
        body = await request.json()
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="JSON body required") from exc
    try:
        policy = parse_policy(body)
    except ValidationError as exc:
        msg = "; ".join(
            ".".join(str(p) for p in e.get("loc", ())) + ": " + e.get("msg", "")
            for e in exc.errors(include_url=False)
        )
        raise HTTPException(status_code=400, detail=f"invalid policy: {msg}") from exc
    stored = policy.model_dump(exclude_none=True, exclude_defaults=True)
    await db.set_api_key_decision_policy(key_id, stored)
    return JSONResponse({"id": key_id, "policy": stored})


@router.delete("/api/keys/{key_id}/decision-policy")
async def api_delete_key_policy(key_id: int):
    db = deps.get_db()
    await _require_key(db, key_id)
    if not await db.delete_api_key_decision_policy(key_id):
        raise HTTPException(status_code=404, detail="API key has no decision policy")
    return JSONResponse({"ok": True})
