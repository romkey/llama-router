"""Dashboard API for decision-question calibrations.

A calibration belongs to one model and one question wording (see
:func:`llama_router.decisions.question_key`). It is either given directly
(``temperature`` and ``offsets``) or fit here from labelled examples: the
uncalibrated distribution the model gave for a state, and the right answer.
"""

from __future__ import annotations

import math
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import TypeAdapter, ValidationError

from .. import decisions as d
from . import deps

router = APIRouter()

_QUESTION = TypeAdapter(d.Question)
MAX_EXAMPLES = 5000


def parse_question(raw: Any) -> Any:
    try:
        return _QUESTION.validate_python(raw)
    except ValidationError as exc:
        msg = "; ".join(e.get("msg", "") for e in exc.errors(include_url=False))
        raise HTTPException(status_code=400, detail=f"invalid question: {msg}")


def _finite(x: Any, what: str) -> float:
    try:
        v = float(x)
    except (TypeError, ValueError):
        raise HTTPException(status_code=400, detail=f"{what} must be a number")
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
    except ValueError:
        raise HTTPException(status_code=400, detail="JSON body required")
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


@router.delete("/api/decisions/calibrations/{calibration_id}")
async def api_delete_calibration(calibration_id: int):
    if not await deps.get_db().delete_decision_calibration(calibration_id):
        raise HTTPException(status_code=404, detail="calibration not found")
    return JSONResponse({"ok": True})
