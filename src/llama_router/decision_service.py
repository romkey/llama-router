"""Serve decision requests on a routed provider.

:mod:`decisions` turns questions into prompts and backend answers into
distributions; this module does the I/O around it:

- **emulated**: each question becomes a one-token chat completion with
  logprobs. The questions of one request go to the same provider at once, one
  backend slot each, and share the state as a prompt prefix.
- **native**: an upstream that serves the decision route itself (LiteLLM in
  front of Jev, say) gets the request as it came, with the model renamed.

Calibrations are read through :class:`CalibrationStore`, an in-memory copy of
the table that is dropped whenever the table changes, so the request path
reads the database at most once after each change.
"""

from __future__ import annotations

import asyncio
import logging
import weakref
from collections.abc import Mapping
from typing import Any

from . import decisions as d
from .database import CHANGE_DECISIONS, Database
from .provider_manager import ProviderManager
from .v1_client import get_v1_client

logger = logging.getLogger(__name__)

_SUMMARY_MAX = 240


class CalibrationStore:
    """Calibrations by (model, question key), cached until the table changes."""

    def __init__(self, db: Database) -> None:
        self._db = db
        self._cache: dict[tuple[str, str], d.Calibration] | None = None
        # Bumped on every change, so a read that a change overtook is not
        # kept as the cache.
        self._generation = 0
        self._lock = asyncio.Lock()
        db.add_change_listener(self._on_change)

    def _on_change(self, reason: str) -> None:
        if reason == CHANGE_DECISIONS:
            self._generation += 1
            self._cache = None

    async def _load(self) -> dict[tuple[str, str], d.Calibration]:
        cache = self._cache
        if cache is not None:
            return cache
        async with self._lock:
            cache = self._cache
            if cache is not None:
                return cache
            generation = self._generation
            built: dict[tuple[str, str], d.Calibration] = {}
            for r in await self._db.list_decision_calibrations():
                try:
                    cal = d.Calibration.from_dict(r)
                except ValueError:
                    logger.warning("Skipping invalid calibration %s", r["id"])
                    continue
                built[(r["model_name"], r["question_key"])] = cal
            # A change while reading may not be in *built*: answer this
            # request with it, but let the next one read again.
            if generation == self._generation:
                self._cache = built
            return built

    async def for_request(
        self, model: str, request: d.DecisionRequest
    ) -> dict[str, d.Calibration]:
        """The calibration of each question in *request* that has one."""
        cache = await self._load()
        out: dict[str, d.Calibration] = {}
        for qid, q in request.questions.items():
            cal = cache.get((model, d.question_key(q)))
            if cal is not None:
                out[qid] = cal
        return out


_stores: weakref.WeakKeyDictionary[Database, CalibrationStore] = (
    weakref.WeakKeyDictionary()
)


def calibrations(db: Database) -> CalibrationStore:
    store = _stores.get(db)
    if store is None:
        store = _stores[db] = CalibrationStore(db)
    return store


async def emulate(
    pm: ProviderManager,
    provider_id: int,
    backend_model: str,
    request: d.DecisionRequest,
    calibration: Mapping[str, d.Calibration] | None = None,
    *,
    acquire: bool = True,
) -> dict[str, Any]:
    """Answer every question of *request* on one provider.

    Returns Jev's response shape without ``model``. Raises
    :class:`decisions.NoLogprobs` when the backend returns no logprobs, and
    :class:`decisions.DecisionError` for an answer that names no label.
    *acquire* false skips the provider's request accounting, for checks
    that must not count as traffic (the auto-benchmarker).
    """
    client = get_v1_client(pm, provider_id)
    calibration = calibration or {}

    async def one(qid: str, q: Any) -> tuple[dict[str, Any], dict[str, Any]]:
        prompt = d.build_prompt(request.state, q)
        body = d.upstream_body(backend_model, prompt)
        if acquire:
            async with pm.acquire_provider(provider_id):
                resp = await client.chat_completions(body)
        else:
            resp = await client.chat_completions(body)
        top = d.top_logprobs_from_response(resp)
        try:
            dist = d.distribution(top, prompt.labels)
        except d.DecisionError as exc:
            raise d.DecisionError(f"question {qid!r}: {exc}") from None
        return d.answer(q, dist, calibration.get(qid)), resp

    items = list(request.questions.items())
    results = await asyncio.gather(
        *(one(qid, q) for qid, q in items), return_exceptions=True
    )
    errors = [r for r in results if isinstance(r, BaseException)]
    if errors:
        # A missing-logprobs answer is about the provider, not one question.
        raise next((e for e in errors if isinstance(e, d.NoLogprobs)), errors[0])
    done: list[tuple[dict[str, Any], dict[str, Any]]] = results  # type: ignore[assignment]
    return {
        "answers": {qid: a for (qid, _q), (a, _r) in zip(items, done)},
        "usage": d.usage_from_responses([r for _a, r in done]),
    }


async def native(
    pm: ProviderManager, provider_id: int, path: str, body: dict[str, Any]
) -> dict[str, Any]:
    """Pass a decision request to an upstream that serves *path* itself."""
    client = get_v1_client(pm, provider_id)
    async with pm.acquire_provider(provider_id):
        return await client.post_json(path, body, op=f"Decisions ({path})")


def summarize(response: Mapping[str, Any], calibrated: int = 0) -> str | None:
    """A request-log line: the number of questions and each answer."""
    answers = response.get("answers")
    if not isinstance(answers, Mapping) or not answers:
        return None
    parts = []
    for qid, a in answers.items():
        if not isinstance(a, Mapping):
            continue
        kind = a.get("type")
        value = a.get(kind) if isinstance(kind, str) else None
        if isinstance(value, float):
            value = round(value, 2)
        parts.append(f"{qid}={value}")
    head = f"questions={len(answers)}"
    if calibrated:
        head += f", calibrated={calibrated}"
    return (head + "; " + ", ".join(parts))[:_SUMMARY_MAX]
