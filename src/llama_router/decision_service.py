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
from collections.abc import Iterable, Mapping, Sequence
from typing import TYPE_CHECKING, Any

import httpx

from . import decisions as d
from .database import CHANGE_DECISIONS, Database
from .provider_manager import ProviderManager
from .routing_table import alias_groups, native_decision_path
from .v1_client import get_v1_client
from .worker_link.protocol import refused as worker_refused

if TYPE_CHECKING:
    from .models import Provider
    from .router import RouteResult

logger = logging.getLogger(__name__)

_SUMMARY_MAX = 240
# Request fields a server may refuse, and the statuses it refuses them with.
_LOGPROB_FIELDS = ("logprobs", "top_logprobs")
_REJECTED = (400, 422)


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
        self, models: str | Sequence[str], request: d.DecisionRequest
    ) -> dict[str, d.Calibration]:
        """The calibration of each question in *request* that has one.

        *models* are names the model goes by, best first (see
        :func:`model_names`): a calibration saved under any of them applies,
        the first name that has one winning.
        """
        names = [models] if isinstance(models, str) else list(models)
        cache = await self._load()
        out: dict[str, d.Calibration] = {}
        for qid, q in request.questions.items():
            key = d.question_key(q)
            for name in names:
                cal = cache.get((name, key))
                if cal is not None:
                    out[qid] = cal
                    break
        return out


def model_names(
    first: Iterable[str | None], aliases: Mapping[str, str] | None = None
) -> list[str]:
    """*first* in order, then every alias-group member of those names, for
    finding a calibration saved under any name the model goes by."""
    names = list(dict.fromkeys(n for n in first if n))
    if aliases:
        groups = alias_groups(dict(aliases))
        for n in list(names):
            names += sorted(groups.get(n, frozenset()) - set(names))
    return names


_stores: weakref.WeakKeyDictionary[Database, CalibrationStore] = (
    weakref.WeakKeyDictionary()
)


def calibrations(db: Database) -> CalibrationStore:
    store = _stores.get(db)
    if store is None:
        store = _stores[db] = CalibrationStore(db)
    return store


class Asker:
    """Asks one provider decision questions, one chat completion each.

    A server that refuses the logprobs fields answers 400 or 422, as does one
    whose context the prompt overflows. On such an answer the asker asks once
    more without them to tell the two apart; every question it asks shares
    that probe. *acquire* false skips the provider's request accounting.
    """

    def __init__(
        self,
        pm: ProviderManager,
        provider_id: int,
        backend_model: str,
        *,
        acquire: bool = True,
    ) -> None:
        self._pm = pm
        self._provider_id = provider_id
        self._client = get_v1_client(pm, provider_id)
        self._backend_model = backend_model
        self._acquire = acquire
        self._probe: asyncio.Task[bool] | None = None

    async def _send(self, body: dict[str, Any]) -> dict[str, Any]:
        if self._acquire:
            async with self._pm.acquire_provider(self._provider_id):
                return await self._client.chat_completions(body)
        return await self._client.chat_completions(body)

    async def _accepts_without_logprobs(self, body: dict[str, Any]) -> bool:
        plain = {k: v for k, v in body.items() if k not in _LOGPROB_FIELDS}
        try:
            await self._send(plain)
        except httpx.HTTPStatusError:
            return False
        return True

    async def ask(
        self, state: d.State, question: Any
    ) -> tuple[d.Distribution, dict[str, Any]]:
        """The question's uncalibrated distribution and the raw response.

        Raises :class:`decisions.NoLogprobs` when the backend refuses or
        omits logprobs, and :class:`decisions.DecisionError` for an answer
        that names no label.
        """
        prompt = d.build_prompt(state, question)
        body = d.upstream_body(self._backend_model, prompt)
        try:
            resp = await self._send(body)
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code not in _REJECTED:
                raise
            if self._probe is None:
                self._probe = asyncio.ensure_future(
                    self._accepts_without_logprobs(body)
                )
            if await self._probe:
                raise d.NoLogprobs(
                    f"backend rejected logprobs (HTTP {exc.response.status_code})"
                ) from exc
            raise
        top = d.top_logprobs_from_response(resp)
        return d.distribution(top, prompt.labels), resp


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
    asker = Asker(pm, provider_id, backend_model, acquire=acquire)
    calibration = calibration or {}

    async def one(qid: str, q: Any) -> tuple[dict[str, Any], dict[str, Any]]:
        try:
            dist, resp = await asker.ask(request.state, q)
        except d.NoLogprobs:
            raise
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


async def pass_through(
    pm: ProviderManager, provider: Provider, path: str, body: dict[str, Any]
) -> dict[str, Any] | None:
    """Send the request on as it came; None when a worker will not relay it.

    A worker older than decision support refuses the route with 403. Its
    stored probe result can still say otherwise, so the request is emulated
    instead and the worker is probed again.
    """
    assert provider.id is not None
    try:
        return await native(pm, provider.id, path, body)
    except httpx.HTTPStatusError as exc:
        resp = exc.response
        if not (provider.is_worker and worker_refused(resp.status_code, resp.text)):
            raise
    logger.info(
        "Worker %s does not relay %s; answering from logprobs instead",
        provider.name,
        path,
    )
    pm.schedule_probe(provider.id, "worker refused decisions")
    return None


async def answer_routed(
    pm: ProviderManager,
    db: Database,
    result: RouteResult,
    request: d.DecisionRequest,
    path: str,
    body: Mapping[str, Any],
    aliases: Mapping[str, str] | None = None,
) -> tuple[dict[str, Any], int]:
    """Answer *request* on the provider *result* routed it to.

    Passed through to the upstream's own decision route when it serves
    one, else emulated with the calibrations stored under any name or
    alias of the model. Returns the response and the number of calibrated
    questions. Raises like :func:`emulate` and :func:`native`.
    """
    provider = result.provider
    assert provider.id is not None
    native_path = native_decision_path(provider, path)
    if native_path is not None:
        out = await pass_through(
            pm, provider, native_path, {**body, "model": result.backend_model}
        )
        if out is not None:
            return out, 0
    names = model_names(
        (result.model_key, result.resolved_model, request.model, result.backend_model),
        aliases,
    )
    cal = await calibrations(db).for_request(names, request)
    answered = await emulate(pm, provider.id, result.backend_model, request, cal)
    pm.stats.record_logprobs(provider.id, ok=True)
    return {"model": result.resolved_model, **answered}, len(cal)
