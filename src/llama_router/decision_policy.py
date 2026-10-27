"""Per-key decision policies: tiered routing and gating.

An API key may carry a policy that asks decision questions (see
:mod:`decisions`) about each generation request before it is routed:

- **tier**: a ``score`` question, typically a difficulty rubric. The most
  probable level picks the model the request is routed to, from the
  policy's map of level to model.
- **gates**: ``noul`` questions with a threshold each. A request whose
  probability reaches a gate's threshold is rejected with 403.

All of a policy's questions go in one decision request, so they cost one
round trip, answered in parallel on the provider the router picks for the
policy's decision model. A key without a policy pays nothing. The policy
never fails a request on its own account: when the decision cannot be made
in time, the request goes on with the model it asked for and no gate.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import TYPE_CHECKING, Any

import httpx
from fastapi import HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from . import decision_service as svc
from . import decisions as d
from .config import settings
from .database import Database
from .httpx_errors import describe_httpx_error
from .provider_manager import ProviderManager
from .request_logger import log_request_background
from .routing_table import decides_natively

if TYPE_CHECKING:
    from .router import Router, RoutingPreferences

logger = logging.getLogger(__name__)

DECISION_PATH = "/v1/decisions"
TIER_ID = "tier"
GATE_PREFIX = "gate:"
# The state sent for a request: the tail of its last user message.
MAX_STATE_CHARS = 16_000
# request.state attribute the request log reads the outcome from.
STATE_ATTR = "decision_policy"


class Tier(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: d.ScoreQuestion
    # Level ("0", "1", ...) to the model a request at that level goes to. A
    # level without a target keeps the requested model.
    targets: dict[str, str]
    # Requested models the tier applies to; empty means every model. A
    # virtual name such as "auto" (with a model fallback for when the
    # decision fails) leaves clients that name a model alone.
    models: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _targets(self) -> Tier:
        levels = {str(i) for i in range(len(self.question.criteria))}
        unknown = sorted(set(self.targets) - levels)
        if unknown:
            raise ValueError(f"targets name unknown levels {unknown}")
        if any(not str(m).strip() for m in self.targets.values()):
            raise ValueError("target models must not be blank")
        if not self.targets:
            raise ValueError("needs at least one target")
        return self


class Gate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: d.NoulQuestion
    # Reject when the question's probability is at or above this.
    threshold: float = Field(gt=0.0, le=1.0)


class DecisionPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # The decision model, routed like any request to /v1/decisions.
    model: str
    tier: Tier | None = None
    gates: dict[str, Gate] = Field(default_factory=dict)

    @field_validator("model")
    @classmethod
    def _model(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("must not be blank")
        return v.strip()

    @field_validator("gates")
    @classmethod
    def _gates(cls, v: dict[str, Gate]) -> dict[str, Gate]:
        if any(not k.strip() for k in v):
            raise ValueError("gate names must not be blank")
        if len(v) >= d.MAX_QUESTIONS:
            raise ValueError(f"at most {d.MAX_QUESTIONS - 1} gates")
        return v

    @model_validator(mode="after")
    def _something(self) -> DecisionPolicy:
        if self.tier is None and not self.gates:
            raise ValueError("needs a tier or at least one gate")
        return self

    def questions_for(self, model: str) -> dict[str, Any]:
        """The questions to ask for a request for *model*, by question id."""
        out: dict[str, Any] = {}
        tier = self.tier
        if tier is not None and (not tier.models or model in tier.models):
            out[TIER_ID] = tier.question
        for name, gate in self.gates.items():
            out[GATE_PREFIX + name] = gate.question
        return out


def parse_policy(raw: Any) -> DecisionPolicy:
    """Validate a stored or submitted policy; raises pydantic's ValidationError."""
    return DecisionPolicy.model_validate(raw)


# --- The state of a request ---


def _text(content: Any) -> str | None:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [
            p["text"]
            for p in content
            if isinstance(p, dict) and isinstance(p.get("text"), str)
        ]
        return "\n".join(parts) if parts else None
    return None


def request_text(body: Any) -> str | None:
    """The last user message of a generation request, or its prompt or input.

    Covers chat messages (OpenAI and Ollama), completion and generate
    prompts, and Responses API input.
    """
    if not isinstance(body, dict):
        return None
    text: str | None = None
    messages = body.get("messages")
    inp = body.get("input")
    prompt = body.get("prompt")
    if isinstance(messages, list):
        for m in reversed(messages):
            if isinstance(m, dict) and m.get("role") == "user":
                text = _text(m.get("content"))
                break
    elif isinstance(inp, str):
        text = inp
    elif isinstance(inp, list):
        for item in reversed(inp):
            if isinstance(item, dict) and item.get("role") == "user":
                text = _text(item.get("content"))
                break
    elif isinstance(prompt, str):
        text = prompt
    elif isinstance(prompt, list) and prompt and isinstance(prompt[-1], str):
        text = prompt[-1]
    if text is None or not text.strip():
        return None
    return text[-MAX_STATE_CHARS:]


# --- The request path ---


class _Failed(Exception):
    """The decision could not be made; the request goes on without it."""


def _note(request: Request, note: str) -> None:
    setattr(request.state, STATE_ATTR, note)


async def _decide(
    rt: Router,
    pm: ProviderManager,
    db: Database,
    prefs: RoutingPreferences | None,
    request: Request,
    req: d.DecisionRequest,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Answer *req* on a routed provider: the answers and the log fields.

    Failures are logged here and raised as :class:`_Failed`.
    """
    for attempt in range(2):
        result = await rt.route(req.model, DECISION_PATH, prefs)
        if not result:
            raise _Failed(f"no available provider for decision model {req.model!r}")
        provider = result.provider
        assert provider.id is not None
        start = time.monotonic()
        log: dict[str, Any] = dict(
            provider=provider,
            protocol="v1",
            endpoint=DECISION_PATH,
            request=request,
            model=req.model,
            routed_model=result.model_key,
            request_body=req.model_dump(exclude_none=True),
        )

        def failed(detail: str, **extra: Any) -> _Failed:
            log_request_background(
                db,
                **log,
                duration_ms=(time.monotonic() - start) * 1000,
                status="error",
                error_detail=f"decision policy: {detail}"[:500],
                **extra,
            )
            return _Failed(f"{detail} (provider {provider.name})")

        try:
            if decides_natively(provider, DECISION_PATH):
                body = req.model_dump(exclude_none=True)
                body["model"] = result.backend_model
                out = await svc.native(pm, provider.id, DECISION_PATH, body)
            else:
                cal = await svc.calibrations(db).for_request(result.model_key, req)
                out = await svc.emulate(pm, provider.id, result.backend_model, req, cal)
                pm.stats.record_logprobs(provider.id, ok=True)
        except d.NoLogprobs:
            pm.stats.record_logprobs(provider.id, ok=False)
            exc = failed("backend returned no logprobs", sample=False)
            if attempt == 0:
                await rt.rebuild()
                continue
            raise exc
        except d.DecisionError as e:
            raise failed(str(e), sample=False)
        except httpx.HTTPStatusError as e:
            raise failed(f"HTTP {e.response.status_code}: {e.response.text[:300]}")
        except httpx.HTTPError as e:
            raise failed(describe_httpx_error(e))
        log["duration_ms"] = (time.monotonic() - start) * 1000
        log["response_body"] = out
        answers = out.get("answers") if isinstance(out, dict) else None
        if not isinstance(answers, dict):
            raise failed("decision response has no answers")
        return answers, log
    raise AssertionError("unreachable")  # pragma: no cover


def _noul(answers: dict[str, Any], qid: str) -> float:
    a = answers.get(qid)
    p = a.get("noul") if isinstance(a, dict) else None
    if not isinstance(p, (int, float)):
        raise _Failed(f"no probability for {qid!r}")
    return float(p)


def _level(answers: dict[str, Any]) -> str:
    """The most probable level of the tier question."""
    a = answers.get(TIER_ID)
    if not isinstance(a, dict):
        raise _Failed("no answer for the tier question")
    probs = a.get("probabilities")
    if isinstance(probs, dict) and probs:
        return str(max(probs, key=lambda k: float(probs[k])))
    score = a.get("score")
    if isinstance(score, (int, float)):
        return str(round(score))
    raise _Failed("no level for the tier question")


async def apply_decision_policy(
    request: Request,
    body: Any,
    model: str,
    prefs: RoutingPreferences | None,
    *,
    rt: Router,
    pm: ProviderManager,
    db: Database,
) -> str:
    """Run the key's decision policy on a generation request.

    Returns the model to route (the tier's target, or *model*), or raises
    403 when a gate rejects the request. Without a policy it returns
    *model* at once.
    """
    policy: DecisionPolicy | None = prefs.decision_policy if prefs else None
    if policy is None:
        return model
    questions = policy.questions_for(model)
    if not questions:
        return model
    state = request_text(body)
    if state is None:
        _note(request, "policy: skipped, no user text")
        return model

    start = time.monotonic()
    req = d.DecisionRequest(model=policy.model, state=state, questions=questions)
    timeout = settings.decision_policy_timeout_seconds
    try:
        answers, log = await asyncio.wait_for(
            _decide(rt, pm, db, prefs, request, req), timeout
        )
        gates = {name: _noul(answers, GATE_PREFIX + name) for name in policy.gates}
        level = _level(answers) if TIER_ID in questions else None
    except Exception as exc:
        ms = (time.monotonic() - start) * 1000
        if isinstance(exc, asyncio.TimeoutError):
            reason = f"timed out after {timeout:g}s"
            log_request_background(
                db,
                provider=None,
                protocol="v1",
                endpoint=DECISION_PATH,
                request=request,
                model=policy.model,
                request_body=req.model_dump(exclude_none=True),
                duration_ms=ms,
                status="error",
                error_detail=f"decision policy: {reason}",
                sample=False,
            )
        elif isinstance(exc, _Failed):
            reason = str(exc)
        else:
            logger.exception("Decision policy failed")
            reason = f"{type(exc).__name__}: {exc}"
        logger.warning(
            "Decision policy (key %s) failed after %.0f ms, routing %r as asked: %s",
            prefs.key_id if prefs else None,
            ms,
            model,
            reason,
        )
        _note(request, f"policy: failed in {ms:.0f}ms")
        return model

    ms = (time.monotonic() - start) * 1000
    parts = [f"policy {ms:.0f}ms"]
    target = model
    if level is not None:
        assert policy.tier is not None
        target = policy.tier.targets.get(level, "").strip() or model
        parts.append(f"tier={level} {model}->{target}")
    parts += [f"{name}={p:.2f}" for name, p in gates.items()]
    rejected = next(
        (n for n, p in gates.items() if p >= policy.gates[n].threshold), None
    )
    if rejected:
        parts.append(f"rejected by {rejected}")
    note = ", ".join(parts)[:240]
    log_request_background(db, **log, request_meta=note)
    _note(request, note)

    if rejected:
        p = gates[rejected]
        threshold = policy.gates[rejected].threshold
        logger.warning(
            "Decision policy (key %s) rejected a request: gate %r p=%.3f >= %.3f",
            prefs.key_id if prefs else None,
            rejected,
            p,
            threshold,
        )
        raise HTTPException(
            status_code=403,
            detail={
                "error": "rejected by the API key's decision policy",
                "gate": rejected,
                "probability": round(p, 4),
                "threshold": threshold,
            },
        )
    if target != model:
        logger.info(
            "Decision policy (key %s) routed %r to %r (level %s) in %.0f ms",
            prefs.key_id if prefs else None,
            model,
            target,
            level,
            ms,
        )
    return target
