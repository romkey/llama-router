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
policy's decision model. A key without a policy pays nothing.

When the decision cannot be made (no provider, an error, a timeout), a policy
with gates refuses the request with 503, since an unchecked request is what the
gates are there to stop; ``decision_policy_fail_open`` lets it through instead.
A policy with only a tier goes on with the model the request asked for.
"""

from __future__ import annotations

import asyncio
import json
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

if TYPE_CHECKING:
    from .router import Router, RoutingPreferences

logger = logging.getLogger(__name__)

DECISION_PATH = "/v1/decisions"
TIER_ID = "tier"
GATE_PREFIX = "gate:"
# Most request text sent as a request's state; the newest is kept.
MAX_STATE_CHARS = 16_000
# Characters a turn costs in the rendered state beyond its text ("[12] " and
# the blank line between turns).
_TURN_OVERHEAD = 8
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
        # One question is left for the tier.
        most = settings.decision_max_questions - 1
        if len(v) > most:
            raise ValueError(f"at most {most} gates")
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


def _arguments(value: Any) -> str | None:
    """Tool-call arguments as text: a JSON string as sent, or an object."""
    if isinstance(value, str):
        return value
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    return None


def _part_text(part: Any) -> str | None:
    if not isinstance(part, dict):
        return None
    text = part.get("text")
    if isinstance(text, str):
        return text
    kind = part.get("type")
    if kind == "tool_use":  # Anthropic: a call the assistant made
        return _call_text(part.get("name"), part.get("input"))
    if kind == "tool_result":  # Anthropic: what the tool returned
        return _text(part.get("content"))
    return None


def _text(content: Any) -> str | None:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [t for t in map(_part_text, content) if t is not None]
        return "\n".join(parts) if parts else None
    return None


def _call_text(name: Any, arguments: Any) -> str | None:
    args = _arguments(arguments)
    if args is None and not isinstance(name, str):
        return None
    return f"call {name if isinstance(name, str) else ''}({args or ''})"


def _calls(m: dict[str, Any]) -> list[str]:
    """A message's tool calls as text: chat ``tool_calls`` (OpenAI, Ollama),
    the older ``function_call`` and a Responses ``function_call`` item.
    Their arguments are text a client or a model wrote, so gates read them.
    """
    calls: list[Any] = []
    tool_calls = m.get("tool_calls")
    if isinstance(tool_calls, list):
        calls += [c.get("function", c) for c in tool_calls if isinstance(c, dict)]
    calls.append(m.get("function_call"))
    if m.get("type") == "function_call":
        calls.append(m)
    out = []
    for c in calls:
        if isinstance(c, dict):
            text = _call_text(c.get("name"), c.get("arguments"))
            if text is not None:
                out.append(text)
    return out


def _turns(body: dict[str, Any]) -> list[str]:
    """Every turn's text, oldest first, from messages of any role.

    A user turn is its text alone; any other is prefixed with its role
    (lower-cased), so the decision model can tell a system prompt or a tool
    result from what the user wrote. A gate has to see them all: text meant
    to get past it can sit in a system prompt, an assistant turn or a tool
    result as well as in a user message.
    """
    turns: list[str] = []
    for field in ("system", "instructions"):  # Ollama, Anthropic; Responses
        text = _text(body.get(field))
        if text is not None and text.strip():
            turns.append(f"system: {text}")
    messages = body.get("messages")
    inp = body.get("input")
    prompt = body.get("prompt")
    if isinstance(messages, list):
        items: list[Any] = messages
    elif isinstance(inp, list):
        items = inp
    elif isinstance(inp, str):
        return [*turns, inp]
    elif isinstance(prompt, str):
        return [*turns, prompt]
    elif isinstance(prompt, list):
        return turns + [p for p in prompt if isinstance(p, str)]
    else:
        return turns
    for m in items:
        if not isinstance(m, dict):
            continue
        role = m.get("role")
        label = role.strip().lower() if isinstance(role, str) else ""
        text = _text(m.get("content"))
        if text is None:
            # Responses API tool results: {"type": "function_call_output", ...}
            text = _text(m.get("output"))
        text = "\n".join(t for t in (text, *_calls(m)) if t)
        if not text.strip():
            continue
        if label != "user":
            kind = m.get("type")
            label = label or (kind if isinstance(kind, str) else "") or "message"
            text = f"{label}: {text}"
        turns.append(text)
    return turns


def request_text(body: Any) -> str | list[str] | None:
    """The text of a generation request: every message of any role, or its
    system prompt and prompt or input, oldest first.

    Every turn goes in, not just the last user message, so a gate also sees
    text placed in an earlier turn, a system prompt or a tool result. Past
    MAX_STATE_CHARS (or half the decision request limit, if lower) the
    oldest text is dropped. One turn is returned as a string, several as a
    list. Covers chat messages (OpenAI, Ollama and Anthropic), completion
    and generate prompts, and Responses API input.
    """
    if not isinstance(body, dict):
        return None
    kept: list[str] = []
    budget = min(MAX_STATE_CHARS, settings.decision_max_state_chars // 2)
    for turn in reversed(_turns(body)):
        if not turn.strip():
            continue
        if budget <= 0:
            break
        kept.append(turn[-budget:])
        budget -= len(kept[-1]) + _TURN_OVERHEAD
    if not kept:
        return None
    kept.reverse()
    return kept[0] if len(kept) == 1 else kept


# --- The request path ---


class _FailedError(Exception):
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

    Failures are logged here and raised as :class:`_FailedError`.
    """
    for attempt in range(2):
        result = await rt.route(req.model, DECISION_PATH, prefs)
        if not result:
            raise _FailedError(
                f"no available provider for decision model {req.model!r}"
            )
        # Raises 429, which the caller passes on rather than failing over.
        await svc.check_spend_cap(db, rt, prefs, result, req)
        provider = result.provider
        provider_id = provider.require_id()
        start = time.monotonic()
        log: dict[str, Any] = {
            "provider": provider,
            "protocol": "v1",
            "endpoint": DECISION_PATH,
            "request": request,
            "model": req.model,
            "routed_model": result.model_key,
            "request_body": req.model_dump(exclude_none=True),
        }

        def failed(
            detail: str,
            *,
            log: dict[str, Any] = log,
            start: float = start,
            provider_name: str = provider.name,
            **extra: Any,
        ) -> _FailedError:
            log_request_background(
                db,
                **log,
                duration_ms=(time.monotonic() - start) * 1000,
                status="error",
                error_detail=f"decision policy: {detail}"[:500],
                **extra,
            )
            return _FailedError(f"{detail} (provider {provider_name})")

        try:
            inputs = rt.inputs
            out, _calibrated = await svc.answer_routed(
                pm,
                db,
                result,
                req,
                DECISION_PATH,
                req.model_dump(exclude_none=True),
                inputs.aliases if inputs else None,
            )
        except d.NoLogprobsError as e:
            pm.stats.record_logprobs(provider_id, ok=False)
            failure = failed("backend returned no logprobs", sample=False)
            if attempt == 0:
                await rt.rebuild()
                continue
            raise failure from e
        except d.DecisionError as e:
            raise failed(str(e), sample=False) from e
        except httpx.HTTPStatusError as e:
            raise failed(
                f"HTTP {e.response.status_code}: {e.response.text[:300]}"
            ) from e
        except httpx.HTTPError as e:
            raise failed(describe_httpx_error(e)) from e
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
        raise _FailedError(f"no probability for {qid!r}")
    return float(p)


def _level(answers: dict[str, Any]) -> str:
    """The most probable level of the tier question."""
    a = answers.get(TIER_ID)
    if not isinstance(a, dict):
        raise _FailedError("no answer for the tier question")
    probs = a.get("probabilities")
    if isinstance(probs, dict) and probs:
        return str(max(probs, key=lambda k: float(probs[k])))
    score = a.get("score")
    if isinstance(score, (int, float)):
        return str(round(score))
    raise _FailedError("no level for the tier question")


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
    403 when a gate rejects the request, 503 when gates could not decide
    (unless ``decision_policy_fail_open``) and 429 when the decision would
    pass the key's spend cap. Without a policy it returns *model* at once.
    """
    policy: DecisionPolicy | None = prefs.decision_policy if prefs else None
    if policy is None:
        return model
    questions = policy.questions_for(model)
    if not questions:
        return model
    state = request_text(body)
    if state is None:
        _note(request, "policy: skipped, no text")
        return model

    start = time.monotonic()
    req: d.DecisionRequest | None = None
    timeout = settings.decision_policy_timeout_seconds
    try:
        req = d.DecisionRequest(model=policy.model, state=state, questions=questions)
        answers, log = await asyncio.wait_for(
            _decide(rt, pm, db, prefs, request, req), timeout
        )
        gates = {name: _noul(answers, GATE_PREFIX + name) for name in policy.gates}
        level = _level(answers) if TIER_ID in questions else None
    except HTTPException:
        raise  # the key's spend cap
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
                request_body=req.model_dump(exclude_none=True) if req else None,
                duration_ms=ms,
                status="error",
                error_detail=f"decision policy: {reason}",
                sample=False,
            )
        elif isinstance(exc, _FailedError):
            reason = str(exc)
        else:
            logger.exception("Decision policy failed")
            reason = f"{type(exc).__name__}: {exc}"
        if policy.gates and not settings.decision_policy_fail_open:
            logger.warning(
                "Decision policy (key %s) failed after %.0f ms, refusing the "
                "request its gates could not check: %s",
                prefs.key_id if prefs else None,
                ms,
                reason,
            )
            _note(request, f"policy: failed in {ms:.0f}ms, refused")
            raise HTTPException(
                status_code=503,
                detail={
                    "error": "the API key's decision policy could not be "
                    "evaluated, so the request was refused",
                },
            ) from None
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
    if level is not None and policy.tier is not None:
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
