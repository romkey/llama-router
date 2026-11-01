"""Helpers for logging dispatched API requests to the database."""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from collections import deque
from collections.abc import AsyncGenerator, AsyncIterator, Callable, Iterable
from typing import Any

import httpx
from fastapi import Request
from fastapi.responses import Response, StreamingResponse

from .background import spawn
from .client_address import client_ip
from .config import settings
from .database import Database
from .endpoints import endpoint_class
from .httpx_errors import describe_httpx_error, forward_upstream_http_error
from .models import Provider, RequestLog
from .pricing import Price
from .routing_stats import ProviderStats
from .spend import tracker_for
from .timing import received_at

logger = logging.getLogger(__name__)

# Live request outcomes feed the routing table through these stats.
_stats: ProviderStats | None = None
# Prices requests by (provider id, model); the router's current price book.
_pricer: Callable[[int, str], Price | None] | None = None

# Upstream answers that say the provider, not the request, is the problem.
# 429 is left out: the provider is busy, not broken. Its client marks it busy
# for the Retry-After period instead (see ProviderManager._response_hook).
_PROVIDER_FAULT_4XX = frozenset({401, 403, 404, 408})
_STATUS_RE = re.compile(r"^(?:HTTP |HTTPStatusError )(\d{3})")
# Final-chunk fields that give generation speed.
_EVAL_COUNT_RE = re.compile(rb'"eval_count"\s*:\s*(\d+)')
_EVAL_DURATION_RE = re.compile(rb'"eval_duration"\s*:\s*(\d+)')
_PREDICTED_PER_SECOND_RE = re.compile(rb'"predicted_per_second"\s*:\s*([0-9.eE+-]+)')
_COMPLETION_TOKENS_RE = re.compile(rb'"completion_tokens"\s*:\s*(\d+)')
# Token counts in a stream's last chunks: OpenAI usage, Responses API usage,
# Ollama's final chunk.
_PROMPT_TOKEN_RES = tuple(
    re.compile(p)
    for p in (
        rb'"prompt_tokens"\s*:\s*(\d+)',
        rb'"input_tokens"\s*:\s*(\d+)',
        rb'"prompt_eval_count"\s*:\s*(\d+)',
    )
)
_COMPLETION_TOKEN_RES = tuple(
    re.compile(p)
    for p in (
        rb'"completion_tokens"\s*:\s*(\d+)',
        rb'"output_tokens"\s*:\s*(\d+)',
        rb'"eval_count"\s*:\s*(\d+)',
    )
)
# Chunks kept from the end of a stream to read those fields from.
_TAIL_CHUNKS = 3
# Request fields that cap the length of the answer.
_MAX_TOKEN_FIELDS = (
    "max_tokens",
    "max_completion_tokens",
    "max_output_tokens",
    "num_predict",
)
# Log status of a stream the client hung up on. Not an error, and not a
# sample for the routing stats either way.
STATUS_CANCELLED = "cancelled"


def attach_stats(stats: ProviderStats | None) -> None:
    """Send request outcomes to *stats* (the provider manager's)."""
    global _stats
    _stats = stats


def attach_pricer(pricer: Callable[[int, str], Price | None] | None) -> None:
    """Price logged requests with *pricer* (the router's ``price_for``)."""
    global _pricer
    _pricer = pricer


def usage_from_body(body: Any) -> tuple[int | None, int | None]:
    """(prompt, completion) tokens a non-streamed response reports."""
    if not isinstance(body, dict):
        return None, None
    usage = body.get("usage")
    if isinstance(usage, dict):
        prompt = usage.get("prompt_tokens", usage.get("input_tokens"))
        completion = usage.get("completion_tokens", usage.get("output_tokens"))
        if isinstance(prompt, int) or isinstance(completion, int):
            return (
                prompt if isinstance(prompt, int) else None,
                completion if isinstance(completion, int) else None,
            )
    prompt, completion = body.get("prompt_eval_count"), body.get("eval_count")
    if isinstance(prompt, int) or isinstance(completion, int):
        return (
            prompt if isinstance(prompt, int) else None,
            completion if isinstance(completion, int) else None,
        )
    return None, None


def usage_from_tail(tail: bytes) -> tuple[int | None, int | None]:
    """(prompt, completion) tokens from a stream's last chunks, if reported.

    OpenAI-style streams carry usage only when the client asked for it
    (``stream_options.include_usage``); Ollama's final chunk always does.
    """

    def last(patterns: Iterable[re.Pattern[bytes]]) -> int | None:
        for pattern in patterns:
            matches = pattern.findall(tail)
            if matches:
                return int(matches[-1])
        return None

    return last(_PROMPT_TOKEN_RES), last(_COMPLETION_TOKEN_RES)


def _requested_max_tokens(body: Any) -> int | None:
    """The output-token limit a request asked for; -1 for "no limit"."""
    if not isinstance(body, dict):
        return None
    options = body.get("options")
    values = [body.get(k) for k in _MAX_TOKEN_FIELDS]
    if isinstance(options, dict):
        values.append(options.get("num_predict"))
    limits = [v for v in values if isinstance(v, int) and not isinstance(v, bool)]
    if not limits:
        return None
    if any(v < 0 for v in limits):
        return -1  # Ollama's num_predict -1: generate until the model stops
    return max(limits)


def _timeout_from_request(error_detail: str | None, request_body: Any) -> bool:
    """A read timeout on a request that asked for a very long answer.

    The backend was busy generating what it was asked for; that says
    nothing about its health, and counting it would let any client trip a
    provider's circuit breaker on purpose.
    """
    if not (error_detail or "").startswith("ReadTimeout"):
        return False
    limit = _requested_max_tokens(request_body)
    if limit is None:
        return False
    return limit < 0 or limit > settings.circuit_timeout_max_tokens


def _is_provider_failure(error_detail: str | None) -> bool:
    m = _STATUS_RE.match(error_detail or "")
    if not m:
        return True  # transport error, timeout, or unknown
    code = int(m.group(1))
    return code >= 500 or code in _PROVIDER_FAULT_4XX


def _tps_from_body(body: Any, duration_ms: float) -> float | None:
    """Generation speed reported in a non-streamed response, if any."""
    if not isinstance(body, dict):
        return None
    count, dur = body.get("eval_count"), body.get("eval_duration")
    if isinstance(count, int) and isinstance(dur, int) and dur > 0:
        return count / (dur / 1e9)
    timings = body.get("timings")
    if isinstance(timings, dict):
        pps = timings.get("predicted_per_second")
        if isinstance(pps, (int, float)) and pps > 0:
            return float(pps)
    usage = body.get("usage")
    if isinstance(usage, dict):
        tokens = usage.get("completion_tokens")
        if isinstance(tokens, int) and tokens > 0 and duration_ms > 0:
            return tokens / (duration_ms / 1000)
    return None


def _tps_from_tail(tail: bytes, gen_ms: float) -> float | None:
    """Generation speed from the last chunks of a stream, if reported."""
    count = _EVAL_COUNT_RE.search(tail)
    dur = _EVAL_DURATION_RE.search(tail)
    if count and dur and int(dur.group(1)) > 0:
        return int(count.group(1)) / (int(dur.group(1)) / 1e9)
    pps = _PREDICTED_PER_SECOND_RE.search(tail)
    if pps:
        try:
            v = float(pps.group(1))
            if v > 0:
                return v
        except ValueError:
            pass
    tokens = _COMPLETION_TOKENS_RE.search(tail)
    if tokens and gen_ms > 0:
        return int(tokens.group(1)) / (gen_ms / 1000)
    return None


def _record_outcome(
    provider: Provider | None,
    model: str | None,
    *,
    status: str,
    error_detail: str | None,
    endpoint: str | None = None,
    ttft_ms: float | None = None,
    tokens_per_second: float | None = None,
    duration_ms: float | None = None,
    request_body: Any = None,
) -> None:
    stats = _stats
    if stats is None or provider is None or provider.id is None or not model:
        return
    if status == STATUS_CANCELLED:
        return  # the client went away; says nothing about the provider
    ok = status == "ok"
    if not ok and not _is_provider_failure(error_detail):
        return  # the client's request was bad; says nothing about the provider
    if not ok and _timeout_from_request(error_detail, request_body):
        return
    stats.record_request(
        provider.id,
        model,
        ok=ok,
        endpoint_class=endpoint_class(endpoint),
        ttft_ms=ttft_ms,
        tokens_per_second=tokens_per_second,
        duration_ms=duration_ms,
    )


def refusal_worth_logging(request: Request, status_code: int) -> bool:
    """Whether an API refusal goes to the request log: 400, 401 and 403
    always, 404 only with an API key, so scanners probing random paths
    cannot fill the log."""
    if status_code == 404:
        return getattr(request.state, "api_key_id", None) is not None
    return status_code in (400, 401, 403)


def _payload_size(body: Any) -> int:
    """Approximate size of the JSON request body in bytes."""
    try:
        return len(json.dumps(body).encode())
    except Exception:
        return 0


def _request_meta_summary(body: Any) -> str | None:
    if body is None:
        return None
    if not isinstance(body, (dict, list)):
        return None

    flags = {
        "images": False,
        "attachments": False,
        "audio": False,
        "tools": False,
    }
    counts = {"messages": 0, "items": 0, "questions": 0}
    stream = None

    if isinstance(body, dict):
        if isinstance(body.get("messages"), list):
            counts["messages"] = len(body["messages"])
        if isinstance(body.get("questions"), dict):
            counts["questions"] = len(body["questions"])
        if isinstance(body.get("input"), list):
            counts["items"] = len(body["input"])
        elif body.get("input") is not None:
            counts["items"] = 1
        if "stream" in body:
            stream = bool(body.get("stream"))

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            for k, v in node.items():
                lk = str(k).lower()
                if "image" in lk:
                    flags["images"] = True
                if lk in {"attachments", "files", "file"}:
                    flags["attachments"] = True
                if "audio" in lk or "voice" in lk:
                    flags["audio"] = True
                if lk in {"tools", "tool_calls"}:
                    flags["tools"] = True
                walk(v)
        elif isinstance(node, list):
            for item in node:
                walk(item)

    walk(body)
    parts: list[str] = []
    if counts["messages"]:
        parts.append(f"messages={counts['messages']}")
    if counts["questions"]:
        parts.append(f"questions={counts['questions']}")
    if counts["items"]:
        parts.append(f"input_items={counts['items']}")
    if stream is not None:
        parts.append(f"stream={str(stream).lower()}")
    for k in ("images", "attachments", "audio", "tools"):
        if flags[k]:
            parts.append(k)
    if isinstance(body, dict):
        if body.get("pull_api"):
            parts.append(f"pull_api={body.get('pull_api')}")
        if body.get("benchmark_api"):
            parts.append(f"benchmark_api={body.get('benchmark_api')}")

    if not parts:
        return None
    s = ", ".join(parts)
    return s[:240]


async def log_request(
    db: Database,
    *,
    provider: Provider | None,
    protocol: str,
    endpoint: str,
    request: Request | None = None,
    source_ip: str | None = None,
    model: str | None,
    request_body: Any = None,
    response_size: int = 0,
    duration_ms: float,
    streamed: bool = False,
    status: str = "ok",
    error_detail: str | None = None,
    response_body: Any = None,
    routed_model: str | None = None,
    sample: bool = False,
    ttfb_ms: float | None = None,
    router_ms: float | None = None,
    prompt_tokens: int | None = None,
    completion_tokens: int | None = None,
    api_key_id: int | None = None,
    request_meta: str | None = None,
) -> None:
    """Write one request to the log.

    *response_body* (non-streamed responses) is measured here instead of in
    the handler, so serialising it again never delays the response.
    *sample* also feeds the outcome to the routing stats under *routed_model*.
    *request_meta* replaces the summary worked out from *request_body*.
    """
    if response_body is not None and not response_size:
        response_size = _payload_size(response_body)
    if sample:
        _record_outcome(
            provider,
            routed_model or model,
            status=status,
            error_detail=error_detail,
            endpoint=endpoint,
            tokens_per_second=(
                _tps_from_body(response_body, duration_ms) if status == "ok" else None
            ),
            duration_ms=duration_ms,
            request_body=request_body,
        )
    ip = source_ip or (client_ip(request) if request else "internal")
    if api_key_id is None and request is not None:
        api_key_id = getattr(request.state, "api_key_id", None)
    if prompt_tokens is None and completion_tokens is None:
        prompt_tokens, completion_tokens = usage_from_body(response_body)
    cost = None
    if (
        _pricer is not None
        and provider is not None
        and provider.id is not None
        and (prompt_tokens is not None or completion_tokens is not None)
    ):
        price = _pricer(provider.id, routed_model or model or "")
        if price is not None:
            cost = price.cost(prompt_tokens, completion_tokens)
    if cost and api_key_id is not None:
        tracker_for(db).add(api_key_id, cost)
    entry = RequestLog(
        provider_id=provider.id if provider else None,
        provider_name=provider.name if provider else None,
        protocol=protocol,
        endpoint=endpoint,
        source_ip=ip,
        model=model,
        routed_model=routed_model if routed_model != model else None,
        request_size=_payload_size(request_body),
        response_size=response_size,
        request_meta=request_meta or _request_meta_summary(request_body),
        duration_ms=duration_ms,
        ttfb_ms=ttfb_ms,
        router_ms=router_ms,
        status=status,
        streamed=streamed,
        error_detail=error_detail,
        locality=provider.locality if provider else None,
        api_key_id=api_key_id,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        cost_usd=cost,
    )
    await db.save_request_log(entry)


def log_request_background(
    db: Database, *, dispatched_at: float | None = None, **kwargs: Any
) -> None:
    """Log after the response is sent; the handler does not wait for it.

    *dispatched_at* is when the request went to the backend. Handlers that
    log as soon as the backend answers can omit it: it is then worked out
    from *duration_ms*.
    """
    kwargs.setdefault("sample", True)
    request = kwargs.get("request")
    if request is not None:
        policy = getattr(request.state, "decision_policy", None)
        if policy and kwargs.get("request_meta") is None:
            # What the key's decision policy did, and how long it took.
            summary = _request_meta_summary(kwargs.get("request_body"))
            kwargs["request_meta"] = (
                f"{policy}; {summary}" if summary else str(policy)
            )[:240]
        # Resolve what is needed now; the request object is done after.
        kwargs["source_ip"] = kwargs.get("source_ip") or client_ip(request)
        if kwargs.get("api_key_id") is None:
            kwargs["api_key_id"] = getattr(request.state, "api_key_id", None)
        if kwargs.get("router_ms") is None:
            kwargs["router_ms"] = _router_ms(
                request, dispatched_at, kwargs.get("duration_ms", 0.0)
            )
        kwargs["request"] = None
    spawn(log_request(db, **kwargs), name="log-request")


def _router_ms(
    request: Request, dispatched_at: float | None, duration_ms: float
) -> float | None:
    """Time from arrival at the router to dispatch, or None if unstamped."""
    arrived = received_at(request.scope)
    if arrived is None:
        return None
    if dispatched_at is None:
        dispatched_at = time.monotonic() - duration_ms / 1000
    return max(0.0, (dispatched_at - arrived) * 1000)


class StreamLogger:
    """Wraps an async byte iterator to measure total response size and duration."""

    def __init__(
        self,
        inner: AsyncIterator[bytes],
        *,
        db: Database,
        provider: Provider,
        protocol: str,
        endpoint: str,
        request: Request,
        model: str | None,
        request_body: Any,
        start_time: float,
        routed_model: str | None = None,
    ):
        self._inner = inner
        self._db = db
        self._provider = provider
        self._protocol = protocol
        self._endpoint = endpoint
        self._request = request
        self._model = model
        self._request_body = request_body
        self._start = start_time
        self._routed_model = routed_model
        self._total_bytes = 0
        self._first_chunk_at: float | None = None
        self._tail: deque[bytes] = deque(maxlen=_TAIL_CHUNKS)

    async def __aiter__(self) -> AsyncGenerator[bytes]:
        status = "ok"
        error_detail = None
        try:
            async for chunk in self._inner:
                if self._first_chunk_at is None:
                    self._first_chunk_at = time.monotonic()
                self._total_bytes += len(chunk)
                self._tail.append(chunk)
                yield chunk
        except asyncio.CancelledError, GeneratorExit:
            # The client disconnected before the stream finished.
            status = STATUS_CANCELLED
            error_detail = "client disconnected before the response finished"
            raise
        except Exception as exc:
            status = "error"
            if isinstance(exc, httpx.HTTPError):
                error_detail = describe_httpx_error(exc)[:500]
                logger.error(
                    "Upstream streaming failure on %s (model=%r, provider=%s): %s",
                    self._endpoint,
                    self._model,
                    self._provider.name,
                    describe_httpx_error(exc),
                )
            else:
                error_detail = str(exc)[:500]
            raise
        finally:
            ended = time.monotonic()
            duration = (ended - self._start) * 1000
            ttft_ms = None
            tps = None
            prompt_tokens, completion_tokens = usage_from_tail(b"".join(self._tail))
            if self._first_chunk_at is not None:
                ttft_ms = (self._first_chunk_at - self._start) * 1000
                if status == "ok":
                    tps = _tps_from_tail(
                        b"".join(self._tail), (ended - self._first_chunk_at) * 1000
                    )
            _record_outcome(
                self._provider,
                self._routed_model or self._model,
                status=status,
                error_detail=error_detail,
                endpoint=self._endpoint,
                ttft_ms=ttft_ms,
                tokens_per_second=tps,
                duration_ms=duration,
                request_body=self._request_body,
            )
            try:
                log_request_background(
                    self._db,
                    provider=self._provider,
                    protocol=self._protocol,
                    endpoint=self._endpoint,
                    request=self._request,
                    model=self._model,
                    request_body=self._request_body,
                    response_size=self._total_bytes,
                    duration_ms=duration,
                    streamed=True,
                    status=status,
                    error_detail=error_detail,
                    sample=False,
                    ttfb_ms=ttft_ms,
                    dispatched_at=self._start,
                    routed_model=self._routed_model,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                )
            except Exception:
                logger.exception("Could not schedule request log write")


async def stream_response(logged: StreamLogger, media_type: str) -> Response:
    """Start *logged* and answer with it as a stream.

    The first chunk is awaited before the response begins. If the upstream
    refuses the request, the client gets that error as a normal HTTP reply
    instead of a 200 whose body stops short (which clients report as a
    transfer-length error).
    """
    stream = logged.__aiter__()
    try:
        first = await stream.__anext__()
    except StopAsyncIteration:
        return StreamingResponse(iter(()), media_type=media_type)
    except httpx.HTTPStatusError as exc:
        return forward_upstream_http_error(exc)
    except BaseException:
        await stream.aclose()
        raise

    async def chunks() -> AsyncIterator[bytes]:
        try:
            yield first
            async for chunk in stream:
                yield chunk
        finally:
            await stream.aclose()

    return StreamingResponse(chunks(), media_type=media_type)
