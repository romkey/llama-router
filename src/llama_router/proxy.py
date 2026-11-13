"""The shared path of the JSON-body proxy endpoints.

Every Ollama (``/api/...``) and OpenAI-style (``/v1/...``) endpoint that takes
a JSON body does the same thing: read the body, route the model, take an
upstream slot, call the provider (streamed or not), log the request and turn
an upstream failure into the right reply. :func:`proxy_json` is that path; a
handler names its endpoint and the client method to call.
"""

from __future__ import annotations

import copy
import logging
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from types import ModuleType
from typing import Any, Literal

import httpx
from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse, Response

from .api import deps as ollama_deps
from .auth import (
    model_allowed,
    require_model_access,
    route_within_cap,
    routing_preferences_from_request,
)
from .capacity import ProviderAtCapacity, SlotGate
from .client_address import client_ip
from .config import settings
from .database import Database
from .decision_policy import apply_decision_policy
from .httpx_errors import describe_httpx_error, forward_upstream_http_error
from .limits import (
    DeadlineExceeded,
    clamp_generation_params,
    request_deadline,
    upstream_slot,
)
from .llamacpp_api import deps as v1_deps
from .models import Provider
from .provider_manager import ProviderManager
from .request_body import json_object
from .request_logger import (
    StreamLogger,
    ask_stream_usage,
    log_request_background,
    prime_stream,
    stream_body,
)
from .router import PinUnavailableError, Router, RouteResult, RoutingPreferences
from .spend import RESERVATION_STATE, Reservation
from .v1_client import NoReachableAddressError, get_v1_client

logger = logging.getLogger(__name__)

Protocol = Literal["ollama", "v1"]
Body = dict[str, Any]
# The client is the provider's Ollama API client (``protocol="ollama"``) or
# its V1Client (``protocol="v1"``).
Call = Callable[[Any, Body], Awaitable[Any]]
StreamCall = Callable[[Any, Body], AsyncIterator[bytes]]

_DEPS: dict[Protocol, ModuleType] = {"ollama": ollama_deps, "v1": v1_deps}
_STREAM_MEDIA_TYPE: dict[Protocol, str] = {
    "ollama": "application/x-ndjson",
    "v1": "text/event-stream",
}
# Ollama streams unless told not to; the OpenAI API does not.
_STREAM_DEFAULT: dict[Protocol, bool] = {"ollama": True, "v1": False}


async def proxy_json(
    request: Request,
    *,
    protocol: Protocol,
    endpoint: str,
    call: Call | None,
    stream_call: StreamCall | None,
    generation: bool = True,
    stream_usage: bool = False,
    check: Callable[[Provider, Body], None] | None = None,
    media_type: Callable[[Body], str] | None = None,
) -> Response:
    """Proxy one JSON-body request to the provider its model routes to.

    *call* answers a non-streamed request with a JSON object; *stream_call*
    yields a streamed one. With no *call*, every request is streamed; with no
    *stream_call*, none is.

    *generation* requests have their sampling parameters clamped, go through
    the key's decision policy and keep session affinity (forgotten when the
    upstream fails). *stream_usage* asks a cloud provider to end the stream
    with its usage. *check* may refuse the routed request before it is sent.
    *media_type* picks a streamed reply's content type from the body.
    """
    deps = _DEPS[protocol]
    db: Database = deps.get_db()
    prefs = await routing_preferences_from_request(db, request)
    body = await json_object(request)
    if generation:
        clamp_generation_params(body)
    model = body.get("model")
    if not model:
        raise HTTPException(status_code=400, detail="model is required")

    rt: Router = deps.get_router()
    pm: ProviderManager = deps.get_pm()
    await require_model_access(rt, prefs, model)
    session: str | None = None
    if generation:
        chosen = await apply_decision_policy(
            request, body, model, prefs, rt=rt, pm=pm, db=db
        )
        if chosen != model and not await model_allowed(rt, prefs, chosen):
            # The policy picked a model the key may not use: keep the one
            # the client asked for, as for any choice that can't be served.
            logger.warning(
                "Decision policy chose %s, which API key %s may not use; "
                "routing %s instead",
                chosen,
                prefs.key_id if prefs else None,
                model,
            )
            chosen = model
        model = chosen
        session = rt.session_for(request.headers, body, prefs, client_ip(request))

    result = await route_within_cap(
        rt, db, request, model, prefs, body, session=session
    )
    if not result:
        raise HTTPException(
            status_code=404, detail=f"No available provider for model '{model}'"
        )
    failover = _Failover(rt, db, request, model, prefs, body, session)

    def client(provider_id: int) -> Any:
        if protocol == "ollama":
            return pm.ollama_api_client(provider_id)
        return get_v1_client(pm, provider_id)

    def prepare(chosen: RouteResult) -> Provider:
        body["model"] = chosen.backend_model
        if check is not None:
            check(chosen.provider, body)
        return chosen.provider

    if stream_call is not None and (
        call is None or body.get("stream", _STREAM_DEFAULT[protocol])
    ):
        slot = await upstream_slot(request, prefs, stream=True)
        content_type = media_type(body) if media_type else _STREAM_MEDIA_TYPE[protocol]
        # As the client sent it: asking a cloud provider for usage changes it,
        # and the next attempt may go elsewhere.
        sent_options = "stream_options" in body
        client_options = copy.deepcopy(body.get("stream_options"))
        try:
            while True:
                if sent_options:
                    body["stream_options"] = copy.deepcopy(client_options)
                else:
                    body.pop("stream_options", None)
                provider = prepare(result)
                provider_id = provider.require_id()
                # Usage in the stream's last event lets the request be costed.
                hide_usage = ask_stream_usage(body, provider) if stream_usage else False
                held = failover.reservation()
                # Waits in line when the provider is at its max_concurrent
                # limit; the stream holds the slot until it ends.
                gate = await pm.take_slot(provider_id, slot.deadline)

                async def chunks(
                    pid: int = provider_id, gate: SlotGate | None = gate
                ) -> AsyncIterator[bytes]:
                    try:
                        async with pm.acquire_provider(pid, slot=gate, slot_taken=True):
                            async for chunk in slot.stream(
                                stream_call(client(pid), body)
                            ):
                                yield chunk
                    except Exception:
                        rt.forget_session(session)
                        raise

                logged = StreamLogger(
                    chunks(),
                    db=db,
                    provider=provider,
                    protocol=protocol,
                    endpoint=endpoint,
                    request=request,
                    model=model,
                    routed_model=result.model_key,
                    request_body=body,
                    start_time=time.monotonic(),
                    hide_usage=hide_usage,
                )
                try:
                    first, stream = await prime_stream(logged)
                except Exception as exc:
                    # Nothing has reached the client yet: another provider
                    # may still serve it. The failed attempt is logged.
                    nxt = await failover.next(exc, result, held)
                    if nxt is not None:
                        result = nxt
                        continue
                    slot.release()
                    if isinstance(exc, httpx.HTTPStatusError):
                        return forward_upstream_http_error(exc)
                    raise
                return stream_body(first, stream, content_type, on_close=slot.release)
        except BaseException:
            slot.release()
            raise

    if call is None:
        raise TypeError("proxy_json needs call or stream_call")
    slot = await upstream_slot(request, prefs)
    provider = result.provider
    start = time.monotonic()
    try:
        # One slot, and one request deadline, across every attempt.
        async with slot:
            while True:
                provider = prepare(result)
                provider_id = provider.require_id()
                held = failover.reservation()
                start = time.monotonic()
                try:
                    async with pm.acquire_provider(provider_id, deadline=slot.deadline):
                        reply = await call(client(provider_id), body)
                except ProviderAtCapacity:
                    # Not the provider's failure: it was never sent the request.
                    raise
                except Exception as exc:
                    rt.forget_session(session)
                    _log_failure(
                        db, protocol, endpoint, request, model, result, body, exc, start
                    )
                    nxt = await failover.next(exc, result, held)
                    if nxt is not None:
                        result = nxt
                        continue
                    if isinstance(exc, httpx.HTTPStatusError):
                        return forward_upstream_http_error(exc)
                    raise
                log_request_background(
                    db,
                    provider=provider,
                    protocol=protocol,
                    endpoint=endpoint,
                    request=request,
                    model=model,
                    routed_model=result.model_key,
                    request_body=body,
                    response_body=reply,
                    duration_ms=(time.monotonic() - start) * 1000,
                )
                return JSONResponse(content=reply)
    except DeadlineExceeded as exc:
        # The deadline cancelled the call in flight: log it as that call's
        # failure, as any other.
        rt.forget_session(session)
        _log_failure(db, protocol, endpoint, request, model, result, body, exc, start)
        raise


def _log_failure(
    db: Database,
    protocol: Protocol,
    endpoint: str,
    request: Request,
    model: str,
    result: RouteResult,
    body: Body,
    exc: Exception,
    start: float,
) -> None:
    """Log one failed upstream call against the provider that failed it, so
    its stats see the failure."""
    provider = result.provider
    duration = (time.monotonic() - start) * 1000
    if isinstance(exc, httpx.HTTPStatusError):
        logger.warning(
            "Backend %s returned HTTP %d for %s %s",
            provider.name,
            exc.response.status_code,
            endpoint,
            model,
        )
        detail = f"HTTP {exc.response.status_code}: {exc.response.text[:400]}"
    elif isinstance(exc, httpx.HTTPError):
        detail = describe_httpx_error(exc)[:500]
        logger.error(
            "Upstream failure on %s (model=%r, provider=%s): %s",
            endpoint,
            model,
            provider.name,
            describe_httpx_error(exc),
        )
    else:
        detail = str(exc)[:500]
    log_request_background(
        db,
        provider=provider,
        protocol=protocol,
        endpoint=endpoint,
        request=request,
        model=model,
        routed_model=result.model_key,
        request_body=body,
        response_size=0,
        duration_ms=duration,
        status="error",
        error_detail=detail,
    )


# Upstream answers worth trying elsewhere: the provider is full or its
# gateway failed. A 500, 4xx or the upstream refusing our credentials is
# not: another provider would most likely answer the same, or the client
# needs to see it.
_FAILOVER_STATUSES = frozenset({429, 502, 503})


def may_fail_over(exc: BaseException) -> bool:
    """*exc* happened before the provider took the request, or the provider
    turned it away as busy: another provider can try. Read timeouts are not
    retried, since the provider may still be working on it."""
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code in _FAILOVER_STATUSES
    return isinstance(
        exc,
        (
            httpx.ConnectError,
            httpx.ConnectTimeout,
            httpx.PoolTimeout,
            NoReachableAddressError,
        ),
    )


class _Failover:
    """Routes a request again after its provider failed it, within
    ``LLAMA_ROUTER_UPSTREAM_FAILOVER_ATTEMPTS`` and the request deadline.

    Every attempt goes back through :func:`route_within_cap`, so model
    access, locality, spend caps and hub scope are checked again; providers
    that already failed are excluded. The previous attempt's spend
    reservation is released first, so a capped key holds one estimate at a
    time.
    """

    def __init__(
        self,
        rt: Router,
        db: Database,
        request: Request,
        model: str,
        prefs: RoutingPreferences | None,
        body: Body,
        session: str | None,
    ) -> None:
        self._rt = rt
        self._db = db
        self._request = request
        self._model = model
        self._prefs = prefs
        self._body = body
        self._session = session
        self._tried: set[int] = set()
        self._deadline = request_deadline(request)

    def reservation(self) -> Reservation | None:
        """The spend reservation the current attempt's routing took."""
        held = getattr(self._request.state, RESERVATION_STATE, None)
        return held if isinstance(held, Reservation) else None

    def _allowed(self, result: RouteResult) -> bool:
        prefs = self._prefs
        if prefs is None:
            return True
        if prefs.hub_provider is not None:
            return False  # the hub asked for that one provider
        # A strict pin is never served by another provider.
        return not (
            prefs.strict_pins and result.resolved_model in prefs.pinned_providers
        )

    async def next(
        self, exc: BaseException, failed: RouteResult, held: Reservation | None
    ) -> RouteResult | None:
        """Where to try next after *failed* raised *exc*, or None to give up
        and answer with *exc*."""
        failed_id = failed.provider.require_id()
        self._tried.add(failed_id)
        if held is not None:
            held.release()
        if (
            len(self._tried) > settings.upstream_failover_attempts
            or not may_fail_over(exc)
            or not self._allowed(failed)
            or (self._deadline is not None and time.monotonic() >= self._deadline)
        ):
            return None
        try:
            nxt = await route_within_cap(
                self._rt,
                self._db,
                self._request,
                self._model,
                self._prefs,
                self._body,
                session=self._session,
                exclude=frozenset(self._tried),
            )
        except HTTPException, PinUnavailableError:
            # Nowhere else it may go (a cap, a strict pin): the client gets
            # the provider's own error.
            return None
        if nxt is not None:
            logger.warning(
                "Failing over %s from provider %s to %s after: %s",
                self._model,
                failed.provider.name,
                nxt.provider.name,
                type(exc).__name__,
            )
        return nxt
