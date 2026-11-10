"""The shared path of the JSON-body proxy endpoints.

Every Ollama (``/api/...``) and OpenAI-style (``/v1/...``) endpoint that takes
a JSON body does the same thing: read the body, route the model, take an
upstream slot, call the provider (streamed or not), log the request and turn
an upstream failure into the right reply. :func:`proxy_json` is that path; a
handler names its endpoint and the client method to call.
"""

from __future__ import annotations

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
from .client_address import client_ip
from .database import Database
from .decision_policy import apply_decision_policy
from .httpx_errors import describe_httpx_error, forward_upstream_http_error
from .limits import clamp_generation_params, upstream_slot
from .llamacpp_api import deps as v1_deps
from .models import Provider
from .provider_manager import ProviderManager
from .request_body import json_object
from .request_logger import (
    StreamLogger,
    ask_stream_usage,
    log_request_background,
    stream_response,
)
from .router import Router
from .v1_client import get_v1_client

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

    provider = result.provider
    routed = result.model_key

    provider_id = provider.require_id()
    body["model"] = result.backend_model
    if check is not None:
        check(provider, body)

    def client() -> Any:
        if protocol == "ollama":
            return pm.ollama_api_client(provider_id)
        return get_v1_client(pm, provider_id)

    start = time.monotonic()

    if stream_call is not None and (
        call is None or body.get("stream", _STREAM_DEFAULT[protocol])
    ):
        slot = await upstream_slot(request, prefs, stream=True)
        # Usage in the stream's last event lets the request be costed.
        hide_usage = ask_stream_usage(body, provider) if stream_usage else False

        async def chunks() -> AsyncIterator[bytes]:
            try:
                async with slot, pm.acquire_provider(provider_id):
                    async for chunk in slot.stream(stream_call(client(), body)):
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
            routed_model=routed,
            request_body=body,
            start_time=start,
            hide_usage=hide_usage,
        )
        content_type = media_type(body) if media_type else _STREAM_MEDIA_TYPE[protocol]
        return await stream_response(logged, content_type)

    if call is None:
        raise TypeError("proxy_json needs call or stream_call")
    slot = await upstream_slot(request, prefs)
    try:
        async with slot, pm.acquire_provider(provider_id):
            reply = await call(client(), body)
            duration = (time.monotonic() - start) * 1000
            log_request_background(
                db,
                provider=provider,
                protocol=protocol,
                endpoint=endpoint,
                request=request,
                model=model,
                routed_model=routed,
                request_body=body,
                response_body=reply,
                duration_ms=duration,
            )
            return JSONResponse(content=reply)
    except httpx.HTTPStatusError as exc:
        rt.forget_session(session)
        duration = (time.monotonic() - start) * 1000
        logger.warning(
            "Backend %s returned HTTP %d for %s %s",
            provider.name,
            exc.response.status_code,
            endpoint,
            model,
        )
        log_request_background(
            db,
            provider=provider,
            protocol=protocol,
            endpoint=endpoint,
            request=request,
            model=model,
            routed_model=routed,
            request_body=body,
            response_size=0,
            duration_ms=duration,
            status="error",
            error_detail=f"HTTP {exc.response.status_code}: {exc.response.text[:400]}",
        )
        return forward_upstream_http_error(exc)
    except Exception as exc:
        rt.forget_session(session)
        duration = (time.monotonic() - start) * 1000
        err_detail = str(exc)[:500]
        if isinstance(exc, httpx.HTTPError):
            err_detail = describe_httpx_error(exc)[:500]
            logger.error(
                "Upstream failure on %s (model=%r, provider=%s): %s",
                endpoint,
                model,
                provider.name,
                describe_httpx_error(exc),
            )
        log_request_background(
            db,
            provider=provider,
            protocol=protocol,
            endpoint=endpoint,
            request=request,
            model=model,
            routed_model=routed,
            request_body=body,
            response_size=0,
            duration_ms=duration,
            status="error",
            error_detail=err_detail,
        )
        raise
