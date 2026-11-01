from __future__ import annotations

import logging
import time
from collections.abc import AsyncIterator

import httpx
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse, Response

from ..auth import route_within_cap, routing_preferences_from_request
from ..client_address import client_ip
from ..decision_policy import apply_decision_policy
from ..httpx_errors import describe_httpx_error, forward_upstream_http_error
from ..request_body import json_object
from ..request_logger import (
    StreamLogger,
    ask_stream_usage,
    log_request_background,
    stream_response,
)
from ..v1_client import get_v1_client
from . import deps

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/v1/chat/completions")
@router.post("/chat/completions")
async def chat_completions(request: Request) -> Response:
    endpoint_path = request.url.path
    body = await json_object(request)
    model = body.get("model")
    if not model:
        raise HTTPException(status_code=400, detail="model is required")

    rt = deps.get_router()
    pm = deps.get_pm()
    db = deps.get_db()
    prefs = await routing_preferences_from_request(db, request)
    model = await apply_decision_policy(
        request, body, model, prefs, rt=rt, pm=pm, db=db
    )
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
    if result.resolved_model != model:
        body["model"] = result.resolved_model

    provider_id = provider.require_id()
    body["model"] = result.backend_model
    client = get_v1_client(pm, provider_id)
    stream = body.get("stream", False)
    start = time.monotonic()

    if stream:
        # Usage in the stream's last event lets the request be costed.
        hide_usage = ask_stream_usage(body, provider)

        async def generate() -> AsyncIterator[bytes]:
            try:
                async with pm.acquire_provider(provider_id):
                    async for chunk in client.chat_completions_stream(body):
                        yield chunk
            except Exception:
                rt.forget_session(session)
                raise

        logged = StreamLogger(
            generate(),
            db=db,
            provider=provider,
            protocol="v1",
            endpoint=endpoint_path,
            request=request,
            model=model,
            routed_model=routed,
            request_body=body,
            start_time=start,
            hide_usage=hide_usage,
        )
        return await stream_response(logged, "text/event-stream")

    try:
        async with pm.acquire_provider(provider_id):
            resp = await client.chat_completions(body)
            duration = (time.monotonic() - start) * 1000
            log_request_background(
                db,
                provider=provider,
                protocol="v1",
                endpoint=endpoint_path,
                request=request,
                model=model,
                routed_model=routed,
                request_body=body,
                response_body=resp,
                duration_ms=duration,
            )
            return JSONResponse(content=resp)
    except httpx.HTTPStatusError as exc:
        rt.forget_session(session)
        duration = (time.monotonic() - start) * 1000
        logger.warning(
            "Backend %s returned HTTP %d for %s %s",
            provider.name,
            exc.response.status_code,
            endpoint_path,
            model,
        )
        log_request_background(
            db,
            provider=provider,
            protocol="v1",
            endpoint=endpoint_path,
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
                endpoint_path,
                model,
                provider.name,
                describe_httpx_error(exc),
            )
        log_request_background(
            db,
            provider=provider,
            protocol="v1",
            endpoint=endpoint_path,
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
