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
from ..limits import clamp_generation_params, upstream_slot
from ..request_body import json_object
from ..request_logger import StreamLogger, log_request_background, stream_response
from . import deps

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/api/chat")
async def chat(request: Request) -> Response:
    db = deps.get_db()
    prefs = await routing_preferences_from_request(db, request)
    body = await json_object(request)
    clamp_generation_params(body)
    model = body.get("model")
    if not model:
        raise HTTPException(status_code=400, detail="model is required")

    rt = deps.get_router()
    pm = deps.get_pm()
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
    client = pm.ollama_api_client(provider_id)
    stream = body.get("stream", True)
    start = time.monotonic()

    if stream:
        slot = await upstream_slot(request, prefs, stream=True)

        async def generate() -> AsyncIterator[bytes]:
            try:
                async with slot, pm.acquire_provider(provider_id):
                    async for chunk in slot.stream(client.chat_stream(body)):
                        yield chunk
            except Exception:
                rt.forget_session(session)
                raise

        logged = StreamLogger(
            generate(),
            db=db,
            provider=provider,
            protocol="ollama",
            endpoint="/api/chat",
            request=request,
            model=model,
            routed_model=routed,
            request_body=body,
            start_time=start,
        )
        return await stream_response(logged, "application/x-ndjson")

    slot = await upstream_slot(request, prefs)
    try:
        async with slot, pm.acquire_provider(provider_id):
            reply = await client.chat(body)
            duration = (time.monotonic() - start) * 1000
            log_request_background(
                db,
                provider=provider,
                protocol="ollama",
                endpoint="/api/chat",
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
            "Backend %s returned HTTP %d for /api/chat %s",
            provider.name,
            exc.response.status_code,
            model,
        )
        log_request_background(
            db,
            provider=provider,
            protocol="ollama",
            endpoint="/api/chat",
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
                "Upstream failure on /api/chat (model=%r, provider=%s): %s",
                model,
                provider.name,
                describe_httpx_error(exc),
            )
        log_request_background(
            db,
            provider=provider,
            protocol="ollama",
            endpoint="/api/chat",
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
