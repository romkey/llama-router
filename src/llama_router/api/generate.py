from __future__ import annotations

import logging
import time
from collections.abc import AsyncIterator

import httpx
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

from ..auth import routing_preferences_from_request
from ..decision_policy import apply_decision_policy
from ..httpx_errors import describe_httpx_error, forward_upstream_http_error
from ..request_logger import StreamLogger, log_request_background
from ..translate import TranslationError
from . import deps

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/api/generate")
async def generate(request: Request) -> Response:
    body = await request.json()
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
    session = rt.session_for(request.headers, body, prefs)

    route_result = await rt.route(model, request.url.path, prefs, session=session)
    if not route_result:
        raise HTTPException(
            status_code=404, detail=f"No available provider for model '{model}'"
        )

    provider = route_result.provider
    routed = route_result.model_key
    if route_result.resolved_model != model:
        body["model"] = route_result.resolved_model

    provider_id = provider.require_id()
    body["model"] = route_result.backend_model
    client = pm.ollama_api_client(provider_id)
    if hasattr(client, "validate"):
        try:
            client.validate("generate", body)
        except TranslationError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
    stream = body.get("stream", True)
    start = time.monotonic()

    if stream:

        async def generate_chunks() -> AsyncIterator[bytes]:
            try:
                async with pm.acquire_provider(provider_id):
                    async for chunk in client.generate_stream(body):
                        yield chunk
            except Exception:
                rt.forget_session(session)
                raise

        logged = StreamLogger(
            generate_chunks(),
            db=db,
            provider=provider,
            protocol="ollama",
            endpoint="/api/generate",
            request=request,
            model=model,
            routed_model=routed,
            request_body=body,
            start_time=start,
        )
        return StreamingResponse(logged, media_type="application/x-ndjson")

    try:
        async with pm.acquire_provider(provider_id):
            payload = await client.generate(body)
            duration = (time.monotonic() - start) * 1000
            log_request_background(
                db,
                provider=provider,
                protocol="ollama",
                endpoint="/api/generate",
                request=request,
                model=model,
                routed_model=routed,
                request_body=body,
                response_body=payload,
                duration_ms=duration,
            )
            return JSONResponse(content=payload)
    except httpx.HTTPStatusError as exc:
        rt.forget_session(session)
        duration = (time.monotonic() - start) * 1000
        logger.warning(
            "Backend %s returned HTTP %d for /api/generate %s",
            provider.name,
            exc.response.status_code,
            model,
        )
        log_request_background(
            db,
            provider=provider,
            protocol="ollama",
            endpoint="/api/generate",
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
                "Upstream failure on /api/generate (model=%r, provider=%s): %s",
                model,
                provider.name,
                describe_httpx_error(exc),
            )
        log_request_background(
            db,
            provider=provider,
            protocol="ollama",
            endpoint="/api/generate",
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
