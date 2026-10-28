from __future__ import annotations

import logging
import time

import httpx
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse

from ..auth import routing_preferences_from_request
from ..httpx_errors import describe_httpx_error, forward_upstream_http_error
from ..request_logger import log_request_background
from . import deps

logger = logging.getLogger(__name__)
router = APIRouter()


async def _handle_embedding(
    request: Request, endpoint: str, method: str
) -> JSONResponse:
    body = await request.json()
    model = body.get("model")
    if not model:
        raise HTTPException(status_code=400, detail="model is required")

    rt = deps.get_router()
    pm = deps.get_pm()
    db = deps.get_db()
    prefs = await routing_preferences_from_request(db, request)

    result = await rt.route(model, request.url.path, prefs)
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
    start = time.monotonic()
    pm.acquire(provider_id)
    try:
        result = await getattr(client, method)(body)
        duration = (time.monotonic() - start) * 1000
        log_request_background(
            db,
            provider=provider,
            protocol="ollama",
            endpoint=endpoint,
            request=request,
            model=model,
            routed_model=routed,
            request_body=body,
            response_body=result,
            duration_ms=duration,
        )
        return JSONResponse(content=result)
    except httpx.HTTPStatusError as exc:
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
            protocol="ollama",
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
            protocol="ollama",
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
    finally:
        pm.release(provider_id)


@router.post("/api/embeddings")
async def embeddings(request: Request) -> JSONResponse:
    return await _handle_embedding(request, "/api/embeddings", "embeddings")


@router.post("/api/embed")
async def embed(request: Request) -> JSONResponse:
    return await _handle_embedding(request, "/api/embed", "embed")
