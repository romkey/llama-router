from __future__ import annotations

import logging
import time

import httpx
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse, Response

from ..auth import routing_preferences_from_request
from ..httpx_errors import describe_httpx_error, forward_upstream_http_error
from ..request_logger import log_request_background
from ..v1_client import get_v1_client
from . import deps

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/v1/images/generations")
async def images_generations(request: Request) -> JSONResponse:
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
    client = get_v1_client(pm, provider_id)
    start = time.monotonic()
    try:
        async with pm.acquire_provider(provider_id):
            resp = await client.images_generations(body)
            duration = (time.monotonic() - start) * 1000
            log_request_background(
                db,
                provider=provider,
                protocol="v1",
                endpoint="/v1/images/generations",
                request=request,
                model=model,
                routed_model=routed,
                request_body=body,
                response_body=resp,
                duration_ms=duration,
            )
            return JSONResponse(content=resp)
    except httpx.HTTPStatusError as exc:
        duration = (time.monotonic() - start) * 1000
        logger.warning(
            "Backend %s returned HTTP %d for /v1/images/generations %s",
            provider.name,
            exc.response.status_code,
            model,
        )
        log_request_background(
            db,
            provider=provider,
            protocol="v1",
            endpoint="/v1/images/generations",
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
                "Upstream failure on /v1/images/generations "
                "(model=%r, provider=%s): %s",
                model,
                provider.name,
                describe_httpx_error(exc),
            )
        log_request_background(
            db,
            provider=provider,
            protocol="v1",
            endpoint="/v1/images/generations",
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


@router.post("/v1/images/edits")
async def images_edits(request: Request) -> Response:
    """Proxy multipart image edit requests."""
    content_type = request.headers.get("content-type", "")
    raw_body = await request.body()
    log_body = {"attachments": True, "content_type": content_type}

    form = await request.form()
    model = form.get("model")
    if not model or not isinstance(model, str):
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
    provider_id = provider.require_id()
    client = get_v1_client(pm, provider_id)
    start = time.monotonic()
    try:
        async with pm.acquire_provider(provider_id):
            resp = await client.images_edits(raw_body, content_type)
            duration = (time.monotonic() - start) * 1000
            log_request_background(
                db,
                provider=provider,
                protocol="v1",
                endpoint="/v1/images/edits",
                request=request,
                model=model,
                routed_model=routed,
                request_body=log_body,
                response_size=len(resp.content),
                duration_ms=duration,
            )
            return Response(
                content=resp.content,
                status_code=resp.status_code,
                media_type=resp.headers.get("content-type", "application/json"),
            )
    except httpx.HTTPStatusError as exc:
        duration = (time.monotonic() - start) * 1000
        logger.warning(
            "Backend %s returned HTTP %d for /v1/images/edits %s",
            provider.name,
            exc.response.status_code,
            model,
        )
        log_request_background(
            db,
            provider=provider,
            protocol="v1",
            endpoint="/v1/images/edits",
            request=request,
            model=model,
            routed_model=routed,
            request_body=log_body,
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
                "Upstream failure on /v1/images/edits (model=%r, provider=%s): %s",
                model,
                provider.name,
                describe_httpx_error(exc),
            )
        log_request_background(
            db,
            provider=provider,
            protocol="v1",
            endpoint="/v1/images/edits",
            request=request,
            model=model,
            routed_model=routed,
            request_body=log_body,
            response_size=0,
            duration_ms=duration,
            status="error",
            error_detail=err_detail,
        )
        raise
