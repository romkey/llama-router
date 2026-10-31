from __future__ import annotations

import logging
import time
from collections.abc import AsyncIterator
from typing import Any

import httpx
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse, Response

from ..auth import from_hub, hub_provider, routing_preferences_from_request
from ..httpx_errors import describe_httpx_error, forward_upstream_http_error
from ..request_logger import StreamLogger, log_request_background, stream_response
from ..v1_client import get_v1_client
from . import deps

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/v1/audio/speech")
async def audio_speech(request: Request) -> Response:
    """Text-to-speech: JSON body in, streaming audio bytes out."""
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

    async def generate() -> AsyncIterator[bytes]:
        async with pm.acquire_provider(provider_id):
            async for chunk in client.audio_speech(body):
                yield chunk

    response_format = body.get("response_format", "mp3")
    media_types = {
        "mp3": "audio/mpeg",
        "opus": "audio/opus",
        "aac": "audio/aac",
        "flac": "audio/flac",
        "wav": "audio/wav",
        "pcm": "audio/pcm",
    }
    media_type = media_types.get(response_format, "audio/mpeg")

    logged = StreamLogger(
        generate(),
        db=db,
        provider=provider,
        protocol="v1",
        endpoint="/v1/audio/speech",
        request=request,
        model=model,
        routed_model=routed,
        request_body=body,
        start_time=start,
    )
    return await stream_response(logged, media_type)


@router.post("/v1/audio/transcriptions")
async def audio_transcriptions(request: Request) -> Response:
    """Speech-to-text: multipart audio file in, JSON transcript out."""
    content_type = request.headers.get("content-type", "")
    raw_body = await request.body()
    log_body = {"audio": True, "attachments": True, "content_type": content_type}

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
            resp = await client.audio_transcriptions(raw_body, content_type)
            duration = (time.monotonic() - start) * 1000
            log_request_background(
                db,
                provider=provider,
                protocol="v1",
                endpoint="/v1/audio/transcriptions",
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
            "Backend %s returned HTTP %d for /v1/audio/transcriptions %s",
            provider.name,
            exc.response.status_code,
            model,
        )
        log_request_background(
            db,
            provider=provider,
            protocol="v1",
            endpoint="/v1/audio/transcriptions",
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
                "Upstream failure on /v1/audio/transcriptions "
                "(model=%r, provider=%s): %s",
                model,
                provider.name,
                describe_httpx_error(exc),
            )
        log_request_background(
            db,
            provider=provider,
            protocol="v1",
            endpoint="/v1/audio/transcriptions",
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


@router.get("/v1/audio/voices")
async def audio_voices(request: Request) -> JSONResponse:
    """List available voices — aggregated from all online providers (a hub
    sees only the providers shared with it)."""
    pm = deps.get_pm()
    infos = await pm.list_provider_infos()
    hub = from_hub(request)
    only = hub_provider(request)

    all_voices: list[dict[str, Any]] = []
    seen: set[str] = set()

    for info in infos:
        if info.provider.id is None or (hub and not pm.shares_with_hub(info.provider)):
            continue
        if only is not None and info.provider.id != only:
            continue
        try:
            client = get_v1_client(pm, info.provider.id)
        except Exception:
            logger.debug(
                "no v1 client for provider %s", info.provider.id, exc_info=True
            )
            continue
        try:
            data = await client.audio_voices()
            for v in data.get("voices", data.get("data", [])):
                vid = v.get("voice_id") or v.get("id") or v.get("name", "")
                if vid and vid not in seen:
                    seen.add(vid)
                    all_voices.append(v)
        except Exception:
            logger.debug(
                "audio_voices failed for provider %s", info.provider.id, exc_info=True
            )
            continue

    return JSONResponse(content={"voices": all_voices})
