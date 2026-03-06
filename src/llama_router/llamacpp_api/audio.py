from __future__ import annotations

import logging
import time

import httpx
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

from ..request_logger import StreamLogger, log_request
from ..v1_client import get_v1_client
from . import deps

logger = logging.getLogger(__name__)
router = APIRouter()


def _forward_backend_error(exc: httpx.HTTPStatusError) -> JSONResponse:
    try:
        body = exc.response.json()
    except Exception:
        body = {"error": exc.response.text or str(exc)}
    return JSONResponse(content=body, status_code=exc.response.status_code)


@router.post("/v1/audio/speech")
async def audio_speech(request: Request):
    """Text-to-speech: JSON body in, streaming audio bytes out."""
    body = await request.json()
    model = body.get("model")
    if not model:
        raise HTTPException(status_code=400, detail="model is required")

    rt = deps.get_router()
    pm = deps.get_pm()
    db = deps.get_db()

    result = await rt.route(model)
    if not result:
        raise HTTPException(
            status_code=404, detail=f"No available provider for model '{model}'"
        )

    provider = result.provider
    if result.resolved_model != model:
        body["model"] = result.resolved_model

    assert provider.id is not None
    body["model"] = pm.resolve_backend_model_name(provider.id, body["model"])
    client = get_v1_client(pm, provider.id)
    start = time.monotonic()

    pm.acquire(provider.id)
    try:

        async def generate():
            try:
                async for chunk in client.audio_speech(body):
                    yield chunk
            finally:
                pm.release(provider.id)

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
            request_body=body,
            start_time=start,
        )
        return StreamingResponse(logged, media_type=media_type)
    except httpx.HTTPStatusError as exc:
        pm.release(provider.id)
        duration = (time.monotonic() - start) * 1000
        logger.warning(
            "Backend %s returned HTTP %d for /v1/audio/speech %s",
            provider.name,
            exc.response.status_code,
            model,
        )
        await log_request(
            db,
            provider=provider,
            protocol="v1",
            endpoint="/v1/audio/speech",
            request=request,
            model=model,
            request_body=body,
            response_size=0,
            duration_ms=duration,
            status="error",
            error_detail=f"HTTP {exc.response.status_code}: {exc.response.text[:400]}",
        )
        return _forward_backend_error(exc)
    except Exception as exc:
        pm.release(provider.id)
        duration = (time.monotonic() - start) * 1000
        await log_request(
            db,
            provider=provider,
            protocol="v1",
            endpoint="/v1/audio/speech",
            request=request,
            model=model,
            request_body=body,
            response_size=0,
            duration_ms=duration,
            status="error",
            error_detail=str(exc)[:500],
        )
        raise


@router.post("/v1/audio/transcriptions")
async def audio_transcriptions(request: Request):
    """Speech-to-text: multipart audio file in, JSON transcript out."""
    content_type = request.headers.get("content-type", "")
    raw_body = await request.body()

    form = await request.form()
    model = form.get("model")
    if not model or not isinstance(model, str):
        raise HTTPException(status_code=400, detail="model is required")

    rt = deps.get_router()
    pm = deps.get_pm()
    db = deps.get_db()

    result = await rt.route(model)
    if not result:
        raise HTTPException(
            status_code=404, detail=f"No available provider for model '{model}'"
        )

    provider = result.provider
    assert provider.id is not None
    client = get_v1_client(pm, provider.id)
    start = time.monotonic()
    pm.acquire(provider.id)
    try:
        resp = await client.audio_transcriptions(raw_body, content_type)
        duration = (time.monotonic() - start) * 1000
        await log_request(
            db,
            provider=provider,
            protocol="v1",
            endpoint="/v1/audio/transcriptions",
            request=request,
            model=model,
            request_body={},
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
        await log_request(
            db,
            provider=provider,
            protocol="v1",
            endpoint="/v1/audio/transcriptions",
            request=request,
            model=model,
            request_body={},
            response_size=0,
            duration_ms=duration,
            status="error",
            error_detail=f"HTTP {exc.response.status_code}: {exc.response.text[:400]}",
        )
        return _forward_backend_error(exc)
    except Exception as exc:
        duration = (time.monotonic() - start) * 1000
        await log_request(
            db,
            provider=provider,
            protocol="v1",
            endpoint="/v1/audio/transcriptions",
            request=request,
            model=model,
            request_body={},
            response_size=0,
            duration_ms=duration,
            status="error",
            error_detail=str(exc)[:500],
        )
        raise
    finally:
        pm.release(provider.id)


@router.get("/v1/audio/voices")
async def audio_voices():
    """List available voices — aggregated from all online providers."""
    pm = deps.get_pm()
    infos = await pm.list_provider_infos()

    all_voices: list[dict] = []
    seen: set[str] = set()

    for info in infos:
        if info.provider.id is None:
            continue
        try:
            client = get_v1_client(pm, info.provider.id)
        except Exception:
            continue
        try:
            data = await client.audio_voices()
            for v in data.get("voices", data.get("data", [])):
                vid = v.get("voice_id") or v.get("id") or v.get("name", "")
                if vid and vid not in seen:
                    seen.add(vid)
                    all_voices.append(v)
        except Exception:
            continue

    return JSONResponse(content={"voices": all_voices})
