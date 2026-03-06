"""Minimal OCI Distribution registry that acts as a pull-through cache.

Ollama pulls models by fetching manifests and blobs from an OCI-like registry.
This app proxies those requests to registry.ollama.ai and caches the responses
on disk so subsequent pulls are served at LAN speed.
"""

from __future__ import annotations

import logging
import time
from typing import AsyncIterator

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from starlette.middleware.base import BaseHTTPMiddleware

from .cache import BlobCache

logger = logging.getLogger(__name__)

UPSTREAM = "https://registry.ollama.ai"
CHUNK_SIZE = 256 * 1024  # 256 KB

app = FastAPI(title="llama-router Registry Cache", redirect_slashes=False)

_cache: BlobCache | None = None


def init_cache(cache: BlobCache) -> None:
    global _cache
    _cache = cache


def _get_cache() -> BlobCache:
    assert _cache is not None, "Registry cache not initialized"
    return _cache


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        start = time.monotonic()
        logger.info(
            "Cache %s %s from %s",
            request.method,
            request.url.path,
            request.client.host if request.client else "?",
        )
        try:
            response = await call_next(request)
        except Exception as exc:
            elapsed = (time.monotonic() - start) * 1000
            logger.error(
                "Cache %s %s → exception after %.0fms: %s",
                request.method,
                request.url.path,
                elapsed,
                exc,
            )
            raise
        elapsed = (time.monotonic() - start) * 1000
        logger.info(
            "Cache %s %s → %d (%.0fms)",
            request.method,
            request.url.path,
            response.status_code,
            elapsed,
        )
        return response


app.add_middleware(RequestLoggingMiddleware)


# Accept both /v2 and /v2/ since redirect_slashes is disabled
@app.get("/v2")
@app.get("/v2/")
async def v2_check():
    """OCI version check — Ollama calls this to verify the registry is reachable."""
    return JSONResponse({})


@app.get("/v2/{name:path}/manifests/{reference}")
async def get_manifest(name: str, reference: str):
    cache = _get_cache()

    cached = cache.get_manifest(name, reference)
    if cached is not None:
        cache.manifest_hits += 1
        logger.info(
            "Manifest cache HIT: %s:%s (%d bytes)", name, reference, len(cached)
        )
        return Response(
            content=cached,
            media_type="application/vnd.docker.distribution.manifest.v2+json",
        )

    cache.manifest_misses += 1
    url = f"{UPSTREAM}/v2/{name}/manifests/{reference}"
    logger.info("Manifest cache MISS: %s:%s — fetching from upstream", name, reference)
    try:
        async with httpx.AsyncClient(follow_redirects=True) as client:
            resp = await client.get(url, timeout=30.0)
    except Exception as exc:
        logger.error("Manifest upstream fetch failed: %s:%s — %s", name, reference, exc)
        raise HTTPException(status_code=502, detail=f"Upstream fetch error: {exc}")

    if resp.status_code != 200:
        logger.error(
            "Manifest upstream HTTP %d for %s:%s",
            resp.status_code,
            name,
            reference,
        )
        status = 404 if resp.status_code == 404 else 502
        raise HTTPException(status_code=status, detail="Upstream error")

    data = resp.content
    cache.save_manifest(name, reference, data)
    logger.info(
        "Manifest fetched and cached: %s:%s (%d bytes)", name, reference, len(data)
    )
    return Response(
        content=data,
        media_type=resp.headers.get(
            "content-type", "application/vnd.docker.distribution.manifest.v2+json"
        ),
    )


@app.head("/v2/{name:path}/blobs/{digest}")
async def head_blob(name: str, digest: str):
    cache = _get_cache()

    if cache.has_blob(digest):
        size = cache.blob_size(digest)
        logger.info("HEAD blob HIT: %s (%s)", digest[:24], _human_bytes(size))
        return Response(
            status_code=200,
            headers={
                "Content-Length": str(size),
                "Docker-Content-Digest": digest,
            },
        )

    url = f"{UPSTREAM}/v2/{name}/blobs/{digest}"
    logger.info("HEAD blob MISS (local), checking upstream: %s", digest[:24])
    try:
        async with httpx.AsyncClient(follow_redirects=True) as client:
            resp = await client.head(url, timeout=30.0)
    except Exception as exc:
        logger.error("HEAD blob upstream failed: %s — %s", digest[:24], exc)
        raise HTTPException(status_code=502, detail=f"Upstream error: {exc}")

    if resp.status_code != 200:
        logger.warning(
            "HEAD blob upstream HTTP %d for %s", resp.status_code, digest[:24]
        )
        status = 404 if resp.status_code == 404 else 502
        raise HTTPException(status_code=status, detail="Upstream error")

    content_length = resp.headers.get("content-length", "0")
    logger.info("HEAD blob upstream OK: %s (%s bytes)", digest[:24], content_length)
    return Response(
        status_code=200,
        headers={
            "Content-Length": content_length,
            "Docker-Content-Digest": digest,
        },
    )


@app.get("/v2/{name:path}/blobs/{digest}")
async def get_blob(name: str, digest: str):
    cache = _get_cache()

    if cache.has_blob(digest):
        cache.blob_hits += 1
        path = cache.blob_path(digest)
        size = cache.blob_size(digest)
        logger.info("Blob cache HIT: %s (%s)", digest[:24], _human_bytes(size))

        async def _stream_file() -> AsyncIterator[bytes]:
            with open(path, "rb") as f:
                while True:
                    chunk = f.read(CHUNK_SIZE)
                    if not chunk:
                        break
                    yield chunk

        return StreamingResponse(
            _stream_file(),
            media_type="application/octet-stream",
            headers={
                "Content-Length": str(size),
                "Docker-Content-Digest": digest,
            },
        )

    cache.blob_misses += 1
    url = f"{UPSTREAM}/v2/{name}/blobs/{digest}"
    logger.info("Blob cache MISS: %s — streaming from upstream", digest[:24])

    async def _stream_and_cache() -> AsyncIterator[bytes]:
        """Stream the blob to the client while saving it to disk."""
        tmp = cache.temp_blob_path(digest)
        total_bytes = 0
        try:
            async with httpx.AsyncClient(follow_redirects=True) as client:
                async with client.stream(
                    "GET", url, timeout=httpx.Timeout(30.0, read=1800.0)
                ) as resp:
                    resp.raise_for_status()
                    with open(tmp, "wb") as f:
                        async for chunk in resp.aiter_bytes(CHUNK_SIZE):
                            f.write(chunk)
                            total_bytes += len(chunk)
                            yield chunk
            cache.commit_blob(digest)
            logger.info("Blob cached: %s (%s)", digest[:24], _human_bytes(total_bytes))
        except Exception as exc:
            cache.remove_temp_blob(digest)
            logger.error(
                "Blob stream-through FAILED: %s after %s — %s",
                digest[:24],
                _human_bytes(total_bytes),
                exc,
            )
            raise

    return StreamingResponse(
        _stream_and_cache(),
        media_type="application/octet-stream",
        headers={"Docker-Content-Digest": digest},
    )


def _human_bytes(n: int) -> str:
    if n >= 1 << 30:
        return f"{n / (1 << 30):.1f} GB"
    if n >= 1 << 20:
        return f"{n / (1 << 20):.1f} MB"
    return f"{n / (1 << 10):.0f} KB"
