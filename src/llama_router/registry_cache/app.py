"""Minimal OCI Distribution registry that acts as a pull-through cache.

Ollama pulls models by fetching manifests and blobs from an OCI-like registry.
This app proxies those requests to registry.ollama.ai and caches the responses
on disk so subsequent pulls are served at LAN speed.

IMPORTANT: Ollama's blob download code makes the initial GET request without
following redirects, then unconditionally reads the Location header to get the
real download URL. This means blob GET endpoints MUST return a 307 redirect —
returning 200 with data directly will fail with "no Location header".

For uncached blobs the redirect points to our own /cache/blobs/ endpoint which
does a single-pass stream-through: data flows from the upstream CDN through our
server to Ollama, and is simultaneously saved to disk. This avoids downloading
the blob twice (once for Ollama, once for the cache).
"""

from __future__ import annotations

import logging
import os
import time
from collections.abc import AsyncIterator
from urllib.parse import quote

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from starlette.responses import FileResponse
from starlette.types import ASGIApp, Receive, Scope, Send

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


class _RequestLogMiddleware:
    """Pure ASGI middleware — safe for StreamingResponse (no body buffering)."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            return await self.app(scope, receive, send)

        start = time.monotonic()
        path = scope.get("path", "?")
        method = scope.get("method", "?")
        client_host = "-"
        if scope.get("client"):
            client_host = scope["client"][0]

        logger.info("Cache %s %s from %s", method, path, client_host)

        status_code = 0

        async def _send_wrapper(message: dict) -> None:
            nonlocal status_code
            if message["type"] == "http.response.start":
                status_code = message.get("status", 0)
            await send(message)

        try:
            await self.app(scope, receive, _send_wrapper)
        except Exception:
            elapsed = (time.monotonic() - start) * 1000
            logger.error("Cache %s %s → exception after %.0fms", method, path, elapsed)
            raise

        elapsed = (time.monotonic() - start) * 1000
        logger.info("Cache %s %s → %d (%.0fms)", method, path, status_code, elapsed)


app.add_middleware(_RequestLogMiddleware)


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
            "Manifest upstream HTTP %d for %s:%s", resp.status_code, name, reference
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
async def get_blob(name: str, digest: str, request: Request):
    """Return a 307 redirect for blob downloads.

    Ollama's downloader expects a redirect — it calls resp.Location()
    unconditionally on the initial response.

    Cached blobs redirect to /cache/blobs/{digest} (FileResponse).
    Uncached blobs also redirect there with a query param so the serve
    endpoint can stream-through from upstream in a single pass (download
    once, cache and serve simultaneously).
    """
    cache = _get_cache()
    headers = {"Docker-Content-Digest": digest}
    base = str(request.base_url).rstrip("/")

    if cache.has_blob(digest):
        cache.blob_hits += 1
        size = cache.blob_size(digest)
        logger.info(
            "Blob cache HIT: %s (%s) — redirecting to local serve",
            digest[:24],
            _human_bytes(size),
        )
        headers["Content-Length"] = str(size)
        headers["Location"] = f"{base}/cache/blobs/{digest}"
        return Response(status_code=307, headers=headers)

    cache.blob_misses += 1

    # Try HEAD to upstream for Content-Length (progress display only).
    # This must NEVER prevent the 307 redirect from being returned —
    # Ollama unconditionally calls resp.Location() and any non-redirect
    # response causes "http: no Location header in response".
    upstream_url = f"{UPSTREAM}/v2/{name}/blobs/{digest}"
    try:
        async with httpx.AsyncClient(follow_redirects=True) as client:
            head = await client.head(upstream_url, timeout=10.0)
            if head.status_code == 200:
                cl = head.headers.get("content-length")
                if cl:
                    headers["Content-Length"] = cl
                    logger.info(
                        "Blob cache MISS: %s (%s) — redirecting to stream-through",
                        digest[:24],
                        _human_bytes(int(cl)),
                    )
                else:
                    logger.info(
                        "Blob cache MISS: %s — redirecting to stream-through",
                        digest[:24],
                    )
            else:
                logger.info(
                    "Blob cache MISS: %s (HEAD upstream %d) — redirecting to stream-through",
                    digest[:24],
                    head.status_code,
                )
    except Exception as exc:
        logger.info(
            "Blob cache MISS: %s (HEAD failed: %s) — redirecting to stream-through",
            digest[:24],
            exc,
        )

    encoded_name = quote(name, safe="")
    headers["Location"] = f"{base}/cache/blobs/{digest}?upstream_name={encoded_name}"
    return Response(status_code=307, headers=headers)


@app.get("/cache/blobs/{digest}")
async def serve_blob(digest: str, upstream_name: str = ""):
    """Serve a blob — from cache or via single-pass stream-through.

    Cached blobs are served with FileResponse (supports Range requests).
    Uncached blobs are streamed from upstream while simultaneously being
    written to disk, so a single download populates the cache and serves
    the client at the same time.
    """
    cache = _get_cache()

    if cache.has_blob(digest):
        path = cache.blob_path(digest)
        size = cache.blob_size(digest)
        logger.info("Serving cached blob: %s (%s)", digest[:24], _human_bytes(size))
        return FileResponse(
            path=str(path),
            media_type="application/octet-stream",
            headers={"Docker-Content-Digest": digest},
        )

    if not upstream_name:
        logger.warning("Serve blob miss, no upstream_name: %s", digest[:24])
        raise HTTPException(status_code=404, detail="Blob not in cache")

    upstream_url = f"{UPSTREAM}/v2/{upstream_name}/blobs/{digest}"
    logger.info("Stream-through starting: %s from %s", digest[:24], upstream_name)

    async def _stream_and_cache() -> AsyncIterator[bytes]:
        # Use pid in temp filename to avoid collisions from concurrent requests
        tmp = cache.blob_path(digest).with_suffix(f".{os.getpid()}.tmp")
        total = 0
        try:
            async with httpx.AsyncClient(follow_redirects=True) as client:
                async with client.stream(
                    "GET",
                    upstream_url,
                    timeout=httpx.Timeout(30.0, read=1800.0),
                ) as resp:
                    resp.raise_for_status()
                    with open(tmp, "wb") as f:
                        async for chunk in resp.aiter_bytes(CHUNK_SIZE):
                            f.write(chunk)
                            total += len(chunk)
                            yield chunk
            final = cache.blob_path(digest)
            tmp.rename(final)
            logger.info(
                "Stream-through complete, cached: %s (%s)",
                digest[:24],
                _human_bytes(total),
            )
        except GeneratorExit:
            tmp.unlink(missing_ok=True)
            logger.warning(
                "Stream-through client disconnected: %s after %s",
                digest[:24],
                _human_bytes(total),
            )
        except Exception as exc:
            tmp.unlink(missing_ok=True)
            logger.error(
                "Stream-through failed: %s after %s — %s",
                digest[:24],
                _human_bytes(total),
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
