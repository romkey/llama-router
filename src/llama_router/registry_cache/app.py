"""Minimal OCI Distribution registry that acts as a pull-through cache.

Ollama pulls models by fetching manifests and blobs from an OCI-like registry.
This app proxies those requests to registry.ollama.ai and caches the responses
on disk so subsequent pulls are served at LAN speed.

IMPORTANT — Ollama's blob download flow (see download.go):
  1. HEAD /v2/{name}/blobs/{digest}  → get Content-Length for partitioning
  2. GET  /v2/{name}/blobs/{digest}  → get the "direct URL" for downloading

For step 2, Ollama uses a custom redirect handler that FOLLOWS same-hostname
redirects but STOPS at cross-hostname redirects (returning http.ErrUseLastResponse).
After the request completes, Ollama unconditionally calls resp.Location().

This means:
  - A 307 to our OWN host is silently followed → Ollama gets the final 200 →
    Location() fails → "http: no Location header in response"
  - A 200 with a Location header works — Go's resp.Location() reads the header
    regardless of status code

Strategy:
  - Cached blobs: return 200 with Location pointing to /cache/blobs/{digest}
  - Uncached blobs: stream bytes from upstream (follow_redirects=True) to Ollama
    while writing to disk (tee), then commit; concurrent GETs for the same
    digest wait on a per-digest asyncio.Event and then serve from cache.

Performance:
  - One process-wide httpx.AsyncClient (lifespan) pools connections to upstream/CDN.
  - Blob sizes from manifests (pre-warmed from disk at startup) avoid upstream
    HEAD when sizes are known.
  - Model pre-caching uses a bounded semaphore for parallel blob downloads.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from starlette.responses import FileResponse
from starlette.types import ASGIApp, Receive, Scope, Send

from ..config import settings
from .cache import BlobCache

logger = logging.getLogger(__name__)

UPSTREAM = "https://registry.ollama.ai"
CHUNK_SIZE = 256 * 1024  # 256 KB

_cache: BlobCache | None = None
_upstream_client: httpx.AsyncClient | None = None
_blob_semaphore: asyncio.Semaphore | None = None
_blob_sizes: dict[str, int] = {}
_active_downloads: dict[str, asyncio.Event] = {}


@asynccontextmanager
async def _lifespan(app: FastAPI):
    global _upstream_client, _blob_semaphore
    _upstream_client = httpx.AsyncClient(
        follow_redirects=True,
        timeout=httpx.Timeout(30.0, read=1800.0),
    )
    _blob_semaphore = asyncio.Semaphore(settings.cache_max_concurrent_blobs)
    yield
    await _upstream_client.aclose()
    _upstream_client = None
    _blob_semaphore = None


app = FastAPI(
    title="llama-router Registry Cache",
    redirect_slashes=False,
    lifespan=_lifespan,
)


def init_cache(cache: BlobCache) -> None:
    global _cache
    _cache = cache
    for manifest_bytes in cache.all_cached_manifests():
        _extract_blob_sizes(manifest_bytes)
    logger.info("Pre-warmed _blob_sizes with %d entries", len(_blob_sizes))


def _get_cache() -> BlobCache:
    assert _cache is not None, "Registry cache not initialized"
    return _cache


def _upstream() -> httpx.AsyncClient:
    assert _upstream_client is not None, "upstream client not initialized"
    return _upstream_client


def _extract_blob_sizes(manifest_bytes: bytes) -> None:
    """Parse an OCI manifest and cache blob digest→size mappings."""
    try:
        manifest = json.loads(manifest_bytes)
        for layer in manifest.get("layers", []):
            digest = layer.get("digest")
            size = layer.get("size")
            if digest and size:
                _blob_sizes[digest] = size
        config = manifest.get("config", {})
        if config.get("digest") and config.get("size"):
            _blob_sizes[config["digest"]] = config["size"]
    except Exception:
        pass


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
        _extract_blob_sizes(cached)
        return Response(
            content=cached,
            media_type="application/vnd.docker.distribution.manifest.v2+json",
        )

    cache.manifest_misses += 1
    url = f"{UPSTREAM}/v2/{name}/manifests/{reference}"
    logger.info("Manifest cache MISS: %s:%s — fetching from upstream", name, reference)
    try:
        client = _upstream()
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
    _extract_blob_sizes(data)
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
    """Return Content-Length for a blob.  Ollama calls this in Prepare()."""
    cache = _get_cache()
    # 1) Local blob  2) Size from manifest (no upstream)  3) Upstream HEAD
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

    if digest in _blob_sizes:
        size = _blob_sizes[digest]
        logger.info(
            "HEAD blob size from manifest: %s (%s)", digest[:24], _human_bytes(size)
        )
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
        client = _upstream()
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
    try:
        _blob_sizes[digest] = int(content_length)
    except ValueError:
        pass
    return Response(
        status_code=200,
        headers={
            "Content-Length": content_length,
            "Docker-Content-Digest": digest,
        },
    )


@app.get("/v2/{name:path}/blobs/{digest}")
async def get_blob(name: str, digest: str, request: Request):
    """Serve blob downloads: cache hit → Location to local file; miss → stream-through."""
    cache = _get_cache()
    base = str(request.base_url).rstrip("/")
    hit_headers = {"Docker-Content-Digest": digest}

    if cache.has_blob(digest):
        cache.blob_hits += 1
        size = cache.blob_size(digest)
        logger.info("Blob cache HIT: %s (%s)", digest[:24], _human_bytes(size))
        hit_headers["Location"] = f"{base}/cache/blobs/{digest}"
        return Response(status_code=200, content=b"", headers=hit_headers)

    if digest in _active_downloads:
        event = _active_downloads[digest]
        await event.wait()
        if cache.has_blob(digest):
            cache.blob_hits += 1
            size = cache.blob_size(digest)
            logger.info(
                "Blob served from cache after wait: %s (%s)",
                digest[:24],
                _human_bytes(size),
            )
            hit_headers["Location"] = f"{base}/cache/blobs/{digest}"
            return Response(status_code=200, content=b"", headers=hit_headers)
        logger.error("Blob wait finished but cache still missing: %s", digest[:24])
        raise HTTPException(status_code=502, detail="Upstream blob download failed")

    cache.blob_misses += 1
    upstream_url = f"{UPSTREAM}/v2/{name}/blobs/{digest}"
    tmp = cache.temp_blob_path(digest)
    event = asyncio.Event()
    _active_downloads[digest] = event
    logger.info("Blob cache MISS: %s — stream-through from upstream", digest[:24])

    async def stream_and_cache() -> AsyncIterator[bytes]:
        try:
            client = _upstream()
            async with client.stream(
                "GET",
                upstream_url,
                timeout=httpx.Timeout(30.0, read=1800.0),
            ) as resp:
                resp.raise_for_status()
                with open(tmp, "wb") as f:
                    async for chunk in resp.aiter_bytes(CHUNK_SIZE):
                        f.write(chunk)
                        yield chunk
            cache.commit_blob(digest)
            logger.info("Streamed and cached blob: %s", digest[:24])
        except Exception as exc:
            cache.remove_temp_blob(digest)
            logger.error("Stream-and-cache failed for %s: %s", digest[:24], exc)
            raise
        finally:
            event.set()
            _active_downloads.pop(digest, None)

    return StreamingResponse(
        stream_and_cache(),
        media_type="application/octet-stream",
        headers={
            "Docker-Content-Digest": digest,
            "Location": f"{base}/cache/blobs/{digest}",
        },
    )


@app.get("/cache/blobs/{digest}")
async def serve_cached_blob(digest: str):
    """Serve a cached blob.  Supports Range requests via FileResponse."""
    cache = _get_cache()
    if not cache.has_blob(digest):
        logger.warning("Serve blob miss (not yet cached): %s", digest[:24])
        raise HTTPException(status_code=404, detail="Blob not in cache")

    path = cache.blob_path(digest)
    size = cache.blob_size(digest)
    logger.info("Serving cached blob: %s (%s)", digest[:24], _human_bytes(size))
    return FileResponse(
        path=str(path),
        media_type="application/octet-stream",
        headers={"Docker-Content-Digest": digest},
    )


async def precache_model(
    cache: BlobCache,
    model: str,
    progress_callback: Any | None = None,
) -> None:
    """Download a model's manifest and all blobs into the cache.

    *model* is in Ollama format, e.g. ``llama3.2:latest`` or ``llama3.2``.
    """
    if ":" in model:
        name_part, tag = model.rsplit(":", 1)
    else:
        name_part, tag = model, "latest"

    if "/" not in name_part:
        oci_name = f"library/{name_part}"
    else:
        oci_name = name_part

    if progress_callback:
        progress_callback(f"fetching manifest for {model}")

    manifest_url = f"{UPSTREAM}/v2/{oci_name}/manifests/{tag}"
    client = _upstream()
    resp = await client.get(manifest_url, timeout=30.0)
    if resp.status_code != 200:
        raise RuntimeError(f"Manifest fetch failed: HTTP {resp.status_code}")

    manifest_data = resp.content
    cache.save_manifest(oci_name, tag, manifest_data)
    _extract_blob_sizes(manifest_data)

    manifest = json.loads(manifest_data)
    layers = manifest.get("layers", [])
    config = manifest.get("config")
    if config and config.get("digest"):
        layers = [config] + layers

    total = len(layers)
    sem = _blob_semaphore
    assert sem is not None, "blob semaphore not initialized"

    async def _fetch_one_blob(i: int, layer: dict) -> None:
        digest = layer.get("digest", "")
        size = layer.get("size", 0)
        if not digest:
            return
        if cache.has_blob(digest):
            if progress_callback:
                progress_callback(
                    f"blob {i}/{total} already cached ({_human_bytes(size)})"
                )
            return
        async with sem:
            if cache.has_blob(digest):
                if progress_callback:
                    progress_callback(
                        f"blob {i}/{total} already cached ({_human_bytes(size)})"
                    )
                return
            if progress_callback:
                progress_callback(
                    f"downloading blob {i}/{total} ({_human_bytes(size)})"
                )
            blob_url = f"{UPSTREAM}/v2/{oci_name}/blobs/{digest}"
            tmp = cache.temp_blob_path(digest)
            total_bytes = 0
            try:
                async with client.stream(
                    "GET",
                    blob_url,
                    timeout=httpx.Timeout(30.0, read=3600.0),
                ) as blob_resp:
                    blob_resp.raise_for_status()
                    with open(tmp, "wb") as f:
                        async for chunk in blob_resp.aiter_bytes(CHUNK_SIZE):
                            f.write(chunk)
                            total_bytes += len(chunk)
                            if progress_callback and size > 0:
                                pct = int(total_bytes * 100 / size)
                                progress_callback(
                                    f"downloading blob {i}/{total} {pct}% "
                                    f"({_human_bytes(total_bytes)}/{_human_bytes(size)})"
                                )
                cache.commit_blob(digest)
                if not cache.has_blob(digest):
                    raise RuntimeError(
                        "Blob commit failed — file not found after rename"
                    )
                logger.info(
                    "Precache blob complete: %s (%s)",
                    digest[:24],
                    _human_bytes(total_bytes),
                )
            except Exception as exc:
                cache.remove_temp_blob(digest)
                raise RuntimeError(
                    f"Blob download failed ({digest[:24]}): {exc}"
                ) from exc

    await asyncio.gather(
        *[_fetch_one_blob(i, layer) for i, layer in enumerate(layers, 1)]
    )
    cached = sum(
        1 for layer in layers if layer.get("digest") and cache.has_blob(layer["digest"])
    )

    if progress_callback:
        progress_callback(f"cached {model} ({total} blobs)")

    logger.info("Pre-cached model %s: %d/%d blobs present", model, cached, total)


def _human_bytes(n: int) -> str:
    if n >= 1 << 30:
        return f"{n / (1 << 30):.1f} GB"
    if n >= 1 << 20:
        return f"{n / (1 << 20):.1f} MB"
    return f"{n / (1 << 10):.0f} KB"
