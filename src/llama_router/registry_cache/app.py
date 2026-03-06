"""Minimal OCI Distribution registry that acts as a pull-through cache.

Ollama pulls models by fetching manifests and blobs from an OCI-like registry.
This app proxies those requests to registry.ollama.ai and caches the responses
on disk so subsequent pulls are served at LAN speed.
"""

from __future__ import annotations

import logging
from typing import AsyncIterator

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

from .cache import BlobCache

logger = logging.getLogger(__name__)

UPSTREAM = "https://registry.ollama.ai"
CHUNK_SIZE = 256 * 1024  # 256 KB

app = FastAPI(title="llama-router Registry Cache")

_cache: BlobCache | None = None


def init_cache(cache: BlobCache) -> None:
    global _cache
    _cache = cache


def _get_cache() -> BlobCache:
    assert _cache is not None, "Registry cache not initialized"
    return _cache


@app.get("/v2/")
async def v2_check():
    """OCI version check — Ollama calls this to verify the registry is reachable."""
    return JSONResponse({})


@app.get("/v2/{name:path}/manifests/{reference}")
async def get_manifest(name: str, reference: str, request: Request):
    cache = _get_cache()

    cached = cache.get_manifest(name, reference)
    if cached is not None:
        cache.manifest_hits += 1
        logger.debug("Manifest cache hit: %s:%s", name, reference)
        return Response(
            content=cached,
            media_type="application/vnd.docker.distribution.manifest.v2+json",
        )

    cache.manifest_misses += 1
    logger.info("Manifest cache miss: %s:%s — fetching from upstream", name, reference)
    url = f"{UPSTREAM}/v2/{name}/manifests/{reference}"
    async with httpx.AsyncClient(follow_redirects=True) as client:
        resp = await client.get(url, timeout=30.0)

    if resp.status_code != 200:
        raise HTTPException(status_code=resp.status_code, detail="Upstream error")

    data = resp.content
    cache.save_manifest(name, reference, data)
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
        return Response(
            status_code=200,
            headers={
                "Content-Length": str(size),
                "Docker-Content-Digest": digest,
            },
        )

    url = f"{UPSTREAM}/v2/{name}/blobs/{digest}"
    async with httpx.AsyncClient(follow_redirects=True) as client:
        resp = await client.head(url, timeout=30.0)

    if resp.status_code != 200:
        raise HTTPException(status_code=resp.status_code, detail="Upstream error")

    return Response(
        status_code=200,
        headers={
            "Content-Length": resp.headers.get("content-length", "0"),
            "Docker-Content-Digest": digest,
        },
    )


@app.get("/v2/{name:path}/blobs/{digest}")
async def get_blob(name: str, digest: str):
    cache = _get_cache()

    if cache.has_blob(digest):
        cache.blob_hits += 1
        logger.debug("Blob cache hit: %s", digest[:24])
        path = cache.blob_path(digest)
        size = cache.blob_size(digest)

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
    logger.info("Blob cache miss: %s — streaming from upstream", digest[:24])
    url = f"{UPSTREAM}/v2/{name}/blobs/{digest}"

    async def _stream_and_cache() -> AsyncIterator[bytes]:
        """Stream the blob to the client while saving it to disk."""
        tmp = cache.temp_blob_path(digest)
        try:
            async with httpx.AsyncClient(follow_redirects=True) as client:
                async with client.stream(
                    "GET", url, timeout=httpx.Timeout(30.0, read=600.0)
                ) as resp:
                    resp.raise_for_status()
                    with open(tmp, "wb") as f:
                        async for chunk in resp.aiter_bytes(CHUNK_SIZE):
                            f.write(chunk)
                            yield chunk
            cache.commit_blob(digest)
            logger.info("Blob cached: %s", digest[:24])
        except Exception:
            cache.remove_temp_blob(digest)
            raise

    return StreamingResponse(
        _stream_and_cache(),
        media_type="application/octet-stream",
        headers={"Docker-Content-Digest": digest},
    )
