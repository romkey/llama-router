from __future__ import annotations

import asyncio
import hashlib
import json
from collections.abc import AsyncIterator
from pathlib import Path

import httpx
import pytest
from pytest_httpx import HTTPXMock

from llama_router.config import settings
from llama_router.registry_cache import app as rc
from llama_router.registry_cache.cache import BlobCache, InvalidCacheKeyError

BLOB = b"x" * 1000
CFG = b"y" * 10
DIGEST = "sha256:" + hashlib.sha256(BLOB).hexdigest()
CFG_DIGEST = "sha256:" + hashlib.sha256(CFG).hexdigest()
MANIFEST = json.dumps(
    {
        "config": {"digest": CFG_DIGEST, "size": 10},
        "layers": [{"digest": DIGEST, "size": len(BLOB)}],
    }
).encode()
UP = rc.UPSTREAM


@pytest.fixture
async def env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> AsyncIterator[tuple[BlobCache, httpx.AsyncClient]]:
    cache = BlobCache(str(tmp_path / "cache"))
    upstream = httpx.AsyncClient(follow_redirects=True)
    monkeypatch.setattr(rc, "_cache", cache)
    monkeypatch.setattr(rc, "_upstream_client", upstream)
    monkeypatch.setattr(rc, "_blob_semaphore", asyncio.Semaphore(2))
    monkeypatch.setattr(rc, "_blob_sizes", {})
    monkeypatch.setattr(rc, "_active_downloads", {})
    monkeypatch.setattr(settings, "cache_enabled", True)
    monkeypatch.setattr(settings, "cache_token", "")
    monkeypatch.setattr(settings, "cache_external_host", "")
    client = httpx.AsyncClient(
        transport=httpx.ASGITransport(app=rc.app), base_url="http://cache"
    )
    yield cache, client
    await client.aclose()
    await upstream.aclose()


def _commit(cache: BlobCache, digest: str, data: bytes) -> None:
    cache.temp_blob_path(digest).write_bytes(data)
    cache.commit_blob(digest, "sha256:" + hashlib.sha256(data).hexdigest())


def test_human_bytes_units() -> None:
    assert rc._human_bytes(2048) == "2 KB"
    assert rc._human_bytes(5 << 20) == "5.0 MB"
    assert rc._human_bytes(3 << 30) == "3.0 GB"


def test_extract_blob_sizes_ignores_garbage(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(rc, "_blob_sizes", {})
    rc._extract_blob_sizes(b"not json")
    assert rc._blob_sizes == {}
    rc._extract_blob_sizes(MANIFEST)
    assert rc._blob_sizes == {DIGEST: len(BLOB), CFG_DIGEST: 10}


def test_init_cache_prewarms_sizes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(rc, "_blob_sizes", {})
    cache = BlobCache(str(tmp_path / "c"))
    cache.save_manifest("library/m", "latest", MANIFEST)
    rc.init_cache(cache)
    assert rc._blob_sizes[DIGEST] == len(BLOB)


async def test_v2_check_and_invalid_name(
    env: tuple[BlobCache, httpx.AsyncClient],
) -> None:
    _, client = env
    assert (await client.get("/v2/")).json() == {}
    resp = await client.get("/v2/bad name/manifests/latest")
    assert resp.status_code == 400


async def test_manifest_miss_then_hit(
    env: tuple[BlobCache, httpx.AsyncClient], httpx_mock: HTTPXMock
) -> None:
    cache, client = env
    httpx_mock.add_response(url=f"{UP}/v2/library/m/manifests/latest", content=MANIFEST)
    first = await client.get("/v2/library/m/manifests/latest")
    assert first.status_code == 200
    assert first.content == MANIFEST
    assert cache.manifest_misses == 1
    second = await client.get("/v2/library/m/manifests/latest")
    assert second.content == MANIFEST
    assert cache.manifest_hits == 1


@pytest.mark.parametrize(("up", "expected"), [(404, 404), (500, 502)])
async def test_manifest_upstream_errors(
    env: tuple[BlobCache, httpx.AsyncClient],
    httpx_mock: HTTPXMock,
    up: int,
    expected: int,
) -> None:
    _, client = env
    httpx_mock.add_response(status_code=up)
    resp = await client.get("/v2/library/m/manifests/latest")
    assert resp.status_code == expected


async def test_manifest_upstream_unreachable(
    env: tuple[BlobCache, httpx.AsyncClient], httpx_mock: HTTPXMock
) -> None:
    _, client = env
    httpx_mock.add_exception(httpx.ConnectError("down"))
    resp = await client.get("/v2/library/m/manifests/latest")
    assert resp.status_code == 502


async def test_head_blob_local_manifest_size_and_upstream(
    env: tuple[BlobCache, httpx.AsyncClient], httpx_mock: HTTPXMock
) -> None:
    cache, client = env
    _commit(cache, DIGEST, BLOB)
    resp = await client.head(f"/v2/library/m/blobs/{DIGEST}")
    assert resp.headers["content-length"] == str(len(BLOB))

    other = "sha256:" + "11" * 32
    rc._blob_sizes[other] = 42
    resp = await client.head(f"/v2/library/m/blobs/{other}")
    assert resp.headers["content-length"] == "42"

    third = "sha256:" + "22" * 32
    httpx_mock.add_response(
        method="HEAD",
        url=f"{UP}/v2/library/m/blobs/{third}",
        headers={"content-length": "77"},
    )
    resp = await client.head(f"/v2/library/m/blobs/{third}")
    assert resp.headers["content-length"] == "77"
    assert rc._blob_sizes[third] == 77


async def test_head_blob_upstream_failures(
    env: tuple[BlobCache, httpx.AsyncClient], httpx_mock: HTTPXMock
) -> None:
    _, client = env
    d = "sha256:" + "33" * 32
    httpx_mock.add_response(method="HEAD", status_code=404)
    assert (await client.head(f"/v2/library/m/blobs/{d}")).status_code == 404
    httpx_mock.add_response(method="HEAD", status_code=503)
    assert (await client.head(f"/v2/library/m/blobs/{d}")).status_code == 502
    httpx_mock.add_exception(httpx.ConnectError("down"))
    assert (await client.head(f"/v2/library/m/blobs/{d}")).status_code == 502


async def test_get_blob_miss_streams_and_caches(
    env: tuple[BlobCache, httpx.AsyncClient], httpx_mock: HTTPXMock
) -> None:
    cache, client = env
    httpx_mock.add_response(url=f"{UP}/v2/library/m/blobs/{DIGEST}", content=BLOB)
    resp = await client.get(f"/v2/library/m/blobs/{DIGEST}")
    assert resp.content == BLOB
    assert cache.has_blob(DIGEST)
    assert cache.blob_misses == 1
    assert rc._active_downloads == {}

    hit = await client.get(f"/v2/library/m/blobs/{DIGEST}")
    assert hit.status_code == 200
    assert hit.content == b""
    assert hit.headers["location"] == f"/cache/blobs/{DIGEST}"
    assert cache.blob_hits == 1


async def test_get_blob_upstream_failure_cleans_up(
    env: tuple[BlobCache, httpx.AsyncClient], httpx_mock: HTTPXMock
) -> None:
    cache, client = env
    httpx_mock.add_response(status_code=500)
    with pytest.raises(httpx.HTTPStatusError):
        await client.get(f"/v2/library/m/blobs/{DIGEST}")
    assert not cache.has_blob(DIGEST)
    assert rc._active_downloads == {}


async def test_get_blob_waits_for_active_download(
    env: tuple[BlobCache, httpx.AsyncClient],
) -> None:
    cache, client = env
    event = asyncio.Event()
    rc._active_downloads[DIGEST] = event
    task = asyncio.create_task(client.get(f"/v2/library/m/blobs/{DIGEST}"))
    await asyncio.sleep(0.01)
    assert not task.done()
    _commit(cache, DIGEST, BLOB)
    event.set()
    resp = await task
    assert resp.status_code == 200
    assert "location" in resp.headers


async def test_get_blob_wait_finishes_without_blob(
    env: tuple[BlobCache, httpx.AsyncClient],
) -> None:
    _, client = env
    event = asyncio.Event()
    event.set()
    rc._active_downloads[DIGEST] = event
    resp = await client.get(f"/v2/library/m/blobs/{DIGEST}")
    assert resp.status_code == 502


async def test_serve_cached_blob(env: tuple[BlobCache, httpx.AsyncClient]) -> None:
    cache, client = env
    assert (await client.get(f"/cache/blobs/{DIGEST}")).status_code == 404
    _commit(cache, DIGEST, BLOB)
    resp = await client.get(f"/cache/blobs/{DIGEST}")
    assert resp.content == BLOB
    ranged = await client.get(f"/cache/blobs/{DIGEST}", headers={"Range": "bytes=0-9"})
    assert ranged.status_code == 206
    assert ranged.content == BLOB[:10]


async def test_precache_model_downloads_all_blobs(
    env: tuple[BlobCache, httpx.AsyncClient], httpx_mock: HTTPXMock
) -> None:
    cache, _ = env
    httpx_mock.add_response(
        url=f"{UP}/v2/library/llama3/manifests/latest", content=MANIFEST
    )
    httpx_mock.add_response(url=f"{UP}/v2/library/llama3/blobs/{DIGEST}", content=BLOB)
    httpx_mock.add_response(
        url=f"{UP}/v2/library/llama3/blobs/{CFG_DIGEST}", content=CFG
    )
    messages: list[str] = []
    await rc.precache_model(cache, "llama3", messages.append)
    assert cache.has_blob(DIGEST)
    assert cache.has_blob(CFG_DIGEST)
    assert any("cached llama3" in m for m in messages)

    # Second run: manifest refetched but blobs already cached.
    httpx_mock.add_response(
        url=f"{UP}/v2/library/llama3/manifests/latest", content=MANIFEST
    )
    messages.clear()
    await rc.precache_model(cache, "llama3:latest", messages.append)
    assert any("already cached" in m for m in messages)


async def test_precache_manifest_failure(
    env: tuple[BlobCache, httpx.AsyncClient], httpx_mock: HTTPXMock
) -> None:
    cache, _ = env
    httpx_mock.add_response(status_code=404)
    with pytest.raises(RuntimeError, match="HTTP 404"):
        await rc.precache_model(cache, "org/model:7b")


async def test_precache_blob_failure_removes_temp(
    env: tuple[BlobCache, httpx.AsyncClient], httpx_mock: HTTPXMock
) -> None:
    cache, _ = env
    httpx_mock.add_response(url=f"{UP}/v2/org/model/manifests/7b", content=MANIFEST)
    httpx_mock.add_response(url=f"{UP}/v2/org/model/blobs/{DIGEST}", status_code=500)
    httpx_mock.add_response(url=f"{UP}/v2/org/model/blobs/{CFG_DIGEST}", content=CFG)
    with pytest.raises(RuntimeError, match="Blob download failed"):
        await rc.precache_model(cache, "org/model:7b")
    assert not cache.has_blob(DIGEST)


async def test_lifespan_sets_and_clears_clients() -> None:
    async with rc._lifespan(rc.app):
        assert rc._upstream_client is not None
        assert rc._blob_semaphore is not None
    assert rc._upstream_client is None


async def test_disabled_cache_returns_503(
    env: tuple[BlobCache, httpx.AsyncClient], monkeypatch: pytest.MonkeyPatch
) -> None:
    _, client = env
    monkeypatch.setattr(settings, "cache_enabled", False)
    assert (await client.get("/v2/")).status_code == 503
    assert (await client.get(f"/cache/blobs/{DIGEST}")).status_code == 503


async def test_cache_token_required_when_set(
    env: tuple[BlobCache, httpx.AsyncClient], monkeypatch: pytest.MonkeyPatch
) -> None:
    cache, client = env
    _commit(cache, DIGEST, BLOB)
    monkeypatch.setattr(settings, "cache_token", "s3cret")
    resp = await client.get("/v2/")
    assert resp.status_code == 401
    assert resp.headers["www-authenticate"] == "Bearer"
    bad = {"Authorization": "Bearer nope"}
    assert (await client.get(f"/cache/blobs/{DIGEST}", headers=bad)).status_code == 401
    good = {"Authorization": "Bearer s3cret"}
    assert (await client.get("/v2/", headers=good)).status_code == 200
    resp = await client.get(f"/cache/blobs/{DIGEST}", headers=good)
    assert resp.content == BLOB


async def test_openapi_and_docs_are_disabled(
    env: tuple[BlobCache, httpx.AsyncClient],
) -> None:
    _, client = env
    for path in ("/docs", "/redoc", "/openapi.json"):
        assert (await client.get(path)).status_code == 404


async def test_blob_location_uses_external_host_not_host_header(
    env: tuple[BlobCache, httpx.AsyncClient], monkeypatch: pytest.MonkeyPatch
) -> None:
    cache, client = env
    _commit(cache, DIGEST, BLOB)
    headers = {"Host": "evil.example"}
    resp = await client.get(f"/v2/library/m/blobs/{DIGEST}", headers=headers)
    assert resp.headers["location"] == f"/cache/blobs/{DIGEST}"
    monkeypatch.setattr(settings, "cache_external_host", "192.168.1.50")
    resp = await client.get(f"/v2/library/m/blobs/{DIGEST}", headers=headers)
    assert resp.headers["location"] == (
        f"http://192.168.1.50:{settings.cache_port}/cache/blobs/{DIGEST}"
    )


async def test_get_blob_digest_mismatch_is_not_cached(
    env: tuple[BlobCache, httpx.AsyncClient], httpx_mock: HTTPXMock
) -> None:
    cache, client = env
    httpx_mock.add_response(url=f"{UP}/v2/library/m/blobs/{DIGEST}", content=b"evil")
    with pytest.raises(ValueError, match="digest mismatch"):
        await client.get(f"/v2/library/m/blobs/{DIGEST}")
    assert not cache.has_blob(DIGEST)
    assert not cache.temp_blob_path(DIGEST).exists()
    assert rc._active_downloads == {}


async def test_manifest_by_digest_is_verified(
    env: tuple[BlobCache, httpx.AsyncClient], httpx_mock: HTTPXMock
) -> None:
    cache, client = env
    ref = "sha256:" + hashlib.sha256(MANIFEST).hexdigest()
    httpx_mock.add_response(url=f"{UP}/v2/library/m/manifests/{ref}", content=MANIFEST)
    assert (await client.get(f"/v2/library/m/manifests/{ref}")).status_code == 200

    wrong = "sha256:" + "00" * 32
    httpx_mock.add_response(
        url=f"{UP}/v2/library/m/manifests/{wrong}", content=MANIFEST
    )
    assert (await client.get(f"/v2/library/m/manifests/{wrong}")).status_code == 502
    assert cache.get_manifest("library/m", wrong) is None


async def test_precache_validates_name_before_fetching(
    env: tuple[BlobCache, httpx.AsyncClient],
) -> None:
    cache, _ = env
    # No upstream response registered: any request would fail the test.
    with pytest.raises(InvalidCacheKeyError):
        await rc.precache_model(cache, "../../evil:latest")
    with pytest.raises(InvalidCacheKeyError):
        await rc.precache_model(cache, "llama3:bad/tag")


async def test_precache_digest_mismatch_fails(
    env: tuple[BlobCache, httpx.AsyncClient], httpx_mock: HTTPXMock
) -> None:
    cache, _ = env
    httpx_mock.add_response(url=f"{UP}/v2/org/model/manifests/7b", content=MANIFEST)
    httpx_mock.add_response(url=f"{UP}/v2/org/model/blobs/{DIGEST}", content=b"evil")
    httpx_mock.add_response(url=f"{UP}/v2/org/model/blobs/{CFG_DIGEST}", content=CFG)
    with pytest.raises(RuntimeError, match="digest mismatch"):
        await rc.precache_model(cache, "org/model:7b")
    assert not cache.has_blob(DIGEST)
    assert not cache.temp_blob_path(DIGEST).exists()


class _Stop(BaseException):
    pass


async def test_precache_cancellation_removes_temp(
    env: tuple[BlobCache, httpx.AsyncClient], httpx_mock: HTTPXMock
) -> None:
    cache, _ = env
    manifest = json.dumps({"layers": [{"digest": DIGEST, "size": len(BLOB)}]})
    httpx_mock.add_response(
        url=f"{UP}/v2/library/m/manifests/latest", content=manifest.encode()
    )
    httpx_mock.add_response(url=f"{UP}/v2/library/m/blobs/{DIGEST}", content=BLOB)

    def _progress(msg: str) -> None:
        if "%" in msg:
            raise _Stop

    with pytest.raises(_Stop):
        await rc.precache_model(cache, "m", _progress)
    assert not cache.temp_blob_path(DIGEST).exists()
    assert not cache.has_blob(DIGEST)
