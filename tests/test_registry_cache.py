from __future__ import annotations

import pytest

from llama_router.registry_cache.cache import (
    BlobCache,
    InvalidCacheKeyError,
    validate_digest,
    validate_name,
    validate_reference,
)

GOOD_DIGEST = "sha256:" + "ab" * 32


@pytest.mark.parametrize("name", ["library/llama3.2", "myorg/custom-model", "Phi3"])
def test_validate_name_accepts_registry_names(name):
    assert validate_name(name) == name


@pytest.mark.parametrize(
    "name", ["", "../etc", "library/../../x", "/abs", "a//b", "a/", "a b"]
)
def test_validate_name_rejects_traversal(name):
    with pytest.raises(InvalidCacheKeyError):
        validate_name(name)


@pytest.mark.parametrize("ref", ["latest", "7b-instruct-q4_K_M", GOOD_DIGEST])
def test_validate_reference_accepts_tags_and_digests(ref):
    assert validate_reference(ref) == ref


@pytest.mark.parametrize("ref", ["", "..", "../x", ".hidden", "a/b", "sha256:zz"])
def test_validate_reference_rejects_bad_values(ref):
    with pytest.raises(InvalidCacheKeyError):
        validate_reference(ref)


@pytest.mark.parametrize(
    "digest", ["", "abc", "sha256:../../x", "sha256:" + "A" * 64, "md5:" + "a" * 32]
)
def test_validate_digest_rejects_bad_values(digest):
    with pytest.raises(InvalidCacheKeyError):
        validate_digest(digest)


def test_blob_and_manifest_paths_stay_under_root(tmp_path):
    cache = BlobCache(str(tmp_path / "cache"))
    root = (tmp_path / "cache").resolve()
    assert cache.blob_path(GOOD_DIGEST).is_relative_to(root / "blobs" / "sha256")
    cache.save_manifest("library/llama3.2", "latest", b"{}")
    assert cache.get_manifest("library/llama3.2", "latest") == b"{}"
    assert (root / "manifests" / "library" / "llama3.2" / "_refs" / "latest").is_file()


def test_nested_names_and_tags_do_not_collide(tmp_path):
    cache = BlobCache(str(tmp_path / "cache"))
    cache.save_manifest("a", "b", b'{"n": 1}')
    cache.save_manifest("a/b", "latest", b'{"n": 2}')
    cache.save_manifest("a_b", "latest", b'{"n": 3}')
    assert cache.get_manifest("a", "b") == b'{"n": 1}'
    assert cache.get_manifest("a/b", "latest") == b'{"n": 2}'
    assert cache.get_manifest("a_b", "latest") == b'{"n": 3}'
    assert cache.stats()["manifest_count"] == 3


def test_legacy_flat_manifests_are_migrated(tmp_path):
    manifests = tmp_path / "cache" / "manifests"
    (manifests / "library_llama3.2").mkdir(parents=True)
    (manifests / "library_llama3.2" / "latest").write_bytes(b"{}")
    (manifests / "myorg_model").mkdir()
    (manifests / "myorg_model" / "7b").write_bytes(b"{}")
    cache = BlobCache(str(tmp_path / "cache"))
    assert cache.get_manifest("library/llama3.2", "latest") == b"{}"
    assert not (manifests / "library_llama3.2").exists()
    # Ambiguous legacy names are dropped and refetched on the next pull.
    assert not (manifests / "myorg_model").exists()
    assert cache.stats()["manifest_count"] == 1


def _put_blob(cache, data, mtime):
    import hashlib
    import os

    digest = "sha256:" + hashlib.sha256(data).hexdigest()
    cache.temp_blob_path(digest).write_bytes(data)
    cache.commit_blob(digest, digest)
    os.utime(cache.blob_path(digest), (mtime, mtime))
    return digest


def test_commit_blob_refuses_digest_mismatch(tmp_path):
    from llama_router.registry_cache.cache import DigestMismatchError

    cache = BlobCache(str(tmp_path / "cache"))
    cache.temp_blob_path(GOOD_DIGEST).write_bytes(b"not it")
    with pytest.raises(DigestMismatchError):
        cache.commit_blob(GOOD_DIGEST, "sha256:" + "00" * 32)
    assert not cache.has_blob(GOOD_DIGEST)
    assert not cache.temp_blob_path(GOOD_DIGEST).exists()


def test_lru_eviction_keeps_cache_under_max_bytes(tmp_path):
    cache = BlobCache(str(tmp_path / "cache"), max_bytes=250)
    old = _put_blob(cache, b"a" * 100, 1000)
    used = _put_blob(cache, b"b" * 100, 2000)
    cache.touch_blob(old)  # now the most recently used
    new = _put_blob(cache, b"c" * 100, 3000)
    assert cache.has_blob(old)
    assert not cache.has_blob(used)
    assert cache.has_blob(new)
    assert cache.stats()["blob_bytes"] <= 250


def test_eviction_spares_recently_used_blobs(tmp_path):
    # A blob handed out by a hit (touched) must survive while Ollama fetches
    # it, even if the cache is over its limit for a while.
    cache = BlobCache(str(tmp_path / "cache"), max_bytes=150)
    in_use = _put_blob(cache, b"a" * 100, 1000)
    cache.touch_blob(in_use)
    new = _put_blob(cache, b"b" * 100, 2000)
    assert cache.has_blob(in_use)
    assert cache.has_blob(new)
    # Once it goes stale, the next eviction catches up.
    import os

    os.utime(cache.blob_path(in_use), (1000, 1000))
    assert cache.evict() == [in_use]


def test_eviction_never_drops_the_new_blob(tmp_path):
    cache = BlobCache(str(tmp_path / "cache"), max_bytes=10)
    big = _put_blob(cache, b"z" * 100, 1)
    assert cache.has_blob(big)
    assert BlobCache(str(tmp_path / "other")).evict() == []


def test_cache_rejects_traversal_inputs(tmp_path):
    cache = BlobCache(str(tmp_path / "cache"))
    with pytest.raises(InvalidCacheKeyError):
        cache.blob_path("sha256:../../../etc/passwd")
    with pytest.raises(InvalidCacheKeyError):
        cache.temp_blob_path("sha256:../../../etc/passwd")
    with pytest.raises(InvalidCacheKeyError):
        cache.save_manifest("../outside", "latest", b"x")
    assert not (tmp_path / "outside").exists()


def test_path_under_rejects_escape_even_without_validation(tmp_path):
    cache = BlobCache(str(tmp_path / "cache"))
    with pytest.raises(InvalidCacheKeyError):
        BlobCache._path_under(cache._manifests_dir, "..", "..", "x")


def test_query_predicates_tolerate_invalid_keys(tmp_path):
    cache = BlobCache(str(tmp_path / "cache"))
    assert cache.has_blob("sha256:../../etc/passwd") is False
    assert cache.has_blob("md5:" + "a" * 32) is False
    assert cache.blob_size("not-a-digest") == 0
    assert cache.has_manifest("library/llama3.2", "../x") is False
    assert cache.get_manifest("../x", "latest") is None
    assert cache.is_model_fully_cached("library/llama3.2", "bad tag") is False


def test_listings_survive_stray_files_and_odd_digests(tmp_path):
    import json

    cache = BlobCache(str(tmp_path / "cache"))
    manifests = tmp_path / "cache" / "manifests"
    # A manifest whose layer digest is not sha256 and a stray file whose
    # name is not a valid tag; neither may abort the listing helpers.
    odd = {"config": {"digest": "md5:" + "a" * 32, "size": 1}, "layers": []}
    refs = manifests / "library" / "odd" / "_refs"
    refs.mkdir(parents=True)
    (refs / "latest").write_bytes(json.dumps(odd).encode())
    (refs / ".stray tag").write_bytes(b"{}")
    (manifests / "bad name!" / "_refs").mkdir(parents=True)
    (manifests / "bad name!" / "_refs" / "latest").write_bytes(b"{}")

    assert cache.cached_models() == set()
    details = cache.cached_model_details()
    assert any(d["name"] == "odd:latest" and not d["fully_cached"] for d in details)
    stats = cache.stats()
    assert stats["manifest_count"] == 2


def test_path_under_accepts_base_with_parent_segments(tmp_path):
    (tmp_path / "a").mkdir()
    base = tmp_path / "a" / ".." / "cache"
    got = BlobCache._path_under(base, "x")
    assert got == tmp_path / "cache" / "x"
    cache = BlobCache(str(base))
    assert cache.blob_path("sha256:" + "a" * 64).is_relative_to(tmp_path / "cache")
