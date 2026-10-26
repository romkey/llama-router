from __future__ import annotations

import pytest

from llama_router.registry_cache.cache import (
    BlobCache,
    InvalidCacheKey,
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
    with pytest.raises(InvalidCacheKey):
        validate_name(name)


@pytest.mark.parametrize("ref", ["latest", "7b-instruct-q4_K_M", GOOD_DIGEST])
def test_validate_reference_accepts_tags_and_digests(ref):
    assert validate_reference(ref) == ref


@pytest.mark.parametrize("ref", ["", "..", "../x", ".hidden", "a/b", "sha256:zz"])
def test_validate_reference_rejects_bad_values(ref):
    with pytest.raises(InvalidCacheKey):
        validate_reference(ref)


@pytest.mark.parametrize(
    "digest", ["", "abc", "sha256:../../x", "sha256:" + "A" * 64, "md5:" + "a" * 32]
)
def test_validate_digest_rejects_bad_values(digest):
    with pytest.raises(InvalidCacheKey):
        validate_digest(digest)


def test_blob_and_manifest_paths_stay_under_root(tmp_path):
    cache = BlobCache(str(tmp_path / "cache"))
    root = (tmp_path / "cache").resolve()
    assert cache.blob_path(GOOD_DIGEST).is_relative_to(root / "blobs" / "sha256")
    cache.save_manifest("library/llama3.2", "latest", b"{}")
    assert cache.get_manifest("library/llama3.2", "latest") == b"{}"
    assert (root / "manifests" / "library_llama3.2" / "latest").is_file()


def test_cache_rejects_traversal_inputs(tmp_path):
    cache = BlobCache(str(tmp_path / "cache"))
    with pytest.raises(InvalidCacheKey):
        cache.blob_path("sha256:../../../etc/passwd")
    with pytest.raises(InvalidCacheKey):
        cache.temp_blob_path("sha256:../../../etc/passwd")
    with pytest.raises(InvalidCacheKey):
        cache.save_manifest("../outside", "latest", b"x")
    assert not (tmp_path / "outside").exists()


def test_path_under_rejects_escape_even_without_validation(tmp_path):
    cache = BlobCache(str(tmp_path / "cache"))
    with pytest.raises(InvalidCacheKey):
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
    (manifests / "library_odd").mkdir(parents=True)
    (manifests / "library_odd" / "latest").write_bytes(json.dumps(odd).encode())
    (manifests / "library_odd" / ".stray tag").write_bytes(b"{}")
    (manifests / "bad name!").mkdir()
    (manifests / "bad name!" / "latest").write_bytes(b"{}")

    assert cache.cached_models() == set()
    details = cache.cached_model_details()
    assert any(d["name"] == "odd:latest" and not d["fully_cached"] for d in details)
    stats = cache.stats()
    assert stats["manifest_count"] == 3
