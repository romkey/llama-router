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
        cache.has_blob("sha256:../../../etc/passwd")
    with pytest.raises(InvalidCacheKey):
        cache.has_manifest("library/llama3.2", "../../secret")
    with pytest.raises(InvalidCacheKey):
        cache.save_manifest("../outside", "latest", b"x")
    assert not (tmp_path / "outside").exists()


def test_path_under_rejects_escape_even_without_validation(tmp_path):
    cache = BlobCache(str(tmp_path / "cache"))
    with pytest.raises(InvalidCacheKey):
        BlobCache._path_under(cache._manifests_dir, "..", "..", "x")
