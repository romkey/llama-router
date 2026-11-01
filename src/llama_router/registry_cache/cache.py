"""Disk-based content-addressed blob and manifest store for OCI registry caching."""

from __future__ import annotations

import contextlib
import hashlib
import json
import logging
import os
import re
import shutil
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# OCI distribution spec: a repository name is one or more lowercase path
# components separated by "/"; a tag is up to 128 chars of [A-Za-z0-9._-]
# starting with a word character; a digest is an algorithm and hex.  Ollama
# model names may carry upper-case letters, so name components accept them.
_NAME_COMPONENT = r"[A-Za-z0-9]+(?:[._-][A-Za-z0-9]+)*"
_NAME_RE = re.compile(rf"^{_NAME_COMPONENT}(?:/{_NAME_COMPONENT})*$")
_TAG_RE = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9._-]{0,127}$")
_DIGEST_RE = re.compile(r"^sha256:[0-9a-f]{64}$")


class InvalidCacheKeyError(ValueError):
    """A repository name, tag or digest that is not a valid OCI identifier."""


def validate_name(name: str) -> str:
    if not name or len(name) > 255 or not _NAME_RE.fullmatch(name):
        raise InvalidCacheKeyError(f"invalid repository name: {name!r}")
    return name


def validate_reference(reference: str) -> str:
    if not reference or not (
        _TAG_RE.fullmatch(reference) or _DIGEST_RE.fullmatch(reference)
    ):
        raise InvalidCacheKeyError(f"invalid reference: {reference!r}")
    return reference


def validate_digest(digest: str) -> str:
    if not digest or not _DIGEST_RE.fullmatch(digest):
        raise InvalidCacheKeyError(f"invalid digest: {digest!r}")
    return digest


class DigestMismatchError(ValueError):
    """Downloaded content whose sha256 does not match the digest it was fetched by."""


# Manifests live at manifests/<name components...>/_refs/<reference>.  Name
# components cannot start with "_", so "_refs" never collides with a name.
_REFS_DIR = "_refs"


def sha256_digest(data: bytes) -> str:
    """Return the ``sha256:<hex>`` digest of *data*."""
    return "sha256:" + hashlib.sha256(data).hexdigest()


class BlobCache:
    # Blobs used more recently than this are never evicted (see evict()).
    EVICT_GRACE_SECONDS = 15 * 60

    def __init__(
        self,
        cache_dir: str,
        manifest_ttl_hours: int = 240,
        max_bytes: int = 0,
    ):
        self._root = Path(os.path.normpath(Path(cache_dir).absolute()))
        self._blobs_dir = self._root / "blobs" / "sha256"
        self._manifests_dir = self._root / "manifests"
        self._manifest_ttl_seconds = manifest_ttl_hours * 3600
        self._max_bytes = max_bytes
        self._blobs_dir.mkdir(parents=True, exist_ok=True)
        self._manifests_dir.mkdir(parents=True, exist_ok=True)
        self._migrate_legacy_manifests()
        self.blob_hits: int = 0
        self.blob_misses: int = 0
        self.manifest_hits: int = 0
        self.manifest_misses: int = 0

    @staticmethod
    def _path_under(base: Path, *parts: str) -> Path:
        """Join *parts* onto *base* and refuse any result that escapes it.

        Callers validate their inputs first; this is the last line of defence
        against path traversal and the check CodeQL recognises as a barrier.
        """
        root = os.path.normpath(base.absolute())
        candidate = os.path.normpath(Path(root).joinpath(*parts))
        if candidate != root and not candidate.startswith(root + os.sep):
            raise InvalidCacheKeyError("path escapes cache directory")
        return Path(candidate)

    def _blob_path(self, digest: str) -> Path:
        """sha256:abc123... -> blobs/sha256/abc123..."""
        short = validate_digest(digest).removeprefix("sha256:")
        return self._path_under(self._blobs_dir, short)

    def _manifest_path(self, name: str, reference: str) -> Path:
        """library/llama3.2, latest -> manifests/library/llama3.2/_refs/latest"""
        components = validate_name(name).split("/")
        return self._path_under(
            self._manifests_dir,
            *components,
            _REFS_DIR,
            validate_reference(reference),
        )

    def _migrate_legacy_manifests(self) -> None:
        """Move manifests from the old flat layout into the nested one.

        Older versions stored ``library/llama3.2`` as
        ``manifests/library_llama3.2/<tag>``.  Only ``library_*`` maps back
        unambiguously ("_" is also allowed inside names); other legacy
        directories are dropped and their manifests are fetched again on the
        next pull.  Blobs are not affected.
        """
        for d in list(self._manifests_dir.iterdir()):
            if not d.is_dir():
                continue
            files = [f for f in d.iterdir() if f.is_file()]
            if not files:
                continue
            name = None
            if d.name.startswith("library_"):
                name = "library/" + d.name.removeprefix("library_")
            for f in files:
                target = None
                if name is not None:
                    with contextlib.suppress(InvalidCacheKeyError):
                        target = self._manifest_path(name, f.name)
                if target is not None and not target.exists():
                    target.parent.mkdir(parents=True, exist_ok=True)
                    f.replace(target)
                else:
                    f.unlink(missing_ok=True)
            with contextlib.suppress(OSError):
                d.rmdir()
            logger.info("Migrated legacy manifest directory %s", d.name)

    def _iter_manifest_files(self) -> Iterator[tuple[str, str, Path]]:
        """Yield (name, reference, path) for every stored manifest file."""
        for refs in sorted(self._manifests_dir.rglob(_REFS_DIR)):
            if not refs.is_dir():
                continue
            name = refs.parent.relative_to(self._manifests_dir).as_posix()
            if not _NAME_RE.fullmatch(name):
                continue
            for f in sorted(refs.iterdir(), key=lambda p: p.name):
                if f.is_file():
                    yield name, f.name, f

    def _is_expired(self, path: Path) -> bool:
        return time.time() - path.stat().st_mtime > self._manifest_ttl_seconds

    @staticmethod
    def _display_name(name: str) -> str:
        return name.removeprefix("library/")

    # --- Blobs ---

    def has_blob(self, digest: str) -> bool:
        """True if *digest* is a valid identifier with a cached blob.

        Digests read back from stored manifests or the filesystem may not be
        valid cache keys; such blobs are simply reported as not cached.
        """
        try:
            return self._blob_path(digest).is_file()
        except InvalidCacheKeyError:
            return False

    def blob_size(self, digest: str) -> int:
        try:
            p = self._blob_path(digest)
        except InvalidCacheKeyError:
            return 0
        return p.stat().st_size if p.is_file() else 0

    def blob_path(self, digest: str) -> Path:
        return self._blob_path(digest)

    def touch_blob(self, digest: str) -> None:
        """Mark a blob as recently used for LRU eviction (atime is unreliable)."""
        with contextlib.suppress(OSError, InvalidCacheKeyError):
            os.utime(self._blob_path(digest))

    def temp_blob_path(self, digest: str) -> Path:
        """Return a temp path for writing a blob atomically."""
        p = self._blob_path(digest)
        return p.with_suffix(".tmp")

    def commit_blob(self, digest: str, computed_digest: str) -> None:
        """Rename temp blob to final location after download completes.

        *computed_digest* is the sha256 of the bytes actually written, hashed
        while streaming.  On a mismatch the temp file is discarded and
        :class:`DigestMismatchError` raised, so the cache never serves bytes
        under a digest they don't have.
        """
        tmp = self.temp_blob_path(digest)
        final = self._blob_path(digest)
        if computed_digest != digest:
            tmp.unlink(missing_ok=True)
            raise DigestMismatchError(
                f"blob digest mismatch: expected {digest}, got {computed_digest}"
            )
        if tmp.is_file():
            tmp.rename(final)
            self.evict(keep=digest)

    def remove_temp_blob(self, digest: str) -> None:
        tmp = self.temp_blob_path(digest)
        tmp.unlink(missing_ok=True)

    def evict(self, keep: str | None = None) -> list[str]:
        """Delete least recently used blobs until the cache fits ``max_bytes``.

        *keep* (the blob just committed) is never evicted, nor is any blob
        used within ``EVICT_GRACE_SECONDS``: a hit hands Ollama a ``Location``
        it fetches in later (ranged) requests, each of which touches the blob,
        so a blob in use must survive until that pull finishes.  The cache can
        therefore exceed ``max_bytes`` while pulls are in flight.  Returns the
        evicted digests; a ``max_bytes`` of 0 means unlimited.
        """
        if self._max_bytes <= 0:
            return []
        recent = time.time() - self.EVICT_GRACE_SECONDS
        blobs: list[tuple[float, int, Path]] = []
        total = 0
        for f in self._blobs_dir.iterdir():
            if not f.is_file() or f.name.endswith(".tmp"):
                continue
            st = f.stat()
            total += st.st_size
            blobs.append((st.st_mtime, st.st_size, f))
        keep_name = keep.removeprefix("sha256:") if keep else None
        evicted: list[str] = []
        for mtime, size, f in sorted(blobs, key=lambda b: b[0]):
            if total <= self._max_bytes or mtime >= recent:
                break
            if f.name == keep_name:
                continue
            f.unlink(missing_ok=True)
            total -= size
            evicted.append(f"sha256:{f.name}")
        if evicted:
            logger.info(
                "Evicted %d blob(s) to keep the cache under %d bytes",
                len(evicted),
                self._max_bytes,
            )
        return evicted

    # --- Manifests ---

    def has_manifest(self, name: str, reference: str) -> bool:
        """True if a non-expired manifest is cached under a valid name and tag."""
        try:
            p = self._manifest_path(name, reference)
        except InvalidCacheKeyError:
            return False
        if not p.is_file():
            return False
        if self._is_expired(p):
            p.unlink(missing_ok=True)
            return False
        return True

    def get_manifest(self, name: str, reference: str) -> bytes | None:
        if not self.has_manifest(name, reference):
            return None
        return self._manifest_path(name, reference).read_bytes()

    def save_manifest(self, name: str, reference: str, data: bytes) -> None:
        p = self._manifest_path(name, reference)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)

    def all_cached_manifests(self) -> Iterator[bytes]:
        """Yield raw bytes of every non-expired cached manifest."""
        for _name, _ref, f in self._iter_manifest_files():
            if not self._is_expired(f):
                yield f.read_bytes()

    def _manifest_blob_digests(self, manifest_bytes: bytes) -> list[str]:
        """Extract all blob digests from a manifest."""
        try:
            manifest = json.loads(manifest_bytes)
        except Exception:
            return []
        digests = []
        config = manifest.get("config", {})
        if config.get("digest"):
            digests.append(config["digest"])
        for layer in manifest.get("layers", []):
            if layer.get("digest"):
                digests.append(layer["digest"])
        return digests

    def is_model_fully_cached(self, name: str, reference: str) -> bool:
        """Check that a model's manifest AND all its blobs are present."""
        manifest_data = self.get_manifest(name, reference)
        if manifest_data is None:
            return False
        for digest in self._manifest_blob_digests(manifest_data):
            if not self.has_blob(digest):
                return False
        return True

    def cached_models(self) -> set[str]:
        """Return model names where the manifest AND all blobs are cached."""
        result: set[str] = set()
        for name, tag, f in self._iter_manifest_files():
            if self._is_expired(f):
                continue
            if self.is_model_fully_cached(name, tag):
                result.add(f"{self._display_name(name)}:{tag}")
        return result

    def cached_model_details(self) -> list[dict[str, Any]]:
        """Return detailed info for each model with a cached manifest.

        Each entry: {name, fully_cached, blobs: [{digest, size, cached}, ...]}
        """
        results: list[dict[str, Any]] = []
        for name, tag, f in self._iter_manifest_files():
            if self._is_expired(f):
                continue
            manifest_data = f.read_bytes()
            try:
                manifest = json.loads(manifest_data)
            except Exception:
                logger.debug("skipping unparseable manifest %s", f, exc_info=True)
                continue
            blobs: list[dict[str, Any]] = []
            all_cached = True
            config = manifest.get("config", {})
            entries = []
            if config.get("digest"):
                entries.append((config["digest"], config.get("size", 0), "config"))
            for layer in manifest.get("layers", []):
                if layer.get("digest"):
                    media = layer.get("mediaType", "")
                    label = media.rsplit(".", 1)[-1] if "." in media else "layer"
                    entries.append((layer["digest"], layer.get("size", 0), label))
            for digest, size, label in entries:
                cached = self.has_blob(digest)
                if not cached:
                    all_cached = False
                blobs.append(
                    {
                        "digest": digest,
                        "size": size,
                        "cached": cached,
                        "label": label,
                    }
                )
            results.append(
                {
                    "name": f"{self._display_name(name)}:{tag}",
                    "fully_cached": all_cached,
                    "blobs": blobs,
                }
            )
        return results

    # --- Stats ---

    def stats(self) -> dict[str, Any]:
        blob_count = 0
        blob_bytes = 0
        for f in self._blobs_dir.iterdir():
            if f.is_file() and not f.name.endswith(".tmp"):
                blob_count += 1
                blob_bytes += f.stat().st_size

        manifest_count = sum(1 for _ in self._iter_manifest_files())

        return {
            "blob_count": blob_count,
            "blob_bytes": blob_bytes,
            "max_bytes": self._max_bytes,
            "manifest_count": manifest_count,
            "cache_dir": str(self._root),
            "blob_hits": self.blob_hits,
            "blob_misses": self.blob_misses,
            "manifest_hits": self.manifest_hits,
            "manifest_misses": self.manifest_misses,
            "cached_models": sorted(self.cached_models()),
            "model_details": self.cached_model_details(),
        }

    def clear(self) -> None:
        """Remove all cached blobs and manifests."""
        if self._blobs_dir.exists():
            shutil.rmtree(self._blobs_dir)
            self._blobs_dir.mkdir(parents=True, exist_ok=True)
        if self._manifests_dir.exists():
            shutil.rmtree(self._manifests_dir)
            self._manifests_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Registry cache cleared")
