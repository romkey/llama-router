"""Disk-based content-addressed blob and manifest store for OCI registry caching."""

from __future__ import annotations

import logging
import shutil
import time
from pathlib import Path

logger = logging.getLogger(__name__)


class BlobCache:
    def __init__(self, cache_dir: str, manifest_ttl_hours: int = 240):
        self._root = Path(cache_dir)
        self._blobs_dir = self._root / "blobs" / "sha256"
        self._manifests_dir = self._root / "manifests"
        self._manifest_ttl_seconds = manifest_ttl_hours * 3600
        self._blobs_dir.mkdir(parents=True, exist_ok=True)
        self._manifests_dir.mkdir(parents=True, exist_ok=True)
        self.blob_hits: int = 0
        self.blob_misses: int = 0
        self.manifest_hits: int = 0
        self.manifest_misses: int = 0

    def _blob_path(self, digest: str) -> Path:
        """sha256:abc123... -> blobs/sha256/abc123..."""
        short = digest.removeprefix("sha256:")
        return self._blobs_dir / short

    def _manifest_path(self, name: str, reference: str) -> Path:
        safe_name = name.replace("/", "_")
        return self._manifests_dir / safe_name / reference

    # --- Blobs ---

    def has_blob(self, digest: str) -> bool:
        return self._blob_path(digest).is_file()

    def blob_size(self, digest: str) -> int:
        p = self._blob_path(digest)
        return p.stat().st_size if p.is_file() else 0

    def blob_path(self, digest: str) -> Path:
        return self._blob_path(digest)

    def temp_blob_path(self, digest: str) -> Path:
        """Return a temp path for writing a blob atomically."""
        p = self._blob_path(digest)
        return p.with_suffix(".tmp")

    def commit_blob(self, digest: str) -> None:
        """Rename temp blob to final location after download completes."""
        tmp = self.temp_blob_path(digest)
        final = self._blob_path(digest)
        if tmp.is_file():
            tmp.rename(final)

    def remove_temp_blob(self, digest: str) -> None:
        tmp = self.temp_blob_path(digest)
        tmp.unlink(missing_ok=True)

    # --- Manifests ---

    def has_manifest(self, name: str, reference: str) -> bool:
        p = self._manifest_path(name, reference)
        if not p.is_file():
            return False
        age = time.time() - p.stat().st_mtime
        if age > self._manifest_ttl_seconds:
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

    def cached_models(self) -> set[str]:
        """Return set of cached model names in Ollama format (e.g. 'llama3.2:latest')."""
        result: set[str] = set()
        if not self._manifests_dir.exists():
            return result
        for d in self._manifests_dir.iterdir():
            if not d.is_dir():
                continue
            name = d.name
            if name.startswith("library_"):
                name = name[len("library_") :]
            else:
                name = name.replace("_", "/", 1)
            for f in d.iterdir():
                if f.is_file():
                    tag = f.name
                    age = time.time() - f.stat().st_mtime
                    if age <= self._manifest_ttl_seconds:
                        result.add(f"{name}:{tag}")
        return result

    # --- Stats ---

    def stats(self) -> dict:
        blob_count = 0
        blob_bytes = 0
        for f in self._blobs_dir.iterdir():
            if f.is_file() and not f.name.endswith(".tmp"):
                blob_count += 1
                blob_bytes += f.stat().st_size

        manifest_count = 0
        for d in self._manifests_dir.iterdir():
            if d.is_dir():
                manifest_count += sum(1 for f in d.iterdir() if f.is_file())

        return {
            "blob_count": blob_count,
            "blob_bytes": blob_bytes,
            "manifest_count": manifest_count,
            "cache_dir": str(self._root),
            "blob_hits": self.blob_hits,
            "blob_misses": self.blob_misses,
            "manifest_hits": self.manifest_hits,
            "manifest_misses": self.manifest_misses,
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
