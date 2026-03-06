"""Disk-based content-addressed blob and manifest store for OCI registry caching."""

from __future__ import annotations

import json
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
        if not self._manifests_dir.exists():
            return result
        for d in self._manifests_dir.iterdir():
            if not d.is_dir():
                continue
            oci_name = d.name.replace("_", "/", 1)
            display_name = d.name
            if display_name.startswith("library_"):
                display_name = display_name[len("library_") :]
            else:
                display_name = display_name.replace("_", "/", 1)
            for f in d.iterdir():
                if not f.is_file():
                    continue
                tag = f.name
                age = time.time() - f.stat().st_mtime
                if age > self._manifest_ttl_seconds:
                    continue
                if self.is_model_fully_cached(oci_name, tag):
                    result.add(f"{display_name}:{tag}")
        return result

    def cached_model_details(self) -> list[dict]:
        """Return detailed info for each model with a cached manifest.

        Each entry: {name, fully_cached, blobs: [{digest, size, cached}, ...]}
        """
        results: list[dict] = []
        if not self._manifests_dir.exists():
            return results
        for d in sorted(self._manifests_dir.iterdir(), key=lambda p: p.name):
            if not d.is_dir():
                continue
            display_name = d.name
            if display_name.startswith("library_"):
                display_name = display_name[len("library_") :]
            else:
                display_name = display_name.replace("_", "/", 1)
            for f in sorted(d.iterdir(), key=lambda p: p.name):
                if not f.is_file():
                    continue
                tag = f.name
                age = time.time() - f.stat().st_mtime
                if age > self._manifest_ttl_seconds:
                    continue
                manifest_data = f.read_bytes()
                try:
                    manifest = json.loads(manifest_data)
                except Exception:
                    continue
                blobs: list[dict] = []
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
                        "name": f"{display_name}:{tag}",
                        "fully_cached": all_cached,
                        "blobs": blobs,
                    }
                )
        return results

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
