"""Small in-memory TTL caches for routing hot paths."""

from __future__ import annotations

import time
from typing import Generic, TypeVar

T = TypeVar("T")


class TTLCache(Generic[T]):
    """Fixed-TTL string-keyed cache."""

    def __init__(self, ttl_seconds: float) -> None:
        self._ttl = ttl_seconds
        self._store: dict[str, tuple[float, T]] = {}

    def get(self, key: str) -> T | None:
        entry = self._store.get(key)
        if entry is None:
            return None
        if time.monotonic() - entry[0] >= self._ttl:
            del self._store[key]
            return None
        return entry[1]

    def set(self, key: str, value: T) -> None:
        self._store[key] = (time.monotonic(), value)

    def invalidate(self, key: str) -> None:
        self._store.pop(key, None)

    def invalidate_prefix(self, prefix: str) -> None:
        for k in list(self._store):
            if k.startswith(prefix):
                del self._store[k]

    def clear(self) -> None:
        self._store.clear()
