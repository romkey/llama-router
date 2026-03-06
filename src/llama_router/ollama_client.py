"""Async HTTP client for communicating with Ollama backends."""

from __future__ import annotations

import time
from collections.abc import AsyncIterator
from typing import Any

import httpx

from .models import ProviderModel

_TIMEOUT = httpx.Timeout(10.0, read=120.0)


class OllamaClient:
    def __init__(self, base_url: str):
        self._base_url = base_url.rstrip("/")
        self._http = httpx.AsyncClient(base_url=self._base_url, timeout=_TIMEOUT)

    async def close(self) -> None:
        await self._http.aclose()

    async def is_reachable(self) -> bool:
        try:
            resp = await self._http.get("/api/version")
            return resp.status_code == 200
        except httpx.HTTPError:
            return False

    async def get_version(self) -> dict | None:
        try:
            resp = await self._http.get("/api/version")
            resp.raise_for_status()
            return resp.json()
        except httpx.HTTPError:
            return None

    async def get_tags(self) -> list[ProviderModel]:
        resp = await self._http.get("/api/tags")
        resp.raise_for_status()
        data = resp.json()
        return [
            ProviderModel(
                provider_id=0,
                name=m["name"],
                size=m.get("size"),
                digest=m.get("digest"),
                modified_at=m.get("modified_at"),
                details=m.get("details"),
            )
            for m in data.get("models", [])
        ]

    async def get_ps(self) -> list[dict]:
        resp = await self._http.get("/api/ps")
        resp.raise_for_status()
        return resp.json().get("models", [])

    async def show(self, model: str) -> dict:
        resp = await self._http.post("/api/show", json={"model": model})
        resp.raise_for_status()
        return resp.json()

    async def chat_stream(self, body: dict) -> AsyncIterator[bytes]:
        body["stream"] = True
        async with self._http.stream("POST", "/api/chat", json=body) as resp:
            resp.raise_for_status()
            async for chunk in resp.aiter_bytes():
                yield chunk

    async def chat(self, body: dict) -> dict:
        body["stream"] = False
        resp = await self._http.post("/api/chat", json=body)
        resp.raise_for_status()
        return resp.json()

    async def embeddings(self, body: dict) -> dict:
        resp = await self._http.post("/api/embeddings", json=body)
        resp.raise_for_status()
        return resp.json()

    async def embed(self, body: dict) -> dict:
        resp = await self._http.post("/api/embed", json=body)
        resp.raise_for_status()
        return resp.json()

    async def pull_stream(
        self,
        model: str,
        cache_registry_url: str | None = None,
    ) -> AsyncIterator[bytes]:
        pull_model = model
        insecure = False
        if cache_registry_url:
            base = cache_registry_url.rstrip("/")
            if "/" not in model or model.startswith("library/"):
                pull_model = f"{base}/library/{model}"
            else:
                pull_model = f"{base}/{model}"
            insecure = True

        async with self._http.stream(
            "POST",
            "/api/pull",
            json={"model": pull_model, "stream": True, "insecure": insecure},
            timeout=httpx.Timeout(10.0, read=600.0),
        ) as resp:
            resp.raise_for_status()
            async for chunk in resp.aiter_bytes():
                yield chunk

    async def delete_model(self, model: str) -> None:
        resp = await self._http.request("DELETE", "/api/delete", json={"model": model})
        resp.raise_for_status()

    async def benchmark_chat(self, model: str, prompt: str) -> dict[str, float]:
        """Run a simple benchmark returning startup_time_ms and tokens_per_second."""
        body = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
        }
        start = time.monotonic()
        resp = await self._http.post(
            "/api/chat",
            json=body,
            timeout=httpx.Timeout(30.0, read=120.0),
        )
        elapsed = time.monotonic() - start
        resp.raise_for_status()
        data = resp.json()

        total_duration_ns = data.get("total_duration", 0)
        load_duration_ns = data.get("load_duration", 0)
        eval_count = data.get("eval_count", 0)
        eval_duration_ns = data.get("eval_duration", 1)

        startup_ms = load_duration_ns / 1_000_000
        tps = (eval_count / eval_duration_ns * 1_000_000_000) if eval_count else 0

        return {"startup_time_ms": startup_ms, "tokens_per_second": tps}
