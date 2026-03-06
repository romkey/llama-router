"""Async HTTP client for communicating with llama.cpp server backends."""

from __future__ import annotations

import time
from collections.abc import AsyncIterator

import httpx

from .models import ProviderModel

_TIMEOUT = httpx.Timeout(10.0, read=120.0)


class LlamaCppClient:
    def __init__(self, base_url: str):
        self._base_url = base_url.rstrip("/")
        self._http = httpx.AsyncClient(base_url=self._base_url, timeout=_TIMEOUT)

    async def close(self) -> None:
        await self._http.aclose()

    async def is_reachable(self) -> bool:
        try:
            resp = await self._http.get("/health")
            return resp.status_code == 200
        except httpx.HTTPError:
            return False

    async def get_models(self) -> list[ProviderModel]:
        resp = await self._http.get("/v1/models")
        resp.raise_for_status()
        data = resp.json()
        return [
            ProviderModel(
                provider_id=0,
                name=m["id"],
                size=None,
                details=m.get("meta"),
            )
            for m in data.get("data", [])
        ]

    async def get_props(self) -> dict | None:
        try:
            resp = await self._http.get("/props")
            resp.raise_for_status()
            return resp.json()
        except httpx.HTTPError:
            return None

    async def chat_completions_stream(self, body: dict) -> AsyncIterator[bytes]:
        body["stream"] = True
        async with self._http.stream("POST", "/v1/chat/completions", json=body) as resp:
            resp.raise_for_status()
            async for chunk in resp.aiter_bytes():
                yield chunk

    async def chat_completions(self, body: dict) -> dict:
        body["stream"] = False
        resp = await self._http.post("/v1/chat/completions", json=body)
        resp.raise_for_status()
        return resp.json()

    async def embeddings(self, body: dict) -> dict:
        resp = await self._http.post("/v1/embeddings", json=body)
        resp.raise_for_status()
        return resp.json()

    async def completions_stream(self, body: dict) -> AsyncIterator[bytes]:
        body["stream"] = True
        async with self._http.stream("POST", "/v1/completions", json=body) as resp:
            resp.raise_for_status()
            async for chunk in resp.aiter_bytes():
                yield chunk

    async def completions(self, body: dict) -> dict:
        body["stream"] = False
        resp = await self._http.post("/v1/completions", json=body)
        resp.raise_for_status()
        return resp.json()

    # --- v1/responses ---

    async def responses_stream(self, body: dict) -> AsyncIterator[bytes]:
        body["stream"] = True
        async with self._http.stream("POST", "/v1/responses", json=body) as resp:
            resp.raise_for_status()
            async for chunk in resp.aiter_bytes():
                yield chunk

    async def responses(self, body: dict) -> dict:
        body["stream"] = False
        resp = await self._http.post("/v1/responses", json=body)
        resp.raise_for_status()
        return resp.json()

    # --- v1/images ---

    async def images_generations(self, body: dict) -> dict:
        resp = await self._http.post(
            "/v1/images/generations", json=body, timeout=httpx.Timeout(10.0, read=600.0)
        )
        resp.raise_for_status()
        return resp.json()

    async def images_edits(self, data: bytes, content_type: str) -> httpx.Response:
        """Proxy a multipart images/edits request, returning the raw response."""
        resp = await self._http.post(
            "/v1/images/edits",
            content=data,
            headers={"Content-Type": content_type},
            timeout=httpx.Timeout(10.0, read=600.0),
        )
        resp.raise_for_status()
        return resp

    # --- v1/audio ---

    async def audio_speech(self, body: dict) -> AsyncIterator[bytes]:
        """Stream audio bytes from a TTS request."""
        async with self._http.stream(
            "POST",
            "/v1/audio/speech",
            json=body,
            timeout=httpx.Timeout(10.0, read=600.0),
        ) as resp:
            resp.raise_for_status()
            async for chunk in resp.aiter_bytes():
                yield chunk

    async def audio_transcriptions(
        self, data: bytes, content_type: str
    ) -> httpx.Response:
        """Proxy a multipart audio/transcriptions request, returning the raw response."""
        resp = await self._http.post(
            "/v1/audio/transcriptions",
            content=data,
            headers={"Content-Type": content_type},
            timeout=httpx.Timeout(10.0, read=600.0),
        )
        resp.raise_for_status()
        return resp

    async def audio_voices(self) -> dict:
        resp = await self._http.get("/v1/audio/voices")
        resp.raise_for_status()
        return resp.json()

    async def benchmark_chat(self, model: str, prompt: str) -> dict[str, float]:
        """Run a chat benchmark returning startup_time_ms and tokens_per_second."""
        body = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
        }
        start = time.monotonic()
        resp = await self._http.post(
            "/v1/chat/completions",
            json=body,
            timeout=httpx.Timeout(30.0, read=120.0),
        )
        elapsed_s = time.monotonic() - start
        resp.raise_for_status()
        data = resp.json()

        usage = data.get("usage", {})
        completion_tokens = usage.get("completion_tokens", 0)
        tps = (
            completion_tokens / elapsed_s if completion_tokens and elapsed_s > 0 else 0
        )

        return {"startup_time_ms": 0, "tokens_per_second": tps}

    async def benchmark_embed(self, model: str, prompt: str) -> dict[str, float]:
        """Run an embedding benchmark returning startup_time_ms and tokens_per_second."""
        body = {"model": model, "input": prompt}
        start = time.monotonic()
        resp = await self._http.post(
            "/v1/embeddings",
            json=body,
            timeout=httpx.Timeout(30.0, read=120.0),
        )
        elapsed_s = time.monotonic() - start
        resp.raise_for_status()
        data = resp.json()

        usage = data.get("usage", {})
        prompt_tokens = usage.get("prompt_tokens", 0)
        tps = prompt_tokens / elapsed_s if prompt_tokens and elapsed_s > 0 else 0

        return {"startup_time_ms": 0, "tokens_per_second": tps}
