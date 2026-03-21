"""Async HTTP client for communicating with llama.cpp server backends."""

from __future__ import annotations

import logging
import time
from collections.abc import AsyncIterator
from typing import Any

import httpx

from .httpx_errors import describe_httpx_error
from .models import ProviderModel

logger = logging.getLogger(__name__)

_TIMEOUT = httpx.Timeout(10.0, read=120.0)


class LlamaCppClient:
    def __init__(self, base_url: str):
        self._base_url = base_url.rstrip("/")
        self._http = httpx.AsyncClient(base_url=self._base_url, timeout=_TIMEOUT)

    def _upstream_url(self, path: str) -> str:
        p = path if path.startswith("/") else f"/{path}"
        return f"{self._base_url}{p}"

    def _log_httpx(
        self, exc: httpx.HTTPError, *, op: str, method: str, path: str
    ) -> None:
        logger.error(
            "%s — %s %s — %s",
            op,
            method,
            self._upstream_url(path),
            describe_httpx_error(exc),
        )

    async def _get(self, path: str, *, op: str, **kwargs: Any) -> httpx.Response:
        try:
            return await self._http.get(path, **kwargs)
        except httpx.HTTPError as exc:
            self._log_httpx(exc, op=op, method="GET", path=path)
            raise

    async def _post(self, path: str, *, op: str, **kwargs: Any) -> httpx.Response:
        try:
            return await self._http.post(path, **kwargs)
        except httpx.HTTPError as exc:
            self._log_httpx(exc, op=op, method="POST", path=path)
            raise

    async def close(self) -> None:
        await self._http.aclose()

    async def is_reachable(self) -> bool:
        try:
            resp = await self._http.get("/health")
            return resp.status_code == 200
        except httpx.HTTPError:
            return False

    async def get_models(self) -> list[ProviderModel]:
        resp = await self._get("/v1/models", op="List llama.cpp models (/v1/models)")
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
        except httpx.HTTPError as exc:
            logger.debug(
                "Optional llama.cpp /props probe failed at %s: %s",
                self._upstream_url("/props"),
                describe_httpx_error(exc),
            )
            return None

    async def chat_completions_stream(self, body: dict) -> AsyncIterator[bytes]:
        body["stream"] = True
        try:
            async with self._http.stream(
                "POST", "/v1/chat/completions", json=body
            ) as resp:
                resp.raise_for_status()
                async for chunk in resp.aiter_bytes():
                    yield chunk
        except httpx.HTTPError as exc:
            self._log_httpx(
                exc,
                op="llama.cpp streaming chat (/v1/chat/completions)",
                method="POST",
                path="/v1/chat/completions",
            )
            raise

    async def chat_completions(self, body: dict) -> dict:
        body["stream"] = False
        resp = await self._post(
            "/v1/chat/completions",
            op="llama.cpp chat completions (/v1/chat/completions)",
            json=body,
        )
        resp.raise_for_status()
        return resp.json()

    async def embeddings(self, body: dict) -> dict:
        resp = await self._post(
            "/v1/embeddings", op="llama.cpp embeddings (/v1/embeddings)", json=body
        )
        resp.raise_for_status()
        return resp.json()

    async def completions_stream(self, body: dict) -> AsyncIterator[bytes]:
        body["stream"] = True
        try:
            async with self._http.stream("POST", "/v1/completions", json=body) as resp:
                resp.raise_for_status()
                async for chunk in resp.aiter_bytes():
                    yield chunk
        except httpx.HTTPError as exc:
            self._log_httpx(
                exc,
                op="llama.cpp streaming completions (/v1/completions)",
                method="POST",
                path="/v1/completions",
            )
            raise

    async def completions(self, body: dict) -> dict:
        body["stream"] = False
        resp = await self._post(
            "/v1/completions", op="llama.cpp completions (/v1/completions)", json=body
        )
        resp.raise_for_status()
        return resp.json()

    # --- v1/responses ---

    async def responses_stream(self, body: dict) -> AsyncIterator[bytes]:
        body["stream"] = True
        try:
            async with self._http.stream("POST", "/v1/responses", json=body) as resp:
                resp.raise_for_status()
                async for chunk in resp.aiter_bytes():
                    yield chunk
        except httpx.HTTPError as exc:
            self._log_httpx(
                exc,
                op="llama.cpp streaming responses (/v1/responses)",
                method="POST",
                path="/v1/responses",
            )
            raise

    async def responses(self, body: dict) -> dict:
        body["stream"] = False
        resp = await self._post(
            "/v1/responses", op="llama.cpp responses (/v1/responses)", json=body
        )
        resp.raise_for_status()
        return resp.json()

    # --- v1/images ---

    async def images_generations(self, body: dict) -> dict:
        resp = await self._post(
            "/v1/images/generations",
            op="llama.cpp image generation (/v1/images/generations)",
            json=body,
            timeout=httpx.Timeout(10.0, read=600.0),
        )
        resp.raise_for_status()
        return resp.json()

    async def images_edits(self, data: bytes, content_type: str) -> httpx.Response:
        """Proxy a multipart images/edits request, returning the raw response."""
        resp = await self._post(
            "/v1/images/edits",
            op="llama.cpp image edits (/v1/images/edits)",
            content=data,
            headers={"Content-Type": content_type},
            timeout=httpx.Timeout(10.0, read=600.0),
        )
        resp.raise_for_status()
        return resp

    # --- v1/audio ---

    async def audio_speech(self, body: dict) -> AsyncIterator[bytes]:
        """Stream audio bytes from a TTS request."""
        try:
            async with self._http.stream(
                "POST",
                "/v1/audio/speech",
                json=body,
                timeout=httpx.Timeout(10.0, read=600.0),
            ) as resp:
                resp.raise_for_status()
                async for chunk in resp.aiter_bytes():
                    yield chunk
        except httpx.HTTPError as exc:
            self._log_httpx(
                exc,
                op="llama.cpp TTS (/v1/audio/speech)",
                method="POST",
                path="/v1/audio/speech",
            )
            raise

    async def audio_transcriptions(
        self, data: bytes, content_type: str
    ) -> httpx.Response:
        """Proxy a multipart audio/transcriptions request, returning the raw response."""
        resp = await self._post(
            "/v1/audio/transcriptions",
            op="llama.cpp audio transcription (/v1/audio/transcriptions)",
            content=data,
            headers={"Content-Type": content_type},
            timeout=httpx.Timeout(10.0, read=600.0),
        )
        resp.raise_for_status()
        return resp

    async def audio_voices(self) -> dict:
        resp = await self._get(
            "/v1/audio/voices", op="llama.cpp audio voices (/v1/audio/voices)"
        )
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
        resp = await self._post(
            "/v1/chat/completions",
            op=f"llama.cpp chat benchmark for {model!r}",
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
        resp = await self._post(
            "/v1/embeddings",
            op=f"llama.cpp embed benchmark for {model!r}",
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
