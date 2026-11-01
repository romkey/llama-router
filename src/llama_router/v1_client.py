"""Protocol-agnostic wrapper for v1/* endpoint access.

Both Ollama and llama.cpp backends serve OpenAI-compatible v1/* endpoints.
This module provides a unified interface for routing requests regardless of
backend type.
"""

from __future__ import annotations

import contextlib
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Any

import httpx

if TYPE_CHECKING:
    from .llamacpp_client import LlamaCppClient
    from .ollama_client import OllamaClient
    from .provider_manager import ProviderManager

# Request fields that steer a router or gateway (OpenRouter's provider, models,
# route and transforms; SDK-only extra_body) or name an endpoint and its
# credentials. The router has already picked the provider, so a client must
# not be able to pass these on and redirect or re-key the upstream call.
ROUTER_ONLY_FIELDS = frozenset(
    {"provider", "models", "route", "transforms", "extra_body", "base_url", "api_key"}
)


def upstream_body(body: dict[str, Any]) -> dict[str, Any]:
    """A copy of a client's request *body* without ``ROUTER_ONLY_FIELDS``."""
    return {k: v for k, v in body.items() if k not in ROUTER_ONLY_FIELDS}


class V1Client:
    """Wraps either an OllamaClient or LlamaCppClient for v1/* calls."""

    def __init__(
        self,
        ollama: OllamaClient | None = None,
        llamacpp: LlamaCppClient | None = None,
    ):
        self._ollama = ollama
        self._llamacpp = llamacpp

    def _require_ollama(self) -> OllamaClient:
        if self._ollama is None:
            raise RuntimeError("this route needs an Ollama upstream")
        return self._ollama

    # --- any JSON route ---

    async def post_json(
        self, path: str, body: dict[str, Any], *, op: str
    ) -> dict[str, Any]:
        """POST *body* to *path* upstream and return the JSON answer."""
        client = self._llamacpp or self._ollama
        if client is None:
            raise RuntimeError("V1Client has no upstream client")
        resp = await client._post(path, op=op, json=body)
        resp.raise_for_status()
        out: dict[str, Any] = resp.json()
        return out

    # --- chat/completions ---

    async def chat_completions_stream(
        self, body: dict[str, Any]
    ) -> AsyncIterator[bytes]:
        if self._llamacpp:
            async for chunk in self._llamacpp.chat_completions_stream(
                upstream_body(body)
            ):
                yield chunk
        elif self._ollama:
            async for chunk in self._ollama.v1_chat_completions_stream(
                upstream_body(body)
            ):
                yield chunk

    async def chat_completions(self, body: dict[str, Any]) -> dict[str, Any]:
        if self._llamacpp:
            return await self._llamacpp.chat_completions(upstream_body(body))
        ollama = self._require_ollama()
        return await ollama.v1_chat_completions(upstream_body(body))

    # --- completions ---

    async def completions_stream(self, body: dict[str, Any]) -> AsyncIterator[bytes]:
        if self._llamacpp:
            async for chunk in self._llamacpp.completions_stream(upstream_body(body)):
                yield chunk
        elif self._ollama:
            async for chunk in self._ollama.v1_completions_stream(upstream_body(body)):
                yield chunk

    async def completions(self, body: dict[str, Any]) -> dict[str, Any]:
        if self._llamacpp:
            return await self._llamacpp.completions(upstream_body(body))
        ollama = self._require_ollama()
        return await ollama.v1_completions(upstream_body(body))

    # --- embeddings ---

    async def embeddings(self, body: dict[str, Any]) -> dict[str, Any]:
        if self._llamacpp:
            return await self._llamacpp.embeddings(upstream_body(body))
        ollama = self._require_ollama()
        return await ollama.v1_embeddings(upstream_body(body))

    # --- responses ---

    async def responses_stream(self, body: dict[str, Any]) -> AsyncIterator[bytes]:
        if self._llamacpp:
            async for chunk in self._llamacpp.responses_stream(upstream_body(body)):
                yield chunk
        elif self._ollama:
            async for chunk in self._ollama.v1_responses_stream(upstream_body(body)):
                yield chunk

    async def responses(self, body: dict[str, Any]) -> dict[str, Any]:
        if self._llamacpp:
            return await self._llamacpp.responses(upstream_body(body))
        ollama = self._require_ollama()
        return await ollama.v1_responses(upstream_body(body))

    # --- images ---

    async def images_generations(self, body: dict[str, Any]) -> dict[str, Any]:
        if self._llamacpp:
            return await self._llamacpp.images_generations(upstream_body(body))
        ollama = self._require_ollama()
        return await ollama.v1_images_generations(upstream_body(body))

    async def images_edits(self, data: bytes, content_type: str) -> httpx.Response:
        if self._llamacpp:
            return await self._llamacpp.images_edits(data, content_type)
        ollama = self._require_ollama()
        return await ollama.v1_images_edits(data, content_type)

    # --- audio ---

    async def audio_speech(self, body: dict[str, Any]) -> AsyncIterator[bytes]:
        if self._llamacpp:
            async for chunk in self._llamacpp.audio_speech(upstream_body(body)):
                yield chunk
        elif self._ollama:
            async for chunk in self._ollama.v1_audio_speech(upstream_body(body)):
                yield chunk

    async def audio_transcriptions(
        self, data: bytes, content_type: str
    ) -> httpx.Response:
        if self._llamacpp:
            return await self._llamacpp.audio_transcriptions(data, content_type)
        ollama = self._require_ollama()
        return await ollama.v1_audio_transcriptions(data, content_type)

    async def audio_voices(self) -> dict[str, Any]:
        if self._llamacpp:
            return await self._llamacpp.audio_voices()
        ollama = self._require_ollama()
        return await ollama.v1_audio_voices()


def get_v1_client(pm: ProviderManager, provider_id: int) -> V1Client:
    """Build a V1Client for the given provider, preferring llama.cpp if available."""
    ollama = None
    llamacpp = None
    with contextlib.suppress(KeyError, Exception):
        llamacpp = pm.get_llamacpp_client(provider_id)
    with contextlib.suppress(KeyError, Exception):
        ollama = pm.get_ollama_client(provider_id)
    return V1Client(ollama=ollama, llamacpp=llamacpp)
