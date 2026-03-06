"""Protocol-agnostic wrapper for v1/* endpoint access.

Both Ollama and llama.cpp backends serve OpenAI-compatible v1/* endpoints.
This module provides a unified interface for routing requests regardless of
backend type.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import TYPE_CHECKING

import httpx

if TYPE_CHECKING:
    from .llamacpp_client import LlamaCppClient
    from .ollama_client import OllamaClient
    from .provider_manager import ProviderManager


class V1Client:
    """Wraps either an OllamaClient or LlamaCppClient for v1/* calls."""

    def __init__(
        self,
        ollama: OllamaClient | None = None,
        llamacpp: LlamaCppClient | None = None,
    ):
        self._ollama = ollama
        self._llamacpp = llamacpp

    # --- chat/completions ---

    async def chat_completions_stream(self, body: dict) -> AsyncIterator[bytes]:
        if self._llamacpp:
            async for chunk in self._llamacpp.chat_completions_stream(body):
                yield chunk
        elif self._ollama:
            async for chunk in self._ollama.v1_chat_completions_stream(body):
                yield chunk

    async def chat_completions(self, body: dict) -> dict:
        if self._llamacpp:
            return await self._llamacpp.chat_completions(body)
        assert self._ollama
        return await self._ollama.v1_chat_completions(body)

    # --- completions ---

    async def completions_stream(self, body: dict) -> AsyncIterator[bytes]:
        if self._llamacpp:
            async for chunk in self._llamacpp.completions_stream(body):
                yield chunk
        elif self._ollama:
            async for chunk in self._ollama.v1_completions_stream(body):
                yield chunk

    async def completions(self, body: dict) -> dict:
        if self._llamacpp:
            return await self._llamacpp.completions(body)
        assert self._ollama
        return await self._ollama.v1_completions(body)

    # --- embeddings ---

    async def embeddings(self, body: dict) -> dict:
        if self._llamacpp:
            return await self._llamacpp.embeddings(body)
        assert self._ollama
        return await self._ollama.v1_embeddings(body)

    # --- responses ---

    async def responses_stream(self, body: dict) -> AsyncIterator[bytes]:
        if self._llamacpp:
            async for chunk in self._llamacpp.responses_stream(body):
                yield chunk
        elif self._ollama:
            async for chunk in self._ollama.v1_responses_stream(body):
                yield chunk

    async def responses(self, body: dict) -> dict:
        if self._llamacpp:
            return await self._llamacpp.responses(body)
        assert self._ollama
        return await self._ollama.v1_responses(body)

    # --- images ---

    async def images_generations(self, body: dict) -> dict:
        if self._llamacpp:
            return await self._llamacpp.images_generations(body)
        assert self._ollama
        return await self._ollama.v1_images_generations(body)

    async def images_edits(self, data: bytes, content_type: str) -> httpx.Response:
        if self._llamacpp:
            return await self._llamacpp.images_edits(data, content_type)
        assert self._ollama
        return await self._ollama.v1_images_edits(data, content_type)

    # --- audio ---

    async def audio_speech(self, body: dict) -> AsyncIterator[bytes]:
        if self._llamacpp:
            async for chunk in self._llamacpp.audio_speech(body):
                yield chunk
        elif self._ollama:
            async for chunk in self._ollama.v1_audio_speech(body):
                yield chunk

    async def audio_transcriptions(
        self, data: bytes, content_type: str
    ) -> httpx.Response:
        if self._llamacpp:
            return await self._llamacpp.audio_transcriptions(data, content_type)
        assert self._ollama
        return await self._ollama.v1_audio_transcriptions(data, content_type)

    async def audio_voices(self) -> dict:
        if self._llamacpp:
            return await self._llamacpp.audio_voices()
        assert self._ollama
        return await self._ollama.v1_audio_voices()


def get_v1_client(pm: ProviderManager, provider_id: int) -> V1Client:
    """Build a V1Client for the given provider, preferring llama.cpp if available."""
    ollama = None
    llamacpp = None
    try:
        llamacpp = pm.get_llamacpp_client(provider_id)
    except (KeyError, Exception):
        pass
    try:
        ollama = pm.get_ollama_client(provider_id)
    except (KeyError, Exception):
        pass
    return V1Client(ollama=ollama, llamacpp=llamacpp)
