"""Protocol-agnostic wrapper for v1/* endpoint access.

Both Ollama and llama.cpp backends serve OpenAI-compatible v1/* endpoints.
This module provides a unified interface for routing requests regardless of
backend type.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import httpx
from starlette.datastructures import FormData, UploadFile

from .httpx_errors import raise_for_stream_status
from .json_io import json_kwargs

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


@dataclass
class MultipartForm:
    """A multipart request rebuilt for the upstream: text fields and files."""

    data: dict[str, str | list[str]] = field(default_factory=dict)
    files: list[tuple[str, tuple[str | None, bytes, str | None]]] = field(
        default_factory=list
    )


async def upstream_form(form: FormData, backend_model: str) -> MultipartForm:
    """Rebuild a client's multipart *form* for the upstream.

    ``model`` becomes the provider's own name for the routed model (a group or
    alias means nothing upstream) and ``ROUTER_ONLY_FIELDS`` are dropped, as
    for JSON bodies.
    """
    out = MultipartForm()
    for key, value in form.multi_items():
        if key in ROUTER_ONLY_FIELDS:
            continue
        if isinstance(value, UploadFile):
            out.files.append(
                (key, (value.filename, await value.read(), value.content_type))
            )
            continue
        text = backend_model if key == "model" else value
        prev = out.data.get(key)
        if prev is None:
            out.data[key] = text
        elif isinstance(prev, list):
            prev.append(text)
        else:
            out.data[key] = [prev, text]
    return out


class NoReachableAddressError(RuntimeError):
    """The routed provider has no client: it has no address to send to.

    The API apps answer it with 503.
    """

    def __init__(self) -> None:
        super().__init__("provider has no reachable address")


class V1Client:
    """Wraps either an OllamaClient or LlamaCppClient for v1/* calls."""

    def __init__(
        self,
        ollama: OllamaClient | None = None,
        llamacpp: LlamaCppClient | None = None,
    ):
        if ollama is None and llamacpp is None:
            raise NoReachableAddressError()
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
        client: LlamaCppClient | OllamaClient = (
            self._llamacpp if self._llamacpp is not None else self._require_ollama()
        )
        resp = await client._post(path, op=op, **await json_kwargs(upstream_body(body)))
        resp.raise_for_status()
        out: dict[str, Any] = resp.json()
        return out

    # --- Anthropic Messages (passed through; never translated) ---

    def _any(self) -> LlamaCppClient | OllamaClient:
        return self._llamacpp if self._llamacpp is not None else self._require_ollama()

    async def messages(
        self, body: dict[str, Any], headers: dict[str, str], path: str = "/v1/messages"
    ) -> dict[str, Any]:
        """POST an Anthropic Messages body (or count_tokens, with *path*) as
        is, with the client's ``anthropic-*`` *headers*; the answer as is."""
        client = self._any()
        resp = await client._post(
            path,
            op=f"Anthropic Messages ({path})",
            **await _with_headers(upstream_body(body), headers),
        )
        resp.raise_for_status()
        out: dict[str, Any] = resp.json()
        return out

    async def messages_stream(
        self, body: dict[str, Any], headers: dict[str, str]
    ) -> AsyncIterator[bytes]:
        """A streamed Messages reply: the upstream's SSE bytes, unchanged."""
        client = self._any()
        async with client._http.stream(
            "POST",
            "/v1/messages",
            **await _with_headers(upstream_body(body), headers),
        ) as resp:
            await raise_for_stream_status(resp)
            async for chunk in resp.aiter_bytes():
                yield chunk

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

    # --- rerank ---

    async def rerank(self, body: dict[str, Any], path: str) -> dict[str, Any]:
        """Rerank on upstream route *path* (``/v1/rerank`` or an alias)."""
        return await self.post_json(path, body, op=f"rerank ({path})")

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

    async def images_edits(self, form: MultipartForm) -> httpx.Response:
        if self._llamacpp:
            return await self._llamacpp.images_edits(form)
        ollama = self._require_ollama()
        return await ollama.v1_images_edits(form)

    # --- audio ---

    async def audio_speech(self, body: dict[str, Any]) -> AsyncIterator[bytes]:
        if self._llamacpp:
            async for chunk in self._llamacpp.audio_speech(upstream_body(body)):
                yield chunk
        elif self._ollama:
            async for chunk in self._ollama.v1_audio_speech(upstream_body(body)):
                yield chunk

    async def audio_transcriptions(self, form: MultipartForm) -> httpx.Response:
        if self._llamacpp:
            return await self._llamacpp.audio_transcriptions(form)
        ollama = self._require_ollama()
        return await ollama.v1_audio_transcriptions(form)

    async def audio_voices(self) -> dict[str, Any]:
        if self._llamacpp:
            return await self._llamacpp.audio_voices()
        ollama = self._require_ollama()
        return await ollama.v1_audio_voices()


async def _with_headers(
    body: dict[str, Any], headers: dict[str, str]
) -> dict[str, Any]:
    """httpx arguments sending *body* as JSON with *headers* as well: a large
    body brings a Content-Type header of its own, merged rather than
    passed twice."""
    kwargs = await json_kwargs(body)
    kwargs["headers"] = {**kwargs.get("headers", {}), **headers}
    return kwargs


def get_v1_client(pm: ProviderManager, provider_id: int) -> V1Client:
    """Build a V1Client for the given provider, preferring llama.cpp if available."""
    return V1Client(
        ollama=pm.find_ollama_client(provider_id),
        llamacpp=pm.find_llamacpp_client(provider_id),
    )
