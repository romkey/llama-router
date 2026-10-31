"""Async HTTP client for OpenAI-style backends: llama.cpp, vLLM, LiteLLM, cloud APIs."""

from __future__ import annotations

import logging
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Any
from urllib.parse import urlsplit

import httpx

from .httpx_errors import describe_httpx_error, raise_for_stream_status
from .models import ProviderModel
from .upstream_url import split_api_base

logger = logging.getLogger(__name__)

_TIMEOUT = httpx.Timeout(10.0, read=120.0)


# The rest of a llama.cpp request's wall time (beyond the server's prompt and
# generation timings) includes network and queueing, not just a model load.
# Loading a model from disk takes at least about a second, so only a remainder
# that long is taken as a cold start.
LOAD_MIN_MS = 1000.0


def _chat_metrics(data: dict, elapsed_s: float) -> dict[str, float | None]:
    """Generation speed and startup from a llama.cpp chat response.

    llama-server reports its own ``timings`` (prompt and generation time and
    speed). Generation speed comes from there; startup is the rest of the
    wall time, which is the model load when llama-server had to load it
    (router mode) and close to zero when it was already loaded. Without
    timings, fall back to completion tokens over wall time.
    """
    timings = data.get("timings") or {}
    pps = timings.get("predicted_per_second")
    prompt_ms = timings.get("prompt_ms")
    predicted_ms = timings.get("predicted_ms")
    elapsed_ms = elapsed_s * 1000
    if isinstance(pps, (int, float)) and pps > 0:
        startup = None
        if isinstance(prompt_ms, (int, float)) and isinstance(
            predicted_ms, (int, float)
        ):
            rest = elapsed_ms - prompt_ms - predicted_ms
            startup = rest if rest >= LOAD_MIN_MS else None
        return {"startup_time_ms": startup, "tokens_per_second": float(pps)}
    usage = data.get("usage") or {}
    completion_tokens = usage.get("completion_tokens", 0)
    tps = completion_tokens / elapsed_s if completion_tokens and elapsed_s > 0 else 0
    return {"startup_time_ms": None, "tokens_per_second": tps}


class OpenAIClient:
    """Client for an OpenAI-style upstream: llama.cpp, vLLM, LiteLLM, cloud APIs.

    Routes are written as ``/v1/...`` and land on the address's API base (see
    ``upstream_url``); server-native routes such as ``/health`` and ``/props``
    go to the server root.
    """

    def __init__(
        self,
        base_url: str,
        transport: httpx.AsyncBaseTransport | None = None,
        *,
        timeout: httpx.Timeout | None = None,
        headers: dict[str, str] | None = None,
        on_response: Callable[[httpx.Response], Awaitable[None]] | None = None,
        on_request: Callable[[httpx.Request], Awaitable[None]] | None = None,
    ):
        self._base_url = base_url.rstrip("/")
        root, api_path = split_api_base(self._base_url)
        self._root = root
        self._root_path = urlsplit(root).path
        self._api_path = api_path
        self._http = httpx.AsyncClient(
            base_url=root,
            timeout=timeout or _TIMEOUT,
            transport=transport,
            headers=headers,
            event_hooks={
                "request": [self._to_api_base, *([on_request] if on_request else [])],
                "response": [on_response] if on_response else [],
            },
        )

    def _map_path(self, path: str) -> str:
        """Origin-relative path that a root-relative *path* reaches."""
        if path == "/v1" or path.startswith("/v1/"):
            return self._api_path + path[3:]
        return self._root_path + path

    async def _to_api_base(self, request: httpx.Request) -> None:
        prefix = self._root_path + "/v1"
        p = request.url.path
        if self._api_path != prefix and (p == prefix or p.startswith(prefix + "/")):
            request.url = request.url.copy_with(path=self._api_path + p[len(prefix) :])

    def _upstream_url(self, path: str) -> str:
        p = path if path.startswith("/") else f"/{path}"
        origin = self._root[: len(self._root) - len(self._root_path)]
        return f"{origin}{self._map_path(p)}"

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

    async def is_reachable(self, timeout: httpx.Timeout | None = None) -> bool:
        """llama.cpp-style ``/health``, else the model list most servers have."""
        kw = {"timeout": timeout} if timeout else {}
        for path in ("/health", "/v1/models"):
            try:
                resp = await self._http.get(path, **kw)
            except httpx.HTTPError:
                return False
            if resp.status_code == 200:
                return True
        return False

    async def get_models(self) -> list[ProviderModel]:
        resp = await self._get("/v1/models", op="List llama.cpp models (/v1/models)")
        resp.raise_for_status()
        data = resp.json()
        # Most servers wrap the list in {"data": [...]}; some return it bare.
        # TypeSafe's is {"models": [{"name": ...}]}.
        if isinstance(data, list):
            items = data
        else:
            items = data.get("data") or [
                {"id": m.get("name"), **m}
                for m in data.get("models") or []
                if isinstance(m, dict)
            ]
        out = []
        for m in items:
            if not isinstance(m, dict) or not m.get("id"):
                continue
            details = dict(m.get("meta") or {})
            # Kept for routing and cost: OpenRouter lists prices and context.
            for key in ("pricing", "context_length", "owned_by"):
                if m.get(key) is not None:
                    details[key] = m[key]
            out.append(
                ProviderModel(
                    provider_id=0, name=str(m["id"]), size=None, details=details or None
                )
            )
        return out

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
                await raise_for_stream_status(resp)
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
                await raise_for_stream_status(resp)
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
                await raise_for_stream_status(resp)
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
                await raise_for_stream_status(resp)
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
        """Proxy a multipart audio/transcriptions request; return the raw response."""
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

        return _chat_metrics(data, elapsed_s)

    async def benchmark_embed(self, model: str, prompt: str) -> dict[str, float]:
        """Run an embedding benchmark; return startup_time_ms and tokens_per_second."""
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

        # llama-server does not report a load time for embeddings.
        return {"startup_time_ms": None, "tokens_per_second": tps}


# The client served llama.cpp first; older code and tests use this name.
LlamaCppClient = OpenAIClient
