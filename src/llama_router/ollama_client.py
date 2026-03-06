"""Async HTTP client for communicating with Ollama backends."""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator
from typing import Any
from urllib.parse import urlparse

import httpx

from .models import ProviderModel

logger = logging.getLogger(__name__)

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

    async def pull_model(
        self,
        model: str,
        cache_registry_url: str | None = None,
        progress_callback: Any | None = None,
    ) -> dict:
        """Pull a model, streaming NDJSON progress. Returns the final status.

        *progress_callback*, if provided, is called with a dict containing
        ``status``, and optionally ``percent``, ``completed``, ``total`` for
        each meaningful progress update.

        Raises ``RuntimeError`` if Ollama reports an error in the stream.
        """
        pull_model = model
        insecure = False
        if cache_registry_url:
            parsed = urlparse(cache_registry_url)
            host_port = parsed.netloc or parsed.path.rstrip("/")
            if "/" not in model or model.startswith("library/"):
                pull_model = f"{host_port}/library/{model}"
            else:
                pull_model = f"{host_port}/{model}"
            insecure = True

        logger.info(
            "Starting pull: model=%s pull_model=%s insecure=%s base_url=%s",
            model,
            pull_model,
            insecure,
            self._base_url,
        )

        last_status: dict = {}
        last_logged_pct: int = -1
        last_cb_pct: int = -1

        async with self._http.stream(
            "POST",
            "/api/pull",
            json={"model": pull_model, "stream": True, "insecure": insecure},
            timeout=httpx.Timeout(30.0, read=1800.0),
        ) as resp:
            resp.raise_for_status()
            buffer = b""
            async for chunk in resp.aiter_bytes():
                buffer += chunk
                while b"\n" in buffer:
                    line, buffer = buffer.split(b"\n", 1)
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        msg = json.loads(line)
                    except json.JSONDecodeError:
                        logger.warning("Pull %s: non-JSON line: %s", model, line[:200])
                        continue

                    if "error" in msg:
                        error_msg = msg["error"]
                        logger.error("Pull %s FAILED: %s", model, error_msg)
                        raise RuntimeError(f"Ollama pull error: {error_msg}")

                    last_status = msg
                    status_text = msg.get("status", "")

                    total = msg.get("total", 0)
                    completed = msg.get("completed", 0)
                    if total > 0:
                        pct = int(completed * 100 / total)
                        if pct >= last_logged_pct + 10:
                            logger.info(
                                "Pull %s: %s — %d%% (%s / %s)",
                                model,
                                status_text,
                                pct,
                                _human_bytes(completed),
                                _human_bytes(total),
                            )
                            last_logged_pct = pct
                        if progress_callback and pct >= last_cb_pct + 2:
                            progress_callback(
                                {
                                    "status": status_text,
                                    "percent": pct,
                                    "completed": completed,
                                    "total": total,
                                }
                            )
                            last_cb_pct = pct
                    elif status_text:
                        logger.info("Pull %s: %s", model, status_text)
                        if progress_callback:
                            progress_callback({"status": status_text})

        final_status = last_status.get("status", "unknown")
        logger.info("Pull %s completed: %s", model, final_status)
        return last_status

    # --- OpenAI-compatible v1/* endpoints (served by Ollama natively) ---

    async def v1_chat_completions_stream(self, body: dict) -> AsyncIterator[bytes]:
        body["stream"] = True
        async with self._http.stream("POST", "/v1/chat/completions", json=body) as resp:
            resp.raise_for_status()
            async for chunk in resp.aiter_bytes():
                yield chunk

    async def v1_chat_completions(self, body: dict) -> dict:
        body["stream"] = False
        resp = await self._http.post("/v1/chat/completions", json=body)
        resp.raise_for_status()
        return resp.json()

    async def v1_completions_stream(self, body: dict) -> AsyncIterator[bytes]:
        body["stream"] = True
        async with self._http.stream("POST", "/v1/completions", json=body) as resp:
            resp.raise_for_status()
            async for chunk in resp.aiter_bytes():
                yield chunk

    async def v1_completions(self, body: dict) -> dict:
        body["stream"] = False
        resp = await self._http.post("/v1/completions", json=body)
        resp.raise_for_status()
        return resp.json()

    async def v1_embeddings(self, body: dict) -> dict:
        resp = await self._http.post("/v1/embeddings", json=body)
        resp.raise_for_status()
        return resp.json()

    async def v1_responses_stream(self, body: dict) -> AsyncIterator[bytes]:
        body["stream"] = True
        async with self._http.stream("POST", "/v1/responses", json=body) as resp:
            resp.raise_for_status()
            async for chunk in resp.aiter_bytes():
                yield chunk

    async def v1_responses(self, body: dict) -> dict:
        body["stream"] = False
        resp = await self._http.post("/v1/responses", json=body)
        resp.raise_for_status()
        return resp.json()

    async def v1_images_generations(self, body: dict) -> dict:
        resp = await self._http.post(
            "/v1/images/generations", json=body, timeout=httpx.Timeout(10.0, read=600.0)
        )
        resp.raise_for_status()
        return resp.json()

    async def v1_images_edits(self, data: bytes, content_type: str) -> httpx.Response:
        resp = await self._http.post(
            "/v1/images/edits",
            content=data,
            headers={"Content-Type": content_type},
            timeout=httpx.Timeout(10.0, read=600.0),
        )
        resp.raise_for_status()
        return resp

    async def v1_audio_speech(self, body: dict) -> AsyncIterator[bytes]:
        async with self._http.stream(
            "POST",
            "/v1/audio/speech",
            json=body,
            timeout=httpx.Timeout(10.0, read=600.0),
        ) as resp:
            resp.raise_for_status()
            async for chunk in resp.aiter_bytes():
                yield chunk

    async def v1_audio_transcriptions(
        self, data: bytes, content_type: str
    ) -> httpx.Response:
        resp = await self._http.post(
            "/v1/audio/transcriptions",
            content=data,
            headers={"Content-Type": content_type},
            timeout=httpx.Timeout(10.0, read=600.0),
        )
        resp.raise_for_status()
        return resp

    async def v1_audio_voices(self) -> dict:
        resp = await self._http.get("/v1/audio/voices")
        resp.raise_for_status()
        return resp.json()

    async def delete_model(self, model: str) -> None:
        resp = await self._http.request("DELETE", "/api/delete", json={"model": model})
        resp.raise_for_status()

    async def benchmark_chat(self, model: str, prompt: str) -> dict[str, float]:
        """Run a chat benchmark returning startup_time_ms and tokens_per_second."""
        body = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
        }
        resp = await self._http.post(
            "/api/chat",
            json=body,
            timeout=httpx.Timeout(30.0, read=120.0),
        )
        resp.raise_for_status()
        data = resp.json()

        load_duration_ns = data.get("load_duration", 0)
        eval_count = data.get("eval_count", 0)
        eval_duration_ns = data.get("eval_duration", 1)

        startup_ms = load_duration_ns / 1_000_000
        tps = (eval_count / eval_duration_ns * 1_000_000_000) if eval_count else 0

        return {"startup_time_ms": startup_ms, "tokens_per_second": tps}

    async def benchmark_embed(self, model: str, prompt: str) -> dict[str, float]:
        """Run an embedding benchmark returning startup_time_ms and tokens_per_second.

        tokens_per_second here represents input processing speed
        (prompt_eval_count / eval_duration).
        """
        body = {"model": model, "input": prompt}
        resp = await self._http.post(
            "/api/embed",
            json=body,
            timeout=httpx.Timeout(30.0, read=120.0),
        )
        resp.raise_for_status()
        data = resp.json()

        load_duration_ns = data.get("load_duration", 0)
        prompt_eval_count = data.get("prompt_eval_count", 0)
        total_duration_ns = data.get("total_duration", 0)

        startup_ms = load_duration_ns / 1_000_000
        eval_ns = total_duration_ns - load_duration_ns
        tps = (
            (prompt_eval_count / eval_ns * 1_000_000_000)
            if prompt_eval_count and eval_ns > 0
            else 0
        )

        return {"startup_time_ms": startup_ms, "tokens_per_second": tps}


def _human_bytes(n: int) -> str:
    if n >= 1 << 30:
        return f"{n / (1 << 30):.1f} GB"
    if n >= 1 << 20:
        return f"{n / (1 << 20):.1f} MB"
    return f"{n / (1 << 10):.0f} KB"
