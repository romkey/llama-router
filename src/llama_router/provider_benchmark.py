"""Benchmarking a model on a provider."""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

import httpx

from .config import settings
from .httpx_errors import describe_httpx_error
from .llamacpp_client import LlamaCppClient
from .models import (
    BenchmarkResult,
    Provider,
    ProviderProtocol,
    measured_cold_start,
    plausible_tokens_per_second,
)
from .ollama_client import OllamaClient

if TYPE_CHECKING:
    from .client_pool import ClientPool
    from .database import Database

logger = logging.getLogger(__name__)


def _transient_chat_benchmark_failure(exc: BaseException) -> bool:
    """Return True if chat benchmark likely failed transiently — do not try embed."""
    if isinstance(exc, httpx.TimeoutException):
        return True
    if isinstance(
        exc,
        (
            httpx.ConnectError,
            httpx.ReadError,
            httpx.WriteError,
            httpx.RemoteProtocolError,
            httpx.LocalProtocolError,
        ),
    ):
        return True
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code in (408, 425, 429, 500, 502, 503, 504)
    return False


class Benchmarking:
    """ProviderManager's benchmark runs."""

    if TYPE_CHECKING:
        # Set up and provided by ProviderManager.
        _db: Database
        clients: ClientPool

    async def benchmark_provider(
        self,
        provider_id: int,
        model_name: str,
        benchmark_api: str | None = None,
        *,
        source: str = "manual",
    ) -> BenchmarkResult:
        """Run a benchmark. Raises ``RuntimeError`` with a human-readable
        message on failure so the caller can surface it to the user.

        *source* is ``manual`` or ``auto`` (the idle auto-benchmarker). When
        the model was already loaded, the run cannot measure a cold start, so
        its startup time is stored as unknown rather than as near zero.
        """
        from .request_logger import log_request

        provider = await self._db.get_provider(provider_id)
        if not provider:
            raise RuntimeError(f"Provider {provider_id} not found")

        backend_name = await self._db.get_backend_model_name(provider_id, model_name)
        log_endpoint = "auto-benchmark" if source == "auto" else "benchmark"

        start = time.monotonic()
        protocol: str | None = None

        try:
            metrics: dict[str, float] | None = None

            api_choice = (benchmark_api or "auto").strip().lower()
            if api_choice != "auto":
                wanted = ProviderProtocol.parse(api_choice)
                if wanted is not provider.protocol:
                    raise RuntimeError(
                        f"Provider {provider.name} speaks the "
                        f"{provider.protocol.value} API, not {wanted.value}"
                    )
            client: OllamaClient | LlamaCppClient | None
            if provider.is_ollama:
                client = self.clients.ollama.get(provider_id)
            else:
                client = self.clients.llamacpp.get(provider_id)
            if client is None:
                raise RuntimeError(f"No client available for provider {provider.name}")
            protocol = provider.protocol.value
            # Not counted as a request (that would reset idle detection), but
            # the client must stay open if an edit replaces it meanwhile.
            pinned = self.clients.pin(provider_id)
            try:
                metrics = await self._try_benchmark(
                    client, backend_name, model_name, provider_id
                )
            finally:
                await self.clients.unpin(pinned)

            result = BenchmarkResult(
                provider_id=provider_id,
                model_name=model_name,
                protocol=protocol,
                # The backend's own load time says whether this was a cold
                # start; the router's view of what is loaded can be stale.
                # A cloud service has no cold start the router can measure.
                startup_time_ms=(
                    None
                    if provider.is_cloud
                    else measured_cold_start(metrics["startup_time_ms"])
                ),
                tokens_per_second=plausible_tokens_per_second(
                    metrics["tokens_per_second"]
                ),
                source=source,
            )
            await self._db.save_benchmark(result)

            duration = (time.monotonic() - start) * 1000
            await log_request(
                self._db,
                provider=provider,
                protocol=protocol or "unknown",
                endpoint=log_endpoint,
                model=model_name,
                duration_ms=duration,
                source_ip="internal",
            )

            return result
        except RuntimeError:
            duration = (time.monotonic() - start) * 1000
            await log_request(
                self._db,
                provider=provider,
                protocol=protocol or "unknown",
                endpoint=log_endpoint,
                model=model_name,
                duration_ms=duration,
                source_ip="internal",
                status="error",
                error_detail=f"Benchmark failed for {model_name}",
            )
            raise
        except Exception as exc:
            duration = (time.monotonic() - start) * 1000
            detail = self._format_benchmark_error(exc, model_name, provider)
            logger.error("Benchmark failed: %s", detail)
            await log_request(
                self._db,
                provider=provider,
                protocol=protocol or "unknown",
                endpoint=log_endpoint,
                model=model_name,
                duration_ms=duration,
                source_ip="internal",
                status="error",
                error_detail=detail,
            )
            raise RuntimeError(detail) from exc

    async def _try_benchmark(
        self,
        client: OllamaClient | LlamaCppClient,
        backend_name: str,
        display_name: str,
        provider_id: int,
    ) -> dict[str, float]:
        """Try chat benchmark first; fall back to embed only for non-transient errors.

        LLMs that do not implement embeddings often return HTTP 501 on embed. If chat
        failed for a timeout or 5xx while the model was loading, falling back to embed
        surfaces that misleading 501 instead of the real chat error — so we skip embed
        in those cases. If embed returns 501 after a prior chat error, we report both.
        """
        chat_err: BaseException | None = None
        try:
            return await client.benchmark_chat(backend_name, settings.benchmark_prompt)
        except Exception as exc:
            chat_err = exc
            if _transient_chat_benchmark_failure(exc):
                raise exc
            detail = (
                describe_httpx_error(exc)
                if isinstance(exc, httpx.HTTPError)
                else str(exc)
            )
            logger.info(
                "Chat benchmark failed for %s on provider %d (%s), trying embed",
                display_name,
                provider_id,
                detail,
            )

        try:
            return await client.benchmark_embed(backend_name, settings.benchmark_prompt)
        except Exception as embed_err:
            if (
                chat_err is not None
                and isinstance(embed_err, httpx.HTTPStatusError)
                and embed_err.response.status_code == 501
            ):
                ce = (
                    describe_httpx_error(chat_err)
                    if isinstance(chat_err, httpx.HTTPError)
                    else str(chat_err)
                )
                raise RuntimeError(
                    f"Chat benchmark failed ({ce}); this model does not support "
                    f"embeddings, so there is no fallback. If chat failed while the "
                    f"model was loading or the GPU was busy, retry the benchmark."
                ) from embed_err
            raise embed_err from chat_err

    async def _format_provider_urls(self, provider_id: int) -> str:
        addrs = await self._db.get_addresses(provider_id)
        if not addrs:
            return "(no addresses)"
        return ", ".join(a.url for a in addrs)

    @staticmethod
    def _format_benchmark_error(
        exc: Exception, model_name: str, provider: Provider
    ) -> str:
        if isinstance(exc, httpx.HTTPStatusError):
            status = exc.response.status_code
            try:
                body = exc.response.json()
                msg = body.get("error", exc.response.text[:200])
            except Exception:
                msg = exc.response.text[:200] if exc.response.text else ""
            return (
                f"Backend {provider.name} returned HTTP {status} "
                f"for {model_name}: {msg}"
            )
        if isinstance(exc, httpx.RequestError):
            return (
                f"Could not reach backend {provider.name} for {model_name}: "
                f"{describe_httpx_error(exc)}"
            )
        return f"Benchmark failed for {model_name} on {provider.name}: {exc}"
