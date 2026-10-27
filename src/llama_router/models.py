from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import TypedDict

from pydantic import BaseModel, Field


class HotModel(TypedDict, total=False):
    name: str
    size: int
    size_vram: int
    expires_at: str


WORKER_URL_SCHEME = "worker://"


class ProviderStatus(str, Enum):
    IDLE = "idle"
    BUSY = "busy"
    OFFLINE = "offline"
    UNKNOWN = "unknown"


class ProviderType(str, Enum):
    OLLAMA = "ollama"
    LLAMACPP = "llamacpp"
    BOTH = "both"


class Provider(BaseModel):
    id: int | None = None
    name: str
    provider_type: ProviderType = ProviderType.OLLAMA
    status: ProviderStatus = ProviderStatus.UNKNOWN
    machine_type: str | None = None
    gpu_type: str | None = None
    gpu_ram: str | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None
    # Set for providers backed by a worker that dials in over a WebSocket.
    worker_name: str | None = None
    # Upstream API key, encrypted (see provider_secrets). Never rendered.
    api_key_enc: str | None = Field(default=None, repr=False, exclude=True)
    # Header carrying the key; None means "Authorization: Bearer <key>".
    api_key_header: str | None = None
    # Result of the last endpoint probe (see probe.py), if any.
    capabilities: dict | None = None
    probed_at: datetime | None = None

    @property
    def is_worker(self) -> bool:
        """Provider backed by a worker that dials in over a WebSocket."""
        return self.worker_name is not None

    @property
    def worker_address(self) -> str | None:
        """The address row a worker provider shows, ``worker://<name>``."""
        if self.worker_name is None:
            return None
        return f"{WORKER_URL_SCHEME}{self.worker_name}"

    @property
    def has_api_key(self) -> bool:
        return bool(self.api_key_enc)

    @property
    def supports_ollama(self) -> bool:
        return self.provider_type in (ProviderType.OLLAMA, ProviderType.BOTH)

    @property
    def supports_llamacpp(self) -> bool:
        return self.provider_type in (ProviderType.LLAMACPP, ProviderType.BOTH)

    @property
    def gpu_ram_bytes(self) -> int | None:
        """Parse gpu_ram string (e.g. '24 GB', '192GB', '8192 MB') into bytes."""
        if not self.gpu_ram:
            return None
        import re

        m = re.match(r"([\d.]+)\s*(TB|GB|MB)", self.gpu_ram.strip(), re.IGNORECASE)
        if not m:
            return None
        val = float(m.group(1))
        unit = m.group(2).upper()
        multipliers = {"TB": 1 << 40, "GB": 1 << 30, "MB": 1 << 20}
        return int(val * multipliers[unit])


class ProviderModel(BaseModel):
    id: int | None = None
    provider_id: int
    name: str
    raw_name: str | None = None
    size: int | None = None
    digest: str | None = None
    modified_at: str | None = None
    details: dict | None = None

    @property
    def backend_name(self) -> str:
        """The name the backend actually knows this model by."""
        return self.raw_name or self.name


class BenchmarkResult(BaseModel):
    id: int | None = None
    provider_id: int
    model_name: str
    protocol: str | None = None
    # None when the model was already loaded, so the run could not measure a
    # cold start.
    startup_time_ms: float | None = None
    tokens_per_second: float | None = None
    # "manual" (dashboard or API) or "auto" (the idle auto-benchmarker).
    source: str | None = None
    created_at: datetime | None = None


def model_in_api(model: "ProviderModel", api: str, provider_type: ProviderType) -> bool:
    """Whether *model* is served on the provider's *api* ("ollama"/"llamacpp")."""
    details = model.details or {}
    if api == "ollama":
        if "_in_ollama" in details:
            return bool(details.get("_in_ollama"))
        return provider_type in (ProviderType.OLLAMA, ProviderType.BOTH)
    if api == "llamacpp":
        if "_in_llamacpp" in details:
            return bool(details.get("_in_llamacpp"))
        return provider_type in (ProviderType.LLAMACPP, ProviderType.BOTH)
    return False


class ProviderAddress(BaseModel):
    id: int | None = None
    provider_id: int
    url: str
    llamacpp_url: str | None = None
    is_preferred: bool = False
    is_live: bool = False
    created_at: datetime | None = None


class RequestLog(BaseModel):
    id: int | None = None
    provider_id: int | None = None
    provider_name: str | None = None
    protocol: str  # "ollama" or "llamacpp"
    endpoint: str  # e.g. "/api/chat", "/v1/chat/completions"
    source_ip: str | None = None
    model: str | None = None
    request_size: int = 0
    response_size: int = 0
    request_meta: str | None = None
    duration_ms: float = 0.0
    # Dispatch to the backend's first byte (streamed responses only).
    ttfb_ms: float | None = None
    # Arrival at the router to dispatch: parsing, auth and routing.
    router_ms: float | None = None
    status: str = "ok"  # "ok" or "error"
    streamed: bool = False
    error_detail: str | None = None
    created_at: datetime | None = None


class ProviderInfo(BaseModel):
    provider: Provider
    models: list[ProviderModel] = []
    benchmarks: list[BenchmarkResult] = []
    addresses: list[ProviderAddress] = []
    active_requests: int = 0
    hot_models: list[HotModel] = []
    worker: dict | None = None
    # Why the provider is offline although an address answers, if known.
    status_detail: str | None = None
