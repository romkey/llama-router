from __future__ import annotations

from datetime import datetime
from enum import Enum

from pydantic import BaseModel


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
    url: str
    llamacpp_url: str | None = None
    provider_type: ProviderType = ProviderType.OLLAMA
    status: ProviderStatus = ProviderStatus.UNKNOWN
    created_at: datetime | None = None
    updated_at: datetime | None = None

    @property
    def supports_ollama(self) -> bool:
        return self.provider_type in (ProviderType.OLLAMA, ProviderType.BOTH)

    @property
    def supports_llamacpp(self) -> bool:
        return self.provider_type in (ProviderType.LLAMACPP, ProviderType.BOTH)


class ProviderModel(BaseModel):
    id: int | None = None
    provider_id: int
    name: str
    size: int | None = None
    digest: str | None = None
    modified_at: str | None = None
    details: dict | None = None


class BenchmarkResult(BaseModel):
    id: int | None = None
    provider_id: int
    model_name: str
    startup_time_ms: float | None = None
    tokens_per_second: float | None = None
    created_at: datetime | None = None


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
    duration_ms: float = 0.0
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
