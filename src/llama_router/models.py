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


class ProviderProtocol(str, Enum):
    """The wire protocol the router speaks to a provider.

    Each provider has exactly one. ``ollama`` is the Ollama native API
    (``/api/*``); Ollama also serves the OpenAI-style routes. ``openai`` is the
    OpenAI-style API (``/v1/*``) that llama.cpp, vLLM, LiteLLM, Lemonade,
    LM Studio and the cloud services speak.
    """

    OLLAMA = "ollama"
    OPENAI = "openai"

    @classmethod
    def parse(cls, value: str | None) -> "ProviderProtocol":
        """Accept the protocol names, plus ``llamacpp`` from older clients."""
        v = (value or "").strip().lower()
        if v == "llamacpp":
            return cls.OPENAI
        return cls(v or cls.OLLAMA.value)


# The worker link names its upstreams by the older protocol names; keep the
# wire format so hubs and workers of different versions still talk.
WORKER_TARGETS: dict[ProviderProtocol, str] = {
    ProviderProtocol.OLLAMA: "ollama",
    ProviderProtocol.OPENAI: "llamacpp",
}


ENDPOINT_SUPPORTED = "supported"
ENDPOINT_MISSING = "missing"
ENDPOINT_UNKNOWN = "unknown"
# Probe states that mean the route exists (see probe.SUPPORTED).
_PROBE_SUPPORTED = ("available", "exists", "needs_key")


class Provider(BaseModel):
    id: int | None = None
    name: str
    protocol: ProviderProtocol = ProviderProtocol.OLLAMA
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
    # Server the last probe recognised ("vllm", "ollama", a cloud preset id,
    # ...). Shown in the dashboard; routing never reads it.
    flavor: str | None = None
    # Cloud preset the provider was created from (see cloud_presets).
    preset: str | None = None
    # Static headers sent with every request, e.g. {"anthropic-version": ...}.
    extra_headers: dict[str, str] | None = None
    # Models added by hand, for services that cannot list (all of) theirs.
    manual_models: list[str] | None = None

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
    def is_ollama(self) -> bool:
        return self.protocol is ProviderProtocol.OLLAMA

    @property
    def is_openai(self) -> bool:
        return self.protocol is ProviderProtocol.OPENAI

    def endpoint_state(self, path: str) -> str:
        """What the last probe said about *path*: supported, missing or unknown.

        An OpenAI route also counts under its bare form (``/v1/models`` as
        ``/models``): for an address that is itself the API base
        (``https://api.openai.com/v1``) the bare form is the one that answers.
        """
        caps = self.capabilities or {}
        paths = {path}
        if path.startswith("/v1/"):
            paths.add(path[3:])
        entries = [e for e in caps.get("endpoints") or [] if e.get("path") in paths]
        if not entries:
            return ENDPOINT_UNKNOWN
        if any(e.get("state") in _PROBE_SUPPORTED for e in entries):
            return ENDPOINT_SUPPORTED
        if any(e.get("state") == "missing" for e in entries):
            return ENDPOINT_MISSING
        return ENDPOINT_UNKNOWN

    def lacks(self, path: str) -> bool:
        """The last probe found *path* missing."""
        return self.endpoint_state(path) == ENDPOINT_MISSING

    @property
    def liveness_path(self) -> str | None:
        """The GET the last probe found cheapest for health checks."""
        return (self.capabilities or {}).get("liveness") or None

    @property
    def worker_target(self) -> str:
        """The worker link target that carries this provider's protocol."""
        return WORKER_TARGETS[self.protocol]

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


# A load faster than this means the model was already loaded: the run did not
# measure a cold start. Even a small model read from the page cache takes
# longer to load.
COLD_START_MIN_MS = 50.0


def measured_cold_start(startup_ms: float | None) -> float | None:
    """*startup_ms* if it is a real cold-start measurement, else None."""
    if startup_ms is None or startup_ms < COLD_START_MIN_MS:
        return None
    return startup_ms


class ProviderAddress(BaseModel):
    id: int | None = None
    provider_id: int
    url: str
    is_preferred: bool = False
    is_live: bool = False
    created_at: datetime | None = None


class RequestLog(BaseModel):
    id: int | None = None
    provider_id: int | None = None
    provider_name: str | None = None
    protocol: str  # API the request arrived on: "ollama" or "v1"
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
