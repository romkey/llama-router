"""Probe a base URL to identify the server and the endpoints it exposes.

Used when adding a provider (to suggest its type and first address) and on a
provider's page (to record which endpoints its upstream supports).

Probes never run inference on purpose. GET endpoints are fetched; POST
endpoints get an empty JSON object, which every known server rejects with a
validation error before doing any work. The answer is read as:

======================  ===========================================
response                meaning
======================  ===========================================
2xx                     available
400, 405, 409, 415, 422 exists (rejected the empty probe)
401, 403                needs a key
404, 501                missing
other / transport       error
======================  ===========================================

Some servers (LM Studio, for example) answer unknown routes with 200 and an
"unexpected endpoint" message; those count as missing.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import asdict, dataclass, field
from typing import Any
from urllib.parse import urlparse

import httpx

from .provider_secrets import auth_headers

_TIMEOUT = httpx.Timeout(4.0)
_CONCURRENCY = 12
_MAX_MODELS = 500


@dataclass(frozen=True)
class Endpoint:
    path: str
    method: str  # GET, POST or HEAD
    family: str
    purpose: str


def _ep(family: str, method: str, path: str, purpose: str) -> Endpoint:
    return Endpoint(path=path, method=method, family=family, purpose=purpose)


def _openai(path: str, method: str, purpose: str) -> list[Endpoint]:
    """OpenAI-style routes are served both bare and under /v1."""
    return [
        _ep("openai", method, f"/v1{path}", purpose),
        _ep("openai", method, path, purpose),
    ]


CATALOG: list[Endpoint] = [
    # --- Ollama native API ---
    _ep("ollama", "GET", "/api/version", "server version"),
    _ep("ollama", "GET", "/api/tags", "list local models"),
    _ep("ollama", "GET", "/api/ps", "list loaded models"),
    _ep("ollama", "POST", "/api/show", "model details"),
    _ep("ollama", "POST", "/api/chat", "chat"),
    _ep("ollama", "POST", "/api/generate", "text generation"),
    _ep("ollama", "POST", "/api/embed", "embeddings"),
    _ep("ollama", "POST", "/api/embeddings", "embeddings (legacy)"),
    _ep("ollama", "POST", "/api/pull", "download a model"),
    _ep("ollama", "POST", "/api/create", "create a model"),
    _ep("ollama", "HEAD", "/api/blobs/sha256:" + "0" * 64, "blob upload"),
    # --- llama.cpp server native API ---
    _ep("llamacpp", "GET", "/health", "health"),
    _ep("llamacpp", "GET", "/props", "server properties"),
    _ep("llamacpp", "GET", "/slots", "slot state"),
    _ep("llamacpp", "GET", "/metrics", "Prometheus metrics"),
    _ep("llamacpp", "GET", "/models", "router mode: list models"),
    _ep("llamacpp", "POST", "/models/load", "router mode: load a model"),
    _ep("llamacpp", "POST", "/completion", "native completion"),
    _ep("llamacpp", "POST", "/infill", "code infill"),
    _ep("llamacpp", "POST", "/tokenize", "tokenize"),
    _ep("llamacpp", "POST", "/detokenize", "detokenize"),
    _ep("llamacpp", "POST", "/apply-template", "apply chat template"),
    _ep("llamacpp", "POST", "/embedding", "native embeddings"),
    _ep("llamacpp", "POST", "/rerank", "rerank"),
    _ep("llamacpp", "POST", "/reranking", "rerank (alias)"),
    _ep("llamacpp", "GET", "/lora-adapters", "LoRA adapters"),
    # --- OpenAI-compatible API (bare and /v1) ---
    *_openai("/models", "GET", "list models"),
    *_openai("/chat/completions", "POST", "chat completions"),
    *_openai("/completions", "POST", "text completions"),
    *_openai("/responses", "POST", "Responses API"),
    *_openai("/embeddings", "POST", "embeddings"),
    *_openai("/rerank", "POST", "rerank"),
    *_openai("/moderations", "POST", "moderation"),
    *_openai("/images/generations", "POST", "image generation"),
    *_openai("/images/edits", "POST", "image edits"),
    *_openai("/images/variations", "POST", "image variations"),
    *_openai("/audio/transcriptions", "POST", "speech to text"),
    *_openai("/audio/translations", "POST", "speech translation"),
    *_openai("/audio/speech", "POST", "text to speech"),
    *_openai("/audio/voices", "GET", "list voices"),
    *_openai("/files", "GET", "files"),
    *_openai("/batches", "GET", "batches"),
    *_openai("/fine_tuning/jobs", "GET", "fine-tuning jobs"),
    *_openai("/vector_stores", "GET", "vector stores"),
    *_openai("/assistants", "GET", "assistants"),
    *_openai("/realtime", "GET", "realtime (WebSocket upgrade)"),
    # --- Anthropic-compatible API ---
    _ep("anthropic", "POST", "/v1/messages", "Anthropic Messages"),
    _ep("anthropic", "POST", "/v1/messages/count_tokens", "token counting"),
    # --- Decision models (LiteLLM) ---
    _ep("decisions", "POST", "/v1/decisions", "decision models"),
    _ep("decisions", "POST", "/decisions", "decision models (alias)"),
    _ep("decisions", "POST", "/v1/systemone", "System One decisions"),
    _ep("decisions", "POST", "/systemone", "System One decisions (alias)"),
    # --- Other LiteLLM proxy routes ---
    _ep("litellm", "GET", "/health/liveliness", "liveness"),
    _ep("litellm", "GET", "/health/readiness", "readiness"),
    _ep("litellm", "GET", "/model/info", "model info"),
    _ep("litellm", "GET", "/v1/model/info", "model info"),
    _ep("litellm", "POST", "/responses/compact", "compacted Responses"),
    _ep("litellm", "POST", "/converse", "Bedrock Converse"),
    _ep("litellm", "POST", "/invoke", "Bedrock invoke"),
    _ep("litellm", "POST", "/generateContent", "Gemini generateContent"),
    _ep("litellm", "POST", "/ocr", "OCR"),
    _ep("litellm", "POST", "/search", "search"),
    _ep("litellm", "POST", "/videos", "video generation"),
    _ep("litellm", "POST", "/utils/token_counter", "token counter"),
    _ep("litellm", "POST", "/v1/extraction", "extraction"),
    _ep("litellm", "POST", "/guardrails/apply_guardrail", "guardrails"),
]

FAMILY_LABELS = {
    "ollama": "Ollama",
    "llamacpp": "llama.cpp",
    "openai": "OpenAI-compatible",
    "anthropic": "Anthropic-compatible",
    "decisions": "Decision models",
    "litellm": "LiteLLM",
}

AVAILABLE = "available"
EXISTS = "exists"
NEEDS_KEY = "needs_key"
MISSING = "missing"
ERROR = "error"

SUPPORTED = (AVAILABLE, EXISTS, NEEDS_KEY)


@dataclass
class EndpointResult:
    path: str
    method: str
    family: str
    purpose: str
    state: str
    status_code: int | None = None


@dataclass
class ProbeResult:
    base_url: str
    reachable: bool = False
    auth_required: bool = False
    servers: list[str] = field(default_factory=list)
    version: str | None = None
    suggested_type: str | None = None
    models: list[str] = field(default_factory=list)
    endpoints: list[EndpointResult] = field(default_factory=list)
    duration_ms: float = 0.0
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["supported"] = sorted(
            {f"{e.method} {e.path}" for e in self.endpoints if e.state in SUPPORTED}
        )
        d["families"] = {
            fam: sum(
                1 for e in self.endpoints if e.family == fam and e.state in SUPPORTED
            )
            for fam in FAMILY_LABELS
        }
        return d


def merge_results(results: list[ProbeResult]) -> dict[str, Any]:
    """Combine probes of one provider's URLs into the stored capabilities."""
    endpoints: list[dict[str, Any]] = []
    servers: list[str] = []
    models: list[str] = []
    for r in results:
        for e in r.endpoints:
            endpoints.append({**asdict(e), "base_url": r.base_url})
        servers += [s for s in r.servers if s not in servers]
        models += [m for m in r.models if m not in models]
    supported = sorted(
        {f"{e['method']} {e['path']}" for e in endpoints if e["state"] in SUPPORTED}
    )
    return {
        "base_urls": [r.base_url for r in results],
        "reachable": any(r.reachable for r in results),
        "auth_required": any(r.auth_required for r in results),
        "servers": servers,
        "version": next((r.version for r in results if r.version), None),
        "suggested_type": next(
            (r.suggested_type for r in results if r.suggested_type), None
        ),
        "models": models[:_MAX_MODELS],
        "supported": supported,
        "families": {
            fam: len(
                {
                    e["path"]
                    for e in endpoints
                    if e["family"] == fam and e["state"] in SUPPORTED
                }
            )
            for fam in FAMILY_LABELS
        },
        "endpoints": endpoints,
    }


def normalize_base_url(url: str) -> str:
    """Validate a probe target: http(s) with a host, no credentials, no query."""
    raw = (url or "").strip().rstrip("/")
    parsed = urlparse(raw)
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        raise ValueError("URL must be http:// or https:// with a host")
    if parsed.username or parsed.password:
        raise ValueError("put credentials in the API key field, not the URL")
    if parsed.query or parsed.fragment:
        raise ValueError("URL must not have a query or fragment")
    return raw


def _classify(resp: httpx.Response) -> str:
    code = resp.status_code
    if 200 <= code < 300:
        body = resp.text[:300].lower() if resp.content else ""
        if "unexpected endpoint" in body:
            return MISSING
        return AVAILABLE
    if code in (401, 403):
        return NEEDS_KEY
    if code in (400, 405, 409, 415, 422):
        return EXISTS
    if code in (404, 501):
        return MISSING
    return ERROR


async def _probe_one(
    client: httpx.AsyncClient, ep: Endpoint, sem: asyncio.Semaphore
) -> tuple[EndpointResult, httpx.Response | None]:
    async with sem:
        try:
            if ep.method == "POST":
                resp = await client.post(ep.path, json={})
            elif ep.method == "HEAD":
                resp = await client.head(ep.path)
            else:
                resp = await client.get(ep.path)
        except httpx.HTTPError:
            return (
                EndpointResult(ep.path, ep.method, ep.family, ep.purpose, ERROR, None),
                None,
            )
    return (
        EndpointResult(
            ep.path, ep.method, ep.family, ep.purpose, _classify(resp), resp.status_code
        ),
        resp,
    )


def _json(resp: httpx.Response | None) -> Any:
    if resp is None or not (200 <= resp.status_code < 300):
        return None
    try:
        return resp.json()
    except ValueError:
        return None


def _detect(
    result: ProbeResult, responses: dict[tuple[str, str], httpx.Response | None]
) -> None:
    def get(path: str, method: str = "GET") -> Any:
        return _json(responses.get((method, path)))

    servers: list[str] = []
    version = get("/api/version")
    if isinstance(version, dict) and "version" in version:
        servers.append("ollama")
        result.version = str(version["version"])

    props = get("/props")
    if isinstance(props, dict) and (
        "default_generation_settings" in props
        or "model_path" in props
        or "build_info" in props
        or "total_slots" in props
    ):
        servers.append("llama.cpp")
        if not result.version and props.get("build_info"):
            result.version = str(props["build_info"])

    health = get("/health")
    if isinstance(health, dict) and "providers" in health and "version" in health:
        servers.append("llama-router")
        result.version = result.version or str(health["version"])

    liveliness = responses.get(("GET", "/health/liveliness"))
    if liveliness is not None and liveliness.status_code == 200:
        if "alive" in liveliness.text.lower():
            servers.append("litellm")

    models_json = get("/v1/models") or get("/models")
    if isinstance(models_json, dict) and isinstance(models_json.get("data"), list):
        servers.append("openai-compatible")

    result.servers = servers
    if "ollama" in servers or "llama-router" in servers:
        result.suggested_type = "ollama"
    elif servers:
        result.suggested_type = "llamacpp"

    names: list[str] = []
    tags = get("/api/tags")
    if isinstance(tags, dict):
        names += [str(m.get("name")) for m in tags.get("models", []) if m.get("name")]
    if isinstance(models_json, dict):
        for m in models_json.get("data") or []:
            if isinstance(m, dict) and m.get("id") and str(m["id"]) not in names:
                names.append(str(m["id"]))
    result.models = names[:_MAX_MODELS]


async def probe_url(
    base_url: str,
    *,
    api_key: str | None = None,
    api_key_header: str | None = None,
    headers: dict[str, str] | None = None,
    transport: httpx.AsyncBaseTransport | None = None,
) -> ProbeResult:
    """Probe every catalog endpoint under *base_url* concurrently.

    Pass either *api_key* (with an optional header name) or ready *headers*.
    """
    base = normalize_base_url(base_url)
    result = ProbeResult(base_url=base)
    if headers is None and api_key:
        headers = auth_headers(api_key, api_key_header)
    started = time.monotonic()
    sem = asyncio.Semaphore(_CONCURRENCY)
    async with httpx.AsyncClient(
        base_url=base,
        timeout=_TIMEOUT,
        headers=headers,
        transport=transport,
        follow_redirects=False,
    ) as client:
        pairs = await asyncio.gather(*(_probe_one(client, ep, sem) for ep in CATALOG))
    result.endpoints = [r for r, _ in pairs]
    responses = {(r.method, r.path): resp for r, resp in pairs}
    result.reachable = any(r.status_code is not None for r in result.endpoints)
    result.auth_required = any(r.state == NEEDS_KEY for r in result.endpoints)
    if result.reachable:
        _detect(result, responses)
    else:
        result.error = "no response from any endpoint"
    result.duration_ms = (time.monotonic() - started) * 1000
    return result
