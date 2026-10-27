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
"unexpected endpoint" message; those count as missing. So does a path a
worker refuses to relay (403 "worker does not allow ...").
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import asdict, dataclass, field
from typing import Any
from urllib.parse import urlparse

import httpx

from .cloud_presets import preset_for_url
from .provider_secrets import auth_headers
from .worker_link.protocol import refused as worker_refused

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
    # --- vLLM-specific routes (its OpenAI routes are covered above) ---
    _ep("vllm", "GET", "/version", "server version"),
    _ep("vllm", "GET", "/ping", "liveness (SageMaker)"),
    _ep("vllm", "GET", "/load", "server load"),
    _ep("vllm", "POST", "/invocations", "SageMaker invocations"),
    _ep("vllm", "POST", "/pooling", "pooling models"),
    _ep("vllm", "POST", "/classify", "classification"),
    _ep("vllm", "POST", "/score", "scoring"),
    _ep("vllm", "POST", "/v1/score", "scoring"),
    _ep("vllm", "POST", "/v2/rerank", "rerank (Cohere v2)"),
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
    # --- Other servers, for recognising them (their OpenAI routes are above) ---
    _ep("lemonade", "GET", "/api/v1/health", "Lemonade health"),
    _ep("lemonade", "GET", "/api/v1/models", "Lemonade models"),
    _ep("lmstudio", "GET", "/api/v0/models", "LM Studio models"),
    _ep("sglang", "GET", "/get_server_info", "SGLang server info"),
    _ep("sglang", "GET", "/get_model_info", "SGLang model info"),
    _ep("tgi", "GET", "/info", "TGI server info"),
    _ep("koboldcpp", "GET", "/api/v1/model", "KoboldCpp model"),
    _ep("koboldcpp", "GET", "/api/extra/version", "KoboldCpp version"),
    _ep("localai", "GET", "/readyz", "LocalAI readiness"),
]


def _dedupe(endpoints: list[Endpoint]) -> list[Endpoint]:
    """Probe each method and path once; the first family listed keeps it."""
    seen: set[tuple[str, str]] = set()
    out = []
    for ep in endpoints:
        key = (ep.method, ep.path)
        if key not in seen:
            seen.add(key)
            out.append(ep)
    return out


# Routes in the OpenAI-style API, whichever family the deduplicated catalog
# files them under (bare /models is also llama.cpp's router-mode listing).
_OPENAI_ROUTES = {(e.method, e.path) for e in CATALOG if e.family == "openai"}
CATALOG = _dedupe(CATALOG)

FAMILY_LABELS = {
    "vllm": "vLLM",
    "ollama": "Ollama",
    "llamacpp": "llama.cpp",
    "openai": "OpenAI-compatible",
    "anthropic": "Anthropic-compatible",
    "decisions": "Decision models",
    "litellm": "LiteLLM",
    "lemonade": "Lemonade",
    "lmstudio": "LM Studio",
    "sglang": "SGLang",
    "tgi": "Text Generation Inference",
    "koboldcpp": "KoboldCpp",
    "localai": "LocalAI",
}

# GETs that answer cheaply when a server is up, best first. The first one a
# probe finds available becomes the provider's health check.
LIVENESS_PATHS = (
    "/health",
    "/api/version",
    "/v1/models",
    "/models",
    "/api/v1/health",
    "/health/liveliness",
)

# The most specific recognised server names the provider's flavor.
_FLAVOR_ORDER = (
    "llama-router",
    "ollama",
    "lemonade",
    "lm-studio",
    "koboldcpp",
    "sglang",
    "tgi",
    "vllm",
    "llama.cpp",
    "litellm",
    "localai",
    "openai-compatible",
)

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
    # Most specific server recognised, e.g. "vllm" or a cloud preset id.
    flavor: str | None = None
    # Cloud preset the hostname belongs to, if any.
    preset: str | None = None
    # Path of the cheapest GET that answered, for health checks.
    liveness: str | None = None
    # A key was sent and every endpoint that answered refused it.
    key_rejected: bool = False
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
        "flavor": next((r.flavor for r in results if r.flavor), None),
        "preset": next((r.preset for r in results if r.preset), None),
        "liveness": next((r.liveness for r in results if r.liveness), None),
        "key_rejected": any(r.key_rejected for r in results),
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
        # A worker relaying the probe refuses paths it does not serve; that
        # route is missing through the worker, not behind a key.
        if worker_refused(code, resp.text):
            return MISSING
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


def _str(value: Any) -> str | None:
    return str(value) if value not in (None, "") else None


def _json_any(resp: httpx.Response | None) -> Any:
    """JSON body of any response, including errors."""
    if resp is None:
        return None
    try:
        return resp.json()
    except ValueError:
        return None


def _detect(
    result: ProbeResult, responses: dict[tuple[str, str], httpx.Response | None]
) -> None:
    def get(path: str, method: str = "GET") -> Any:
        resp = responses.get((method, path))
        # A 200 that the classifier calls missing ("unexpected endpoint") is
        # not an answer from that route.
        if resp is None or _classify(resp) != AVAILABLE:
            return None
        return _json(resp)

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

    # vLLM lists its models as owned_by "vllm" and serves /version beside a
    # SageMaker-style /ping. Either sign alone could match another server.
    owned_by_vllm = isinstance(models_json, dict) and any(
        isinstance(m, dict) and m.get("owned_by") == "vllm"
        for m in (models_json.get("data") or [])
        if isinstance(models_json.get("data"), list)
    )
    vllm_version = get("/version")
    ping = responses.get(("GET", "/ping"))
    has_version = isinstance(vllm_version, dict) and "version" in vllm_version
    if owned_by_vllm or (has_version and ping is not None and ping.status_code == 200):
        servers.append("vllm")
        if has_version and not result.version:
            result.version = str(vllm_version["version"])

    lemonade = get("/api/v1/health")
    if isinstance(lemonade, dict) and (
        "model_loaded" in lemonade or "checkpoint_loaded" in lemonade
    ):
        servers.append("lemonade")
        result.version = result.version or _str(lemonade.get("version"))

    lms = get("/api/v0/models")
    if isinstance(lms, dict) and isinstance(lms.get("data"), list):
        servers.append("lm-studio")

    sglang = get("/get_server_info")
    if isinstance(sglang, dict) and sglang:
        servers.append("sglang")
        result.version = result.version or _str(sglang.get("version"))

    tgi = get("/info")
    if (
        isinstance(tgi, dict)
        and "model_id" in tgi
        and ("router" in tgi or "max_input_tokens" in tgi or "max_input_length" in tgi)
    ):
        servers.append("tgi")
        result.version = result.version or _str(tgi.get("version"))

    kobold = get("/api/extra/version")
    if (
        isinstance(kobold, dict)
        and "koboldcpp" in str(kobold.get("result", "")).lower()
    ):
        servers.append("koboldcpp")
        result.version = result.version or _str(kobold.get("version"))

    ready = responses.get(("GET", "/readyz"))
    if (
        ready is not None
        and _classify(ready) == AVAILABLE
        and "openai-compatible" in servers
        and not {"vllm", "llama.cpp", "sglang", "tgi", "litellm"} & set(servers)
    ):
        servers.append("localai")

    messages = responses.get(("POST", "/v1/messages"))
    if messages is not None and messages.status_code in (400, 401, 403):
        err = _json_any(messages)
        if isinstance(err, dict) and err.get("type") == "error":
            servers.append("anthropic-api")

    result.servers = servers
    flavor = next((f for f in _FLAVOR_ORDER if f in servers), None)
    result.flavor = result.preset or flavor
    if "ollama" in servers or "llama-router" in servers:
        result.suggested_type = "ollama"
    elif servers:
        result.suggested_type = "openai"

    # Servers answer with all sorts of shapes; never let one crash the probe.
    names: list[str] = []

    def entries(obj: Any, key: str) -> list[dict[str, Any]]:
        val = obj.get(key) if isinstance(obj, dict) else None
        return [m for m in val if isinstance(m, dict)] if isinstance(val, list) else []

    for m in entries(get("/api/tags"), "models"):
        if m.get("name") and str(m["name"]) not in names:
            names.append(str(m["name"]))
    for m in entries(models_json, "data"):
        if m.get("id") and str(m["id"]) not in names:
            names.append(str(m["id"]))
    result.models = names[:_MAX_MODELS]


async def probe_url(
    base_url: str,
    *,
    api_key: str | None = None,
    api_key_header: str | None = None,
    headers: dict[str, str] | None = None,
    transport: httpx.AsyncBaseTransport | None = None,
    key_sent: bool | None = None,
) -> ProbeResult:
    """Probe every catalog endpoint under *base_url* concurrently.

    Pass either *api_key* (with an optional header name) or ready *headers*;
    *key_sent* says whether those headers carry a key (default: any headers).
    """
    if key_sent is None:
        key_sent = bool(api_key or headers)
    base = normalize_base_url(base_url)
    result = ProbeResult(base_url=base)
    if headers is None and api_key:
        headers = auth_headers(api_key, api_key_header)
    preset = preset_for_url(base)
    catalog = CATALOG
    if preset is not None:
        # A known cloud service: skip the self-hosted servers' routes, which
        # only cost requests against its rate limit.
        result.preset = preset.id
        result.suggested_type = preset.protocol
        families = {"openai", "anthropic"} | (
            {"ollama"} if preset.protocol == "ollama" else set()
        )
        catalog = [
            ep
            for ep in CATALOG
            if ep.family in families or (ep.method, ep.path) in _OPENAI_ROUTES
        ]
        if preset.endpoints:
            # A service that declares its routes is probed on those alone.
            served = set(preset.endpoints)
            catalog = [
                ep for ep in CATALOG if ep.path in served or "/v1" + ep.path in served
            ]
    started = time.monotonic()
    sem = asyncio.Semaphore(_CONCURRENCY)
    async with httpx.AsyncClient(
        base_url=base,
        timeout=_TIMEOUT,
        headers=headers,
        transport=transport,
        follow_redirects=False,
    ) as client:
        pairs = await asyncio.gather(*(_probe_one(client, ep, sem) for ep in catalog))
    result.endpoints = [r for r, _ in pairs]
    responses = {(r.method, r.path): resp for r, resp in pairs}
    result.reachable = any(r.status_code is not None for r in result.endpoints)
    result.auth_required = any(r.state == NEEDS_KEY for r in result.endpoints)
    answered = [r for r in result.endpoints if r.status_code is not None]
    result.key_rejected = bool(
        key_sent and answered and all(r.state == NEEDS_KEY for r in answered)
    )
    states = {(r.method, r.path): r.state for r in result.endpoints}
    result.liveness = next(
        (p for p in LIVENESS_PATHS if states.get(("GET", p)) == AVAILABLE), None
    )
    if result.reachable:
        suggested = result.suggested_type
        _detect(result, responses)
        if preset is not None:
            result.suggested_type = suggested
    else:
        result.error = "no response from any endpoint"
    result.duration_ms = (time.monotonic() - started) * 1000
    return result
