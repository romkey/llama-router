"""The endpoints llama-router proxies, and what routing needs to know of each.

Each endpoint has a protocol (which providers may serve it) and a class.
Timings are only comparable within a class: an embedding call and a chat
completion on the same model behave nothing alike, so live stats are kept
per class.
"""

from __future__ import annotations

GENERATE = "generate"
EMBED = "embed"
IMAGE = "image"
AUDIO = "audio"
# One-token decision questions (see decisions.py): a few hundred milliseconds,
# nothing like a chat completion, so their timings are kept apart.
DECIDE = "decide"
CLASSES = (GENERATE, EMBED, IMAGE, AUDIO, DECIDE)

# path -> (protocol, class). Ollama native routes need an Ollama-capable
# provider (protocol "ollama"); OpenAI-style routes take any provider (None).
ENDPOINTS: dict[str, tuple[str | None, str]] = {
    "/api/chat": ("ollama", GENERATE),
    "/api/generate": ("ollama", GENERATE),
    "/api/embed": ("ollama", EMBED),
    "/api/embeddings": ("ollama", EMBED),
    "/v1/chat/completions": (None, GENERATE),
    "/v1/completions": (None, GENERATE),
    "/v1/responses": (None, GENERATE),
    "/v1/embeddings": (None, EMBED),
    "/v1/images/generations": (None, IMAGE),
    "/v1/images/edits": (None, IMAGE),
    "/v1/audio/speech": (None, AUDIO),
    "/v1/audio/transcriptions": (None, AUDIO),
    # Decision requests: passed through to an upstream that serves the same
    # route, otherwise answered from chat completion logprobs.
    "/v1/decisions": (None, DECIDE),
    "/v1/systemone": (None, DECIDE),
}

DECISION_PATHS = tuple(p for p, (_proto, cls) in ENDPOINTS.items() if cls == DECIDE)

# Kept for callers that only need the protocol.
ENDPOINT_PROTOCOLS: dict[str, str | None] = {p: v[0] for p, v in ENDPOINTS.items()}


class UnknownEndpointError(ValueError):
    pass


def normalize_endpoint(path: str | None) -> str | None:
    """Upstream path a request is served on: bare OpenAI routes gain /v1."""
    if not path:
        return None
    if path.startswith("/v1/") or path.startswith("/api/"):
        return path
    return "/v1" + path


def resolve_endpoint(path: str) -> str:
    """Normalise *path* and check it is a proxied endpoint."""
    norm = normalize_endpoint(path)
    if norm not in ENDPOINTS:
        raise UnknownEndpointError(f"not a routed endpoint: {path!r}")
    return norm


def protocol_for_endpoint(path: str | None) -> str | None:
    norm = normalize_endpoint(path)
    entry = ENDPOINTS.get(norm or "")
    return entry[0] if entry else None


def endpoint_class(path: str | None) -> str:
    norm = normalize_endpoint(path)
    entry = ENDPOINTS.get(norm or "")
    return entry[1] if entry else GENERATE
