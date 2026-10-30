"""Wire format shared by the hub and worker sides.

Control messages are JSON text frames with a ``type`` field. Body bytes travel
in binary frames: one byte of kind, a 4-byte big-endian request id, then the
payload. Many requests are multiplexed over one socket by request id.

Hub -> worker
    ``welcome``   {worker_name, provider_id, version, router_id}
    ``request``   {id, target, method, path, headers}
                  followed by zero or more REQ_BODY frames and one REQ_END
    ``cancel``    {id}

Worker -> hub
    ``hello``     {version, protocols, max_concurrency, software_version, router_id}
    ``response``  {id, status, headers}
                  followed by zero or more RESP_BODY frames and one RESP_END
    ``error``     {id, kind, message}  (instead of, or after, a response)
"""

from __future__ import annotations

import json
import struct
from collections.abc import Iterator
from typing import Any

PROTOCOL_VERSION = 1

WS_PATH = "/worker/ws"

# Binary frame kinds.
REQ_BODY = 1
REQ_END = 2
RESP_BODY = 3
RESP_END = 4

_HEADER = struct.Struct(">BI")

# Body chunks are split to keep individual frames well under max frame size.
MAX_CHUNK = 256 * 1024
# Upper bound for one WebSocket message on both sides.
MAX_MESSAGE_SIZE = 4 * 1024 * 1024

TARGETS = ("ollama", "llamacpp")

# A worker answers a path outside its allowlist with 403 and an error that
# starts with this. Every worker version uses the same words, so the hub can
# tell a refusal from an upstream that wants a key.
REFUSED_PREFIX = "worker does not allow"


def refused(status: int, body: str) -> bool:
    """A worker's reply saying it does not relay that path."""
    return status == 403 and REFUSED_PREFIX in body[:300]


# WebSocket close codes (4000-4999 are application-defined).
CLOSE_UNAUTHORIZED = 4401
CLOSE_REPLACED = 4409
CLOSE_PROTOCOL = 4400
CLOSE_REVOKED = 4403
# The worker and the hub are the same router (same router ID).
CLOSE_LOOP = 4508

# Hop-by-hop or transport headers that must not be relayed in either direction.
_DROP_HEADERS = frozenset(
    {
        "host",
        "connection",
        "keep-alive",
        "proxy-connection",
        "transfer-encoding",
        "upgrade",
        "te",
        "trailer",
        "content-length",
        "content-encoding",
        "accept-encoding",
        "authorization",
        "api-key",
        "cookie",
        "set-cookie",
    }
)


def encode_binary(kind: int, req_id: int, payload: bytes = b"") -> bytes:
    return _HEADER.pack(kind, req_id) + payload


def decode_binary(data: bytes) -> tuple[int, int, bytes]:
    if len(data) < _HEADER.size:
        raise ValueError("binary frame too short")
    kind, req_id = _HEADER.unpack_from(data)
    return kind, req_id, data[_HEADER.size :]


def encode_json(msg: dict[str, Any]) -> str:
    return json.dumps(msg, separators=(",", ":"))


def decode_json(text: str) -> dict[str, Any]:
    msg = json.loads(text)
    if not isinstance(msg, dict) or not isinstance(msg.get("type"), str):
        raise ValueError("control frame must be an object with a type")
    return msg


def filter_headers(pairs: Any) -> list[list[str]]:
    """Drop hop-by-hop and credential headers; normalise to ``[[k, v], ...]``."""
    out: list[list[str]] = []
    if not pairs:
        return out
    items = pairs.items() if hasattr(pairs, "items") else pairs
    for item in items:
        try:
            k, v = item
        except TypeError, ValueError:
            continue
        k = str(k)
        if k.lower() in _DROP_HEADERS:
            continue
        out.append([k, str(v)])
    return out


def chunks(data: bytes, size: int = MAX_CHUNK) -> Iterator[bytes]:
    for i in range(0, len(data), size):
        yield data[i : i + size]


# Sent by a worker on every job it relays to its upstream. When the upstream
# is the worker's own router, it limits the job to the providers shared with
# the hub (Provider.hub_shared). Any other upstream ignores it. A client can
# only narrow its own routing by sending it, so it is not a credential.
HUB_HEADER = "X-Llama-Router-Via"
HUB_HEADER_VALUE = "hub"

# Sent by a worker that shares its providers individually, with the id of the
# one provider (on the worker's router) a relayed job is for. The worker's
# router reads it only alongside HUB_HEADER and only ever narrows routing.
PROVIDER_HEADER = "X-Llama-Router-Provider"

# How a worker shares its providers: as one provider on the hub, or each as
# its own (``catalog`` frames list them).
EXPOSE_COMBINED = "combined"
EXPOSE_INDIVIDUAL = "individual"
EXPOSE_MODES = (EXPOSE_COMBINED, EXPOSE_INDIVIDUAL)
