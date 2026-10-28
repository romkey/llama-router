"""Session affinity: keep one conversation on one provider.

Coding clients send a long conversation as a series of requests, each
repeating the same system prompt and history. llama.cpp and Ollama only reuse
their prompt cache when the next request reaches the same backend, so the
router remembers which provider served each conversation and keeps sending it
there while that provider stays in the best routing tier.

A conversation is identified by, in order:

1. a session header the client sends (``X-Session-ID``, ``Session-ID`` or
   ``session_id``);
2. ``prompt_cache_key``, ``metadata.session_id``, ``user`` or
   ``metadata.user_id`` in the request body;
3. a fingerprint of the start of the conversation: the system prompt and the
   first user message for chat, the instructions and first input item for
   ``/v1/responses``, or the system prompt and the first 2 KB of a long
   completion prompt.

Keys are scoped by API key and requested model. A binding holds the
provider and the model it served, so a conversation that asked for a model
group stays on the same member model as well as the same provider.
Everything here is in memory; nothing touches the database, and bindings
are lost on restart.
"""

from __future__ import annotations

import hashlib
import time
from collections import OrderedDict
from collections.abc import Callable, Iterable, Iterator, Mapping
from typing import Any, NamedTuple

SESSION_HEADERS = ("x-session-id", "session-id", "session_id")
# Roles that make up the stable preamble of a chat conversation.
PREAMBLE_ROLES = frozenset({"system", "developer"})
# A completion prompt is fingerprinted by this many leading characters, and
# only when it is at least this long, so the key stays stable as the prompt
# grows by appending. Short prompts (one-off calls, autocomplete) get none.
PROMPT_PREFIX_CHARS = 2048
# Sweep at most this many expired entries per bind.
_SWEEP_BATCH = 16


def _texts(content: Any) -> Iterator[str]:
    """Text in a message's content: a string, or a list of typed parts."""
    if isinstance(content, str):
        yield content
    elif isinstance(content, list):
        for part in content:
            if isinstance(part, str):
                yield part
            elif isinstance(part, dict):
                text = part.get("text")
                if isinstance(text, str):
                    yield text


def _chat_parts(messages: Any) -> Iterator[str]:
    """Role and text of the preamble plus the first message after it."""
    if not isinstance(messages, list):
        return
    for m in messages:
        if not isinstance(m, dict):
            continue
        role = str(m.get("role") or "")
        yield role
        yield from _texts(m.get("content"))
        if role not in PREAMBLE_ROLES:
            return


def _responses_parts(body: Mapping[str, Any]) -> Iterator[str]:
    instructions = body.get("instructions")
    if isinstance(instructions, str):
        yield "instructions"
        yield instructions
    inp = body.get("input")
    if isinstance(inp, str):
        yield inp
    elif isinstance(inp, list):
        # Same shape as chat messages; system items come first.
        yield from _chat_parts(inp)


def _prompt_parts(body: Mapping[str, Any]) -> Iterator[str]:
    prompt = body.get("prompt")
    if isinstance(prompt, list):
        prompt = prompt[0] if prompt else None
    if not isinstance(prompt, str) or len(prompt) < PROMPT_PREFIX_CHARS:
        return
    system = body.get("system")
    if isinstance(system, str):
        yield "system"
        yield system
    yield prompt[:PROMPT_PREFIX_CHARS]


def _fingerprint(parts: Iterable[str]) -> str | None:
    h = hashlib.blake2b(b"fp", digest_size=16)
    fed = False
    for p in parts:
        if p:
            fed = True
        h.update(p.encode("utf-8", "surrogatepass"))
        h.update(b"\x00")
    return h.hexdigest() if fed else None


def _explicit_id(headers: Mapping[str, str], body: Mapping[str, Any]) -> str | None:
    for name in SESSION_HEADERS:
        v = headers.get(name)
        if v and v.strip():
            return v.strip()
    meta = body.get("metadata")
    meta = meta if isinstance(meta, dict) else {}
    for v in (
        body.get("prompt_cache_key"),
        meta.get("session_id"),
        body.get("user"),
        meta.get("user_id"),
    ):
        if isinstance(v, (str, int)) and str(v).strip():
            return str(v).strip()
    return None


def session_key(
    headers: Mapping[str, str],
    body: Mapping[str, Any],
    key_id: int | None = None,
) -> str | None:
    """The affinity key for a request, or None when it has nothing stable.

    *headers* is a case-insensitive mapping such as ``request.headers``.
    """
    if not isinstance(body, Mapping):
        return None
    model = body.get("model")
    if not isinstance(model, str) or not model:
        return None
    explicit = _explicit_id(headers, body)
    digest: str | None
    if explicit is not None:
        digest = hashlib.blake2b(
            b"id\x00" + explicit.encode("utf-8", "surrogatepass"), digest_size=16
        ).hexdigest()
    elif "messages" in body:
        digest = _fingerprint(_chat_parts(body.get("messages")))
    elif "input" in body or "instructions" in body:
        digest = _fingerprint(_responses_parts(body))
    elif "prompt" in body:
        digest = _fingerprint(_prompt_parts(body))
    else:
        digest = None
    if digest is None:
        return None
    return f"{key_id or 0}:{model}:{digest}"


def short_key(key: str | None) -> str | None:
    """A session key trimmed for display: scope plus the first hash digits."""
    if not key:
        return None
    scope, _, digest = key.rpartition(":")
    return f"{scope}:{digest[:12]}"


class Binding(NamedTuple):
    """Where a session was last served: the provider, and the model it ran
    there (None matches any model on the provider)."""

    provider_id: int
    model: str | None = None

    def matches(self, provider_id: int, model: str) -> bool:
        return self.provider_id == provider_id and (
            self.model is None or self.model == model
        )


class SessionAffinity:
    """Session key -> binding (provider and model), an LRU with an idle
    timeout.

    Synchronous and in memory. Entries are kept in least-recently-used order,
    so expired entries collect at the front and are swept a few at a time as
    new sessions are bound; no background task is needed.
    """

    def __init__(
        self,
        ttl_s: float,
        max_sessions: int,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.ttl_s = float(ttl_s)
        self.max_sessions = max(1, int(max_sessions))
        self._clock = clock
        self._entries: OrderedDict[str, tuple[Binding, float]] = OrderedDict()
        self.hits = 0  # request went to its bound provider
        self.misses = 0  # new session, bound to the normal choice
        self.rebinds = 0  # bound provider was not eligible; moved

    def __len__(self) -> int:
        return len(self._entries)

    def peek(self, key: str | None) -> Binding | None:
        """The live binding for *key*, without touching or counting it."""
        if not key:
            return None
        entry = self._entries.get(key)
        if entry is None:
            return None
        if self._clock() - entry[1] >= self.ttl_s:
            return None
        return entry[0]

    def get(self, key: str | None) -> Binding | None:
        """The live binding for *key*; drops it if it has expired."""
        if not key:
            return None
        entry = self._entries.get(key)
        if entry is None:
            return None
        if self._clock() - entry[1] >= self.ttl_s:
            del self._entries[key]
            return None
        return entry[0]

    def bind(
        self,
        key: str,
        provider_id: int,
        previous: Binding | None = None,
        model: str | None = None,
    ) -> None:
        """Record that *key* was routed to *model* on *provider_id*.

        *previous* is what :meth:`get` returned for this request, used only
        for the hit, miss and rebind counters.
        """
        now = self._clock()
        if previous is None:
            self.misses += 1
        elif previous.provider_id == provider_id and previous.model in (None, model):
            self.hits += 1
        else:
            self.rebinds += 1
        entries = self._entries
        entries[key] = (Binding(provider_id, model), now)
        entries.move_to_end(key)
        for _ in range(_SWEEP_BATCH):
            first = next(iter(entries.items()), None)
            if first is None or now - first[1][1] < self.ttl_s:
                break
            entries.popitem(last=False)
        while len(entries) > self.max_sessions:
            entries.popitem(last=False)

    def forget(self, key: str | None) -> None:
        if key:
            self._entries.pop(key, None)

    def forget_providers_not_in(self, provider_ids: set[int]) -> int:
        """Drop bindings to providers that no longer exist. Returns the count."""
        stale = [
            k
            for k, (b, _) in self._entries.items()
            if b.provider_id not in provider_ids
        ]
        for k in stale:
            del self._entries[k]
        return len(stale)

    def clear(self) -> None:
        self._entries.clear()

    def stats(self) -> dict[str, Any]:
        now = self._clock()
        live = sum(1 for _, seen in self._entries.values() if now - seen < self.ttl_s)
        per_provider: dict[int, int] = {}
        for b, seen in self._entries.values():
            if now - seen < self.ttl_s:
                per_provider[b.provider_id] = per_provider.get(b.provider_id, 0) + 1
        return {
            "sessions": live,
            "per_provider": per_provider,
            "hits": self.hits,
            "misses": self.misses,
            "rebinds": self.rebinds,
            "ttl_seconds": self.ttl_s,
            "max_sessions": self.max_sessions,
        }
