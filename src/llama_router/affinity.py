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

Keys are scoped by API key (by client address for requests without one) and
requested model, and each scope may hold only so many sessions. A binding
holds the
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


def _scope(key_id: int | None, client: str | None) -> str:
    """Who owns a session: the API key, else a hash of the client address
    (kept out of the key in plain form, and free of IPv6 colons)."""
    if key_id:
        return str(key_id)
    if client:
        return "ip-" + hashlib.blake2b(client.encode(), digest_size=5).hexdigest()
    return "0"


def session_key(
    headers: Mapping[str, str],
    body: Mapping[str, Any],
    key_id: int | None = None,
    client: str | None = None,
) -> str | None:
    """The affinity key for a request, or None when it has nothing stable.

    *headers* is a case-insensitive mapping such as ``request.headers``.
    *client* is the client's address; it scopes requests without an API key,
    so one anonymous client cannot claim or overwrite another's session.
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
        # Anthropic Messages keeps its system prompt beside the messages.
        system = body.get("system")
        lead = ["system", *_texts(system)] if system else []
        digest = _fingerprint([*lead, *_chat_parts(body.get("messages"))])
    elif "input" in body or "instructions" in body:
        digest = _fingerprint(_responses_parts(body))
    elif "prompt" in body:
        digest = _fingerprint(_prompt_parts(body))
    else:
        digest = None
    if digest is None:
        return None
    return f"{_scope(key_id, client)}:{model}:{digest}"


def short_key(key: str | None) -> str | None:
    """A session key trimmed for display: scope plus the first hash digits."""
    if not key:
        return None
    scope, _, digest = key.rpartition(":")
    return f"{scope}:{digest[:12]}"


def _scope_of(key: str) -> str:
    """The scope part of a session key (everything before the first colon)."""
    return key.partition(":")[0]


class Binding(NamedTuple):
    """Where a session was last served: the provider, the model it ran there
    (None matches any model on the provider), and the estimated prompt
    tokens of that request, which the provider's prompt cache now holds."""

    provider_id: int
    model: str | None = None
    prompt_tokens: int = 0

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

    *max_per_key* caps the sessions one scope (API key or anonymous client)
    holds; its least recently used go first. 0 means no cap.
    """

    def __init__(
        self,
        ttl_s: float,
        max_sessions: int,
        clock: Callable[[], float] = time.monotonic,
        max_per_key: int = 0,
    ) -> None:
        self.ttl_s = float(ttl_s)
        self.max_sessions = max(1, int(max_sessions))
        self.max_per_key = max(0, int(max_per_key))
        self._clock = clock
        self._entries: OrderedDict[str, tuple[Binding, float]] = OrderedDict()
        # Each scope's keys in the same LRU order as _entries, so the
        # scope's oldest session is found without scanning every entry.
        self._per_scope: dict[str, OrderedDict[str, None]] = {}
        self.hits = 0  # request went to its bound provider
        self.misses = 0  # new session, bound to the normal choice
        self.rebinds = 0  # bound provider was not eligible; moved
        # Moved off an eligible provider because waiting in its queue would
        # cost more than rebuilding the prompt cache elsewhere.
        self.overflows = 0

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
            self._drop(key)
            return None
        return entry[0]

    def bind(
        self,
        key: str,
        provider_id: int,
        previous: Binding | None = None,
        model: str | None = None,
        prompt_tokens: int = 0,
        *,
        overflow: bool = False,
    ) -> None:
        """Record that *key* was routed to *model* on *provider_id* with a
        prompt of about *prompt_tokens*.

        *previous* is what :meth:`get` returned for this request, used only
        for the hit, miss and rebind counters. *overflow* says the session
        left an eligible provider whose queue was too long (counted as an
        overflow, not a rebind).
        """
        now = self._clock()
        if previous is None:
            self.misses += 1
        elif previous.provider_id == provider_id and previous.model in (None, model):
            self.hits += 1
        elif overflow:
            self.overflows += 1
        else:
            self.rebinds += 1
        entries = self._entries
        scope = _scope_of(key)
        scoped = self._per_scope.setdefault(scope, OrderedDict())
        scoped[key] = None
        scoped.move_to_end(key)
        entries[key] = (Binding(provider_id, model, max(0, prompt_tokens)), now)
        entries.move_to_end(key)
        for _ in range(_SWEEP_BATCH):
            first = next(iter(entries), None)
            if first is None or now - entries[first][1] < self.ttl_s:
                break
            self._drop(first)
        held = self._per_scope.get(scope)
        if self.max_per_key and held and len(held) > self.max_per_key:
            # Oldest first, so the scope's least recently used session goes.
            self._drop(next(iter(held)))
        while len(entries) > self.max_sessions:
            self._drop(next(iter(entries)))

    def _drop(self, key: str) -> None:
        if self._entries.pop(key, None) is None:
            return
        scope = _scope_of(key)
        scoped = self._per_scope.get(scope)
        if scoped is not None:
            scoped.pop(key, None)
            if not scoped:
                del self._per_scope[scope]

    def forget(self, key: str | None) -> None:
        if key:
            self._drop(key)

    def forget_providers_not_in(self, provider_ids: set[int]) -> int:
        """Drop bindings to providers that no longer exist. Returns the count."""
        stale = [
            k
            for k, (b, _) in self._entries.items()
            if b.provider_id not in provider_ids
        ]
        for k in stale:
            self._drop(k)
        return len(stale)

    def clear(self) -> None:
        self._entries.clear()
        self._per_scope.clear()

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
            "overflows": self.overflows,
            "ttl_seconds": self.ttl_s,
            "max_sessions": self.max_sessions,
            "max_sessions_per_key": self.max_per_key,
        }
