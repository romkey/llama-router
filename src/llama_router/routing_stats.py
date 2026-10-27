"""In-memory performance samples that feed the routing table.

Everything here is cheap to update from the request path: a few float
operations and a dict lookup. Samples are never persisted; the routing table
falls back to benchmarks until live samples arrive.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field

# Weight of a new sample in the moving averages.
_ALPHA = 0.3
# A failure burst halves its weight every this many seconds without traffic,
# so a provider that was failing gets retried even if nothing is sent to it.
FAILURE_HALF_LIFE_S = 120.0
# A model that served a request this recently is assumed to still be loaded.
# Ollama's default keep_alive is five minutes; stay a little inside it.
WARM_WINDOW_S = 240.0


@dataclass(slots=True)
class Ewma:
    value: float | None = None
    samples: int = 0

    def add(self, x: float) -> None:
        if self.value is None:
            self.value = x
        else:
            self.value += _ALPHA * (x - self.value)
        self.samples += 1


@dataclass(slots=True)
class DecayingRate:
    """Recent failure fraction (0..1) that decays toward 0 over time."""

    value: float = 0.0
    at: float = 0.0
    samples: int = 0

    def current(self, now: float) -> float:
        if self.value <= 0.0:
            return 0.0
        return self.value * 0.5 ** (max(0.0, now - self.at) / FAILURE_HALF_LIFE_S)

    def add(self, failed: bool, now: float) -> None:
        v = self.current(now)
        self.value = v + _ALPHA * ((1.0 if failed else 0.0) - v)
        self.at = now
        self.samples += 1


@dataclass(slots=True)
class ModelStats:
    ttft_ms: Ewma = field(default_factory=Ewma)
    tokens_per_second: Ewma = field(default_factory=Ewma)
    failures: DecayingRate = field(default_factory=DecayingRate)
    last_ok: float = 0.0


class ProviderStats:
    """Moving averages of address round trips and per-model request outcomes."""

    def __init__(self, clock: Callable[[], float] = time.monotonic) -> None:
        self._clock = clock
        self._rtt: dict[int, dict[int, Ewma]] = {}
        self._models: dict[tuple[int, str], ModelStats] = {}
        self._listeners: list[Callable[[str], None]] = []

    def now(self) -> float:
        return self._clock()

    def add_listener(self, fn: Callable[[str], None]) -> None:
        self._listeners.append(fn)

    def _changed(self, reason: str) -> None:
        for fn in self._listeners:
            fn(reason)

    # --- Address round trips (health probes) ---

    def record_rtt(self, provider_id: int, address_id: int, ms: float) -> None:
        self._rtt.setdefault(provider_id, {}).setdefault(address_id, Ewma()).add(ms)
        self._changed("rtt")

    def rtt_ms(
        self, provider_id: int, address_ids: list[int] | None = None
    ) -> float | None:
        """Best (lowest) round trip among *address_ids*, or among all addresses."""
        per_addr = self._rtt.get(provider_id)
        if not per_addr:
            return None
        values = [
            e.value
            for aid, e in per_addr.items()
            if e.value is not None and (address_ids is None or aid in address_ids)
        ]
        return min(values) if values else None

    def address_rtt_ms(self, provider_id: int, address_id: int) -> float | None:
        e = self._rtt.get(provider_id, {}).get(address_id)
        return e.value if e else None

    # --- Request outcomes ---

    def record_request(
        self,
        provider_id: int,
        model: str,
        *,
        ok: bool,
        ttft_ms: float | None = None,
        tokens_per_second: float | None = None,
    ) -> None:
        s = self._models.get((provider_id, model))
        if s is None:
            s = self._models[(provider_id, model)] = ModelStats()
        now = self._clock()
        s.failures.add(not ok, now)
        if ok:
            s.last_ok = now
            if ttft_ms is not None and ttft_ms >= 0:
                s.ttft_ms.add(ttft_ms)
            if tokens_per_second is not None and tokens_per_second > 0:
                s.tokens_per_second.add(tokens_per_second)
        self._changed("request")

    def model(self, provider_id: int, model: str) -> ModelStats | None:
        return self._models.get((provider_id, model))

    def is_warm(self, provider_id: int, model: str, now: float | None = None) -> bool:
        s = self._models.get((provider_id, model))
        if s is None or not s.last_ok:
            return False
        t = self._clock() if now is None else now
        return t - s.last_ok < WARM_WINDOW_S

    def forget_provider(self, provider_id: int) -> None:
        self._rtt.pop(provider_id, None)
        for key in [k for k in self._models if k[0] == provider_id]:
            del self._models[key]
