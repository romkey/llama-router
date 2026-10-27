"""In-memory performance samples that feed the routing table.

Everything here is cheap to update from the request path: a few float
operations and a dict lookup. Samples are never persisted; the routing table
falls back to benchmarks until live samples arrive.

Three kinds of samples are kept:

- per address: a running tally of health-check round trips and failures;
- per provider: a health score from recent request outcomes and probes;
- per (provider, model, endpoint class): time to first token, generation
  speed, service time and failure rate.
"""

from __future__ import annotations

import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from .endpoints import GENERATE

# Weight of a new sample in the moving averages.
_ALPHA = 0.3
# A failure burst halves its weight every this many seconds without traffic,
# so a provider that was failing gets retried even if nothing is sent to it.
FAILURE_HALF_LIFE_S = 120.0
# A model that served a request this recently is assumed to still be loaded.
# Ollama's default keep_alive is five minutes; stay a little inside it.
WARM_WINDOW_S = 240.0
# Health-check samples kept per address (an hour at the default interval).
TALLY_WINDOW = 120
# A provider failing at least this share of recent requests (with enough of
# them) is skipped while another provider can serve the request.
PROVIDER_CIRCUIT_RATE = 0.5
PROVIDER_CIRCUIT_MIN_SAMPLES = 5
# A provider that answered a decision question without logprobs is not
# asked again for this long; then it is retried in case it was upgraded.
NO_LOGPROBS_TTL_S = 3600.0


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
class LatencyTally:
    """Running tally of one address's health-check round trips."""

    ewma: Ewma = field(default_factory=Ewma)
    recent: deque = field(default_factory=lambda: deque(maxlen=TALLY_WINDOW))
    checks: int = 0
    failures: int = 0
    consecutive_failures: int = 0
    last_ms: float | None = None
    last_ok_at: float | None = None
    last_failed_at: float | None = None
    min_ms: float | None = None
    max_ms: float | None = None
    total_ms: float = 0.0

    def add_ok(self, ms: float, now: float) -> None:
        self.ewma.add(ms)
        self.recent.append(ms)
        self.checks += 1
        self.consecutive_failures = 0
        self.last_ms = ms
        self.last_ok_at = now
        self.total_ms += ms
        self.min_ms = ms if self.min_ms is None else min(self.min_ms, ms)
        self.max_ms = ms if self.max_ms is None else max(self.max_ms, ms)

    def add_failure(self, now: float) -> None:
        self.checks += 1
        self.failures += 1
        self.consecutive_failures += 1
        self.last_failed_at = now

    def percentile(self, p: float) -> float | None:
        if not self.recent:
            return None
        ordered = sorted(self.recent)
        idx = min(len(ordered) - 1, max(0, round(p / 100 * (len(ordered) - 1))))
        return ordered[idx]

    def to_dict(self) -> dict[str, Any]:
        ok = self.checks - self.failures
        return {
            "checks": self.checks,
            "failures": self.failures,
            "consecutive_failures": self.consecutive_failures,
            "ewma_ms": _r(self.ewma.value),
            "last_ms": _r(self.last_ms),
            "p50_ms": _r(self.percentile(50)),
            "p95_ms": _r(self.percentile(95)),
            "min_ms": _r(self.min_ms),
            "max_ms": _r(self.max_ms),
            "mean_ms": _r(self.total_ms / ok) if ok else None,
            "window": len(self.recent),
        }


@dataclass(slots=True)
class ModelStats:
    ttft_ms: Ewma = field(default_factory=Ewma)
    tokens_per_second: Ewma = field(default_factory=Ewma)
    # Dispatch to last byte for successful requests.
    duration_ms: Ewma = field(default_factory=Ewma)
    failures: DecayingRate = field(default_factory=DecayingRate)
    last_ok: float = 0.0


@dataclass(slots=True, frozen=True)
class ProviderHealth:
    """How a provider has behaved recently, across all its models."""

    request_failure_rate: float
    probe_failure_rate: float
    request_samples: int
    circuit_open: bool

    @property
    def failure_rate(self) -> float:
        """Chance a request fails for provider-wide reasons (0..1)."""
        return 1.0 - (1.0 - self.request_failure_rate) * (1.0 - self.probe_failure_rate)

    @property
    def score(self) -> int:
        """0 (failing everything) to 100 (healthy)."""
        return round((1.0 - self.failure_rate) * 100)

    def to_dict(self) -> dict[str, Any]:
        return {
            "score": self.score,
            "request_failure_rate": round(self.request_failure_rate, 4),
            "probe_failure_rate": round(self.probe_failure_rate, 4),
            "request_samples": self.request_samples,
            "circuit_open": self.circuit_open,
        }


HEALTHY = ProviderHealth(0.0, 0.0, 0, False)


# Cooldown after a 429 without a usable Retry-After, and the longest one.
DEFAULT_BUSY_S = 10.0
MAX_BUSY_S = 300.0


def retry_after_seconds(value: str | None) -> float:
    """Seconds from a Retry-After header (delta or HTTP date), else the default."""
    raw = (value or "").strip()
    if not raw:
        return DEFAULT_BUSY_S
    try:
        return float(raw)
    except ValueError:
        pass
    try:
        from email.utils import parsedate_to_datetime

        when = parsedate_to_datetime(raw)
    except (TypeError, ValueError):
        return DEFAULT_BUSY_S
    import datetime as _dt

    if when.tzinfo is None:
        when = when.replace(tzinfo=_dt.timezone.utc)
    return max((when - _dt.datetime.now(_dt.timezone.utc)).total_seconds(), 1.0)


class ProviderStats:
    """Health-check tallies, provider health and per-model request outcomes."""

    def __init__(self, clock: Callable[[], float] = time.monotonic) -> None:
        self._clock = clock
        self._tallies: dict[int, dict[int, LatencyTally]] = {}
        self._models: dict[tuple[int, str, str], ModelStats] = {}
        self._requests: dict[int, DecayingRate] = {}
        self._probes: dict[int, DecayingRate] = {}
        self._last_ok: dict[tuple[int, str], float] = {}
        self._listeners: list[Callable[[str], None]] = []
        # Until when each provider asked for no more requests (HTTP 429).
        self._busy_until: dict[int, float] = {}
        # When each provider last returned a chat completion with no logprobs.
        self._no_logprobs: dict[int, float] = {}

    def now(self) -> float:
        return self._clock()

    def add_listener(self, fn: Callable[[str], None]) -> None:
        self._listeners.append(fn)

    def _changed(self, reason: str) -> None:
        for fn in self._listeners:
            fn(reason)

    # --- Health checks ---

    def record_probe(
        self, provider_id: int, address_id: int, *, ok: bool, ms: float | None = None
    ) -> None:
        """One health check of one address: its round trip, or a failure."""
        now = self._clock()
        tally = self._tallies.setdefault(provider_id, {}).setdefault(
            address_id, LatencyTally()
        )
        if ok and ms is not None:
            tally.add_ok(ms, now)
        else:
            tally.add_failure(now)
        self._changed("health check")

    def record_check(self, provider_id: int, *, ok: bool) -> None:
        """One whole health check of a provider: did any address answer?

        Feeds the provider's health score, so a provider that keeps dropping
        out is penalised for a while after it comes back. One dead address
        among live ones does not count against it.
        """
        self._probes.setdefault(provider_id, DecayingRate()).add(not ok, self._clock())
        self._changed("health check")

    def record_rtt(self, provider_id: int, address_id: int, ms: float) -> None:
        self.record_probe(provider_id, address_id, ok=True, ms=ms)

    def rtt_ms(
        self, provider_id: int, address_ids: list[int] | None = None
    ) -> float | None:
        """Best (lowest) moving-average round trip among *address_ids*."""
        per_addr = self._tallies.get(provider_id)
        if not per_addr:
            return None
        values = [
            t.ewma.value
            for aid, t in per_addr.items()
            if t.ewma.value is not None and (address_ids is None or aid in address_ids)
        ]
        return min(values) if values else None

    def tally(self, provider_id: int, address_id: int) -> LatencyTally | None:
        return self._tallies.get(provider_id, {}).get(address_id)

    def address_rtt_ms(self, provider_id: int, address_id: int) -> float | None:
        t = self.tally(provider_id, address_id)
        return t.ewma.value if t else None

    # --- Request outcomes ---

    def record_request(
        self,
        provider_id: int,
        model: str,
        *,
        ok: bool,
        endpoint_class: str = GENERATE,
        ttft_ms: float | None = None,
        tokens_per_second: float | None = None,
        duration_ms: float | None = None,
    ) -> None:
        key = (provider_id, model, endpoint_class)
        s = self._models.get(key)
        if s is None:
            s = self._models[key] = ModelStats()
        now = self._clock()
        s.failures.add(not ok, now)
        self._requests.setdefault(provider_id, DecayingRate()).add(not ok, now)
        if ok:
            s.last_ok = now
            self._last_ok[(provider_id, model)] = now
            if ttft_ms is not None and ttft_ms >= 0:
                s.ttft_ms.add(ttft_ms)
            if tokens_per_second is not None and tokens_per_second > 0:
                s.tokens_per_second.add(tokens_per_second)
            if duration_ms is not None and duration_ms >= 0:
                s.duration_ms.add(duration_ms)
        self._changed("request")

    def model(
        self, provider_id: int, model: str, endpoint_class: str = GENERATE
    ) -> ModelStats | None:
        return self._models.get((provider_id, model, endpoint_class))

    def is_warm(self, provider_id: int, model: str, now: float | None = None) -> bool:
        """The model served any request recently, so it is likely loaded."""
        last = self._last_ok.get((provider_id, model))
        if not last:
            return False
        t = self._clock() if now is None else now
        return t - last < WARM_WINDOW_S

    def provider_health(
        self, provider_id: int, now: float | None = None
    ) -> ProviderHealth:
        t = self._clock() if now is None else now
        req = self._requests.get(provider_id)
        probe = self._probes.get(provider_id)
        if req is None and probe is None:
            return HEALTHY
        req_rate = req.current(t) if req else 0.0
        samples = req.samples if req else 0
        return ProviderHealth(
            request_failure_rate=req_rate,
            probe_failure_rate=probe.current(t) if probe else 0.0,
            request_samples=samples,
            circuit_open=samples >= PROVIDER_CIRCUIT_MIN_SAMPLES
            and req_rate >= PROVIDER_CIRCUIT_RATE,
        )

    # --- Rate limits ---

    def mark_busy(self, provider_id: int, seconds: float) -> None:
        """The provider answered 429: route around it for *seconds*.

        Not a failure: the provider is fine, just full (a cloud rate limit,
        or llama.cpp with every slot taken).
        """
        seconds = min(max(seconds, 1.0), MAX_BUSY_S)
        until = self._clock() + seconds
        if until > self._busy_until.get(provider_id, 0.0):
            self._busy_until[provider_id] = until

    def busy_for(self, provider_id: int, now: float | None = None) -> float:
        """Seconds left of the provider's rate-limit cooldown (0 when none)."""
        until = self._busy_until.get(provider_id)
        if until is None:
            return 0.0
        left = until - (self._clock() if now is None else now)
        if left <= 0:
            self._busy_until.pop(provider_id, None)
            return 0.0
        return left

    # --- Decision support ---

    def record_logprobs(self, provider_id: int, *, ok: bool) -> None:
        """Whether *provider_id* returned logprobs for a decision question."""
        if ok:
            if self._no_logprobs.pop(provider_id, None) is not None:
                self._changed("logprobs")
            return
        first = provider_id not in self._no_logprobs
        self._no_logprobs[provider_id] = self._clock()
        if first:
            self._changed("no logprobs")

    def lacks_logprobs(self, provider_id: int, now: float | None = None) -> bool:
        """Recently answered without logprobs, so it cannot emulate decisions."""
        seen = self._no_logprobs.get(provider_id)
        if seen is None:
            return False
        now = self._clock() if now is None else now
        return now - seen < NO_LOGPROBS_TTL_S

    def forget_provider(self, provider_id: int) -> None:
        self._busy_until.pop(provider_id, None)
        self._tallies.pop(provider_id, None)
        self._requests.pop(provider_id, None)
        self._probes.pop(provider_id, None)
        self._no_logprobs.pop(provider_id, None)
        for key in [k for k in self._models if k[0] == provider_id]:
            del self._models[key]
        for key in [k for k in self._last_ok if k[0] == provider_id]:
            del self._last_ok[key]


def _r(v: float | None) -> float | None:
    return None if v is None else round(v, 2)
