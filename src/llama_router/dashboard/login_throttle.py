"""Back off password sign-in after repeated failures.

Failures are counted per client address and per username, in memory (one
router process; a restart forgets them). After ``FREE_ATTEMPTS`` failures in
a row the key is locked for a delay that doubles with each further failure,
up to ``MAX_DELAY_S``. A success clears both keys; a key with no failure for
``FORGET_AFTER_S`` is forgotten.

The per-username lock lets anyone delay one account's sign-in by guessing
wrong; the cap keeps that to minutes, and single sign-on is unaffected.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

FREE_ATTEMPTS = 5
BASE_DELAY_S = 1.0
MAX_DELAY_S = 300.0
FORGET_AFTER_S = 3600.0
# Most keys remembered; beyond this the stalest are dropped, so a flood of
# made-up usernames or addresses cannot grow memory without bound.
MAX_KEYS = 10_000
# 2**20 seconds is far beyond MAX_DELAY_S already.
_MAX_DOUBLINGS = 20


@dataclass
class _Entry:
    failures: int
    last_failure: float
    locked_until: float


class LoginThrottle:
    def __init__(self) -> None:
        self._entries: dict[str, _Entry] = {}

    def retry_after(self, keys: list[str], now: float | None = None) -> float:
        """Seconds until any of *keys* may try again; 0 when none is locked."""
        t = time.monotonic() if now is None else now
        wait = 0.0
        for key in keys:
            e = self._entries.get(key)
            if e is not None and e.locked_until > t:
                wait = max(wait, e.locked_until - t)
        return wait

    def failure(self, keys: list[str], now: float | None = None) -> None:
        t = time.monotonic() if now is None else now
        for key in keys:
            e = self._entries.get(key)
            if e is None or t - e.last_failure > FORGET_AFTER_S:
                e = _Entry(failures=0, last_failure=t, locked_until=0.0)
                self._entries[key] = e
            e.failures += 1
            e.last_failure = t
            over = e.failures - FREE_ATTEMPTS
            if over >= 0:
                # Capped exponent: 2**1024 overflows a float.
                delay = BASE_DELAY_S * 2 ** min(over, _MAX_DOUBLINGS)
                e.locked_until = t + min(MAX_DELAY_S, delay)
        if len(self._entries) > MAX_KEYS:
            self._prune(t)

    def success(self, keys: list[str]) -> None:
        for key in keys:
            self._entries.pop(key, None)

    def clear(self) -> None:
        self._entries.clear()

    def _prune(self, now: float) -> None:
        for key in [
            k for k, e in self._entries.items() if now - e.last_failure > FORGET_AFTER_S
        ]:
            del self._entries[key]
        excess = len(self._entries) - MAX_KEYS
        if excess > 0:
            stalest = sorted(self._entries, key=lambda k: self._entries[k].last_failure)
            for key in stalest[:excess]:
                del self._entries[key]


def keys_for(client_ip: str, username: str) -> list[str]:
    return [f"ip:{client_ip}", f"user:{username}"]


throttle = LoginThrottle()
