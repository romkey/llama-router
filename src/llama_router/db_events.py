"""Change reasons and the decorator that announces them after a write."""

from __future__ import annotations

import functools
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Concatenate

if TYPE_CHECKING:
    from .repositories.base import Repository

# Reasons passed to change listeners. The router rebuilds its table on the
# routing ones; the auth cache drops entries on the auth ones.
CHANGE_PROVIDERS = "providers"
CHANGE_STATUS = "status"
CHANGE_ADDRESSES = "addresses"
CHANGE_MODELS = "models"
CHANGE_BENCHMARKS = "benchmarks"
CHANGE_FALLBACKS = "fallbacks"
CHANGE_ALIASES = "aliases"
CHANGE_GROUPS = "groups"
CHANGE_PRICES = "prices"
CHANGE_CAPABILITIES = "capabilities"
CHANGE_API_KEYS = "api_keys"
CHANGE_AUTH_SETTINGS = "auth_settings"
CHANGE_DECISIONS = "decision_calibrations"
ROUTING_CHANGES = frozenset(
    {
        CHANGE_PROVIDERS,
        CHANGE_STATUS,
        CHANGE_ADDRESSES,
        CHANGE_MODELS,
        CHANGE_BENCHMARKS,
        CHANGE_FALLBACKS,
        CHANGE_ALIASES,
        CHANGE_GROUPS,
        CHANGE_PRICES,
        CHANGE_CAPABILITIES,
    }
)
AUTH_CHANGES = frozenset({CHANGE_API_KEYS, CHANGE_AUTH_SETTINGS})


def _notifies[S: Repository, **P, R](
    reason: str,
) -> Callable[
    [Callable[Concatenate[S, P], Awaitable[R]]],
    Callable[Concatenate[S, P], Awaitable[R]],
]:
    """Tell change listeners about *reason* after the wrapped write succeeds."""

    def wrap(
        fn: Callable[Concatenate[S, P], Awaitable[R]],
    ) -> Callable[Concatenate[S, P], Awaitable[R]]:
        async def inner(self: S, /, *args: P.args, **kwargs: P.kwargs) -> R:
            result = await fn(self, *args, **kwargs)
            self._changed(reason)
            return result

        functools.update_wrapper(inner, fn)
        return inner

    return wrap
