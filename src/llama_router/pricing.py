"""What a request costs.

Prices are USD per million tokens, per provider and model. They come from
the dashboard (a model name or pattern per provider, ``*`` for all of a
provider's models) or, failing that, from the provider's own model list:
OpenRouter lists a price per token for every model. A local provider with
no price set costs nothing.
"""

from __future__ import annotations

import fnmatch
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from .models import Provider, ProviderModel

# Typical request size used to compare candidates in "cheapest" routing.
NOMINAL_PROMPT_TOKENS = 1000
NOMINAL_COMPLETION_TOKENS = 256


@dataclass(frozen=True, slots=True)
class Price:
    input_per_mtok: float
    output_per_mtok: float
    source: str = "manual"  # "manual", "listed" or "free"

    def cost(self, prompt_tokens: int | None, completion_tokens: int | None) -> float:
        return (
            (prompt_tokens or 0) * self.input_per_mtok
            + (completion_tokens or 0) * self.output_per_mtok
        ) / 1_000_000

    def nominal(self) -> float:
        return self.cost(NOMINAL_PROMPT_TOKENS, NOMINAL_COMPLETION_TOKENS)


FREE = Price(0.0, 0.0, "free")


def listed_price(model: ProviderModel) -> Price | None:
    """Price from the provider's model list (OpenRouter: USD per token)."""
    pricing = (model.details or {}).get("pricing")
    if not isinstance(pricing, Mapping):
        return None
    try:
        raw_prompt = pricing.get("prompt")
        raw_completion = pricing.get("completion")
        if raw_prompt is None or raw_completion is None:
            return None
        prompt = float(raw_prompt)
        completion = float(raw_completion)
    except TypeError, ValueError:
        return None
    if prompt < 0 or completion < 0:  # OpenRouter uses -1 for "varies"
        return None
    return Price(prompt * 1_000_000, completion * 1_000_000, "listed")


class PriceBook:
    """Prices for every (provider, model), resolved once per routing build."""

    def __init__(
        self,
        providers: Iterable[Provider],
        models: Mapping[int, list[ProviderModel]],
        rows: Iterable[Mapping[str, Any]],
    ) -> None:
        self._cloud = {p.id: p.is_cloud for p in providers if p.id is not None}
        self._manual: dict[int, list[tuple[str, Price]]] = {}
        for r in rows:
            self._manual.setdefault(int(r["provider_id"]), []).append(
                (
                    str(r["model_pattern"]),
                    Price(float(r["input_per_mtok"]), float(r["output_per_mtok"])),
                )
            )
        for entries in self._manual.values():
            # Exact names before patterns, and "*" last.
            entries.sort(key=lambda e: (e[0] == "*", any(c in e[0] for c in "*?[")))
        self._listed: dict[tuple[int, str], Price] = {}
        for pid, ms in models.items():
            for m in ms:
                price = listed_price(m)
                if price is not None:
                    self._listed[(pid, m.name)] = price
                    if m.raw_name:
                        self._listed[(pid, m.raw_name)] = price

    def price(self, provider_id: int, model: str) -> Price | None:
        """The price of *model* on the provider; None when unknown."""
        for pattern, price in self._manual.get(provider_id, ()):
            if pattern == model or fnmatch.fnmatchcase(model, pattern):
                return price
        listed = self._listed.get((provider_id, model))
        if listed is not None:
            return listed
        if not self._cloud.get(provider_id, False):
            return FREE
        return None

    def cost(
        self,
        provider_id: int,
        model: str,
        prompt_tokens: int | None,
        completion_tokens: int | None,
    ) -> float | None:
        if prompt_tokens is None and completion_tokens is None:
            return None
        price = self.price(provider_id, model)
        return None if price is None else price.cost(prompt_tokens, completion_tokens)
