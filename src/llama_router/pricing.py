"""What a request costs.

Prices are USD per million tokens, per provider and model. They come from
the dashboard (a model name or pattern per provider, ``*`` for all of a
provider's models) or, failing that, from the provider's own model list:
OpenRouter lists a price per token for every model. A local provider with
no price set costs nothing.
"""

from __future__ import annotations

import fnmatch
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from .models import Provider, ProviderModel

# Typical request size used to compare candidates in "cheapest" routing.
NOMINAL_PROMPT_TOKENS = 1000
NOMINAL_COMPLETION_TOKENS = 256
# Estimating a request's size from its body: characters per prompt token (on
# the low side, so the estimate errs high), and the tokens an inline image or
# audio clip (a data: URL, or bare base64 such as Ollama's ``images`` and
# OpenAI's ``input_audio.data``) is taken to cost rather than its length.
_CHARS_PER_TOKEN = 3
_INLINE_MEDIA_TOKENS = 2000
# A string at least this long made only of base64 characters is media.
_BASE64_RE = re.compile(r"[A-Za-z0-9+/=\r\n]{512,}")
# Request fields that bound the output, in the order they are looked for.
_MAX_OUTPUT_FIELDS = ("max_completion_tokens", "max_tokens", "max_output_tokens")


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


def _prompt_tokens(node: Any) -> int:
    if isinstance(node, str):
        if node.startswith("data:") or _BASE64_RE.fullmatch(node):
            return _INLINE_MEDIA_TOKENS
        return -(-len(node) // _CHARS_PER_TOKEN)
    if isinstance(node, Mapping):
        return sum(_prompt_tokens(k) + _prompt_tokens(v) for k, v in node.items())
    if isinstance(node, list):
        return sum(_prompt_tokens(v) for v in node)
    return 1 if node is not None else 0


def _max_output_tokens(body: Mapping[str, Any]) -> int | None:
    for name in _MAX_OUTPUT_FIELDS:
        v = body.get(name)
        if isinstance(v, int) and not isinstance(v, bool) and v > 0:
            return v
    options = body.get("options")  # Ollama
    if isinstance(options, Mapping):
        v = options.get("num_predict")
        if isinstance(v, int) and not isinstance(v, bool) and v > 0:
            return v
    return None


def estimate_tokens(body: Any, default_completion_tokens: int) -> tuple[int, int]:
    """(prompt, completion) tokens a request might use, erring high.

    The prompt is sized from the body's text; the completion is the request's
    own output limit (times ``n`` choices), else *default_completion_tokens*.
    """
    if not isinstance(body, Mapping):
        return 0, default_completion_tokens
    completion = _max_output_tokens(body) or default_completion_tokens
    n = body.get("n")
    if isinstance(n, int) and not isinstance(n, bool) and n > 1:
        completion *= n
    return _prompt_tokens(body), completion


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
