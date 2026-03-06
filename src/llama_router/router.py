"""Routes requests to the best available provider for a given model."""

from __future__ import annotations

import logging

from .database import Database
from .models import Provider, ProviderStatus
from .provider_manager import ProviderManager

logger = logging.getLogger(__name__)


class RouteResult:
    """Wraps a routing decision, including which model was actually resolved."""

    __slots__ = ("provider", "resolved_model")

    def __init__(self, provider: Provider, resolved_model: str):
        self.provider = provider
        self.resolved_model = resolved_model


class Router:
    def __init__(self, db: Database, provider_manager: ProviderManager):
        self._db = db
        self._pm = provider_manager

    async def route(
        self, model_name: str, protocol: str | None = None
    ) -> RouteResult | None:
        """Pick the best provider for the requested model, following fallbacks.

        Walks the fallback chain if the requested model has no available provider.
        Returns a RouteResult containing the chosen provider and the model name
        that was actually resolved (may differ from the original if a fallback
        was used).
        """
        chain = await self._db.resolve_fallback_chain(model_name)
        for candidate_model in chain:
            result = await self._route_single(candidate_model, protocol)
            if result is not None:
                if candidate_model != model_name:
                    logger.info(
                        "Model %s unavailable; fell back to %s",
                        model_name,
                        candidate_model,
                    )
                return RouteResult(result, candidate_model)
        return None

    async def _route_single(
        self, model_name: str, protocol: str | None = None
    ) -> Provider | None:
        """Pick the best provider for a single model (no fallbacks)."""
        candidates = await self._db.get_providers_for_model(model_name, protocol)
        if not candidates:
            return None

        online = [c for c in candidates if c.status != ProviderStatus.OFFLINE]
        if not online:
            return None

        scored: list[tuple[float, Provider]] = []
        for provider in online:
            assert provider.id is not None
            active = self._pm.active_requests(provider.id)
            bench = await self._db.get_latest_benchmark(provider.id, model_name)
            tps = bench.tokens_per_second if bench and bench.tokens_per_second else 0
            score = active - (tps / 10000)
            scored.append((score, provider))

        scored.sort(key=lambda x: x[0])
        chosen = scored[0][1]
        logger.info(
            "Routing model %s (%s) to provider %s (active=%d)",
            model_name,
            protocol or "any",
            chosen.name,
            self._pm.active_requests(chosen.id),  # type: ignore[arg-type]
        )
        return chosen
