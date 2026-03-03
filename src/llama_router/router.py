"""Routes requests to the best available provider for a given model."""

from __future__ import annotations

import logging

from .database import Database
from .models import Provider, ProviderStatus
from .provider_manager import ProviderManager

logger = logging.getLogger(__name__)


class Router:
    def __init__(self, db: Database, provider_manager: ProviderManager):
        self._db = db
        self._pm = provider_manager

    async def route(self, model_name: str) -> Provider | None:
        """Pick the best provider for the requested model.

        Strategy: least-busy first, then highest tokens/sec as tiebreaker.
        """
        candidates = await self._db.get_providers_for_model(model_name)
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
            # Lower score = better: active requests dominate, fast providers break ties
            score = active - (tps / 10000)
            scored.append((score, provider))

        scored.sort(key=lambda x: x[0])
        chosen = scored[0][1]
        logger.info(
            "Routing model %s to provider %s (active=%d)",
            model_name,
            chosen.name,
            self._pm.active_requests(chosen.id),  # type: ignore[arg-type]
        )
        return chosen
