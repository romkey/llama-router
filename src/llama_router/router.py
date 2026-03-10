"""Routes requests to the best available provider for a given model."""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass

from .database import Database
from .models import Provider, ProviderStatus, ProviderType
from .provider_manager import ProviderManager

logger = logging.getLogger(__name__)


class RouteResult:
    """Wraps a routing decision, including which model was actually resolved."""

    __slots__ = ("provider", "resolved_model")

    def __init__(self, provider: Provider, resolved_model: str):
        self.provider = provider
        self.resolved_model = resolved_model


@dataclass(slots=True)
class RoutingPreferences:
    mode: str = "latency"  # "latency", "throughput", or "chaos"
    allow_fallback: bool = True


class Router:
    def __init__(self, db: Database, provider_manager: ProviderManager):
        self._db = db
        self._pm = provider_manager

    async def route(
        self,
        model_name: str,
        protocol: str | None = None,
        preferences: RoutingPreferences | None = None,
    ) -> RouteResult | None:
        """Pick the best provider for the requested model, following fallbacks.

        Walks the fallback chain if the requested model has no available provider.
        Returns a RouteResult containing the chosen provider and the model name
        that was actually resolved (may differ from the original if a fallback
        was used).
        """
        if preferences and preferences.mode == "chaos":
            return await self._route_chaos(model_name, protocol)

        if preferences and not preferences.allow_fallback:
            chain = [model_name]
        else:
            chain = await self._db.resolve_fallback_chain(model_name)
        for candidate_model in chain:
            result = await self._route_single(
                candidate_model, protocol, preferences=preferences
            )
            if result is not None:
                if candidate_model != model_name:
                    logger.info(
                        "Model %s unavailable; fell back to %s",
                        model_name,
                        candidate_model,
                    )
                return RouteResult(result, candidate_model)
        return None

    async def _route_chaos(
        self, requested_model: str, protocol: str | None = None
    ) -> RouteResult | None:
        """Chaos mode: pick a random online provider and random eligible model."""
        providers = await self._db.list_providers()
        online = [p for p in providers if p.status != ProviderStatus.OFFLINE]
        if protocol == "ollama":
            online = [
                p
                for p in online
                if p.provider_type in (ProviderType.OLLAMA, ProviderType.BOTH)
            ]
        elif protocol == "llamacpp":
            online = [
                p
                for p in online
                if p.provider_type in (ProviderType.LLAMACPP, ProviderType.BOTH)
            ]
        if not online:
            return None

        candidates: list[tuple[Provider, list[str]]] = []
        for p in online:
            if p.id is None:
                continue
            models = await self._db.get_provider_models(p.id)
            eligible: list[str] = []
            for m in models:
                details = m.details or {}
                if protocol == "ollama":
                    if "_in_ollama" in details:
                        if details.get("_in_ollama"):
                            eligible.append(m.name)
                    elif p.provider_type in (ProviderType.OLLAMA, ProviderType.BOTH):
                        eligible.append(m.name)
                elif protocol == "llamacpp":
                    if "_in_llamacpp" in details:
                        if details.get("_in_llamacpp"):
                            eligible.append(m.name)
                    elif p.provider_type in (ProviderType.LLAMACPP, ProviderType.BOTH):
                        eligible.append(m.name)
                else:
                    eligible.append(m.name)
            if eligible:
                candidates.append((p, eligible))

        if not candidates:
            return None

        provider, eligible = random.choice(candidates)
        resolved_model = random.choice(eligible)
        logger.info(
            "Chaos routing requested=%s protocol=%s -> provider=%s model=%s",
            requested_model,
            protocol or "any",
            provider.name,
            resolved_model,
        )
        return RouteResult(provider, resolved_model)

    async def _route_single(
        self,
        model_name: str,
        protocol: str | None = None,
        preferences: RoutingPreferences | None = None,
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
            bench = await self._db.get_latest_benchmark(
                provider.id, model_name, protocol=protocol
            )
            tps = bench.tokens_per_second if bench and bench.tokens_per_second else 0
            startup_ms = (
                bench.startup_time_ms if bench and bench.startup_time_ms else None
            )
            mode = preferences.mode if preferences else "latency"
            if mode == "throughput":
                # Prefer highest TPS, then lower active queue depth.
                score = (active * 100.0) - tps
            else:
                # Prefer low queue depth and low startup latency.
                startup_component = (
                    startup_ms if startup_ms is not None else 1_000_000.0
                )
                score = (active * 1000.0) + startup_component
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
