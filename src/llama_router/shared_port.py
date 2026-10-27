"""Serve the LLM APIs and the worker socket on the dashboard port too.

With ``LLAMA_ROUTER_API_ON_DASHBOARD_PORT`` set, the dashboard port (80)
answers Ollama, OpenAI, decision and worker-link requests as well as the UI.
The Ollama and OpenAI ports keep listening either way.

A request goes to the API app only when one of that app's own API routes
matches its path and method in full; everything else, including ``/health``,
stays with the dashboard. The two apps share no paths except ``/decisions``,
which the dashboard serves as a page (GET) and the API as an endpoint (POST),
so matching on the method keeps both working.
"""

from __future__ import annotations

from fastapi import FastAPI
from starlette.routing import BaseRoute, Match
from starlette.types import ASGIApp, Receive, Scope, Send

# Answered by the dashboard, which reports more detail than the API apps.
_DASHBOARD_ONLY = {"/health"}


def api_routes(api_app: FastAPI) -> list[BaseRoute]:
    """The API app's routes, without FastAPI's docs and schema pages."""
    docs = {
        api_app.openapi_url,
        api_app.docs_url,
        api_app.redoc_url,
        api_app.swagger_ui_oauth2_redirect_url,
    }
    return [r for r in api_app.routes if getattr(r, "path", None) not in docs]


def routes_to_api(scope: Scope, routes: list[BaseRoute]) -> bool:
    if scope["type"] not in ("http", "websocket"):
        return False
    if scope["path"] in _DASHBOARD_ONLY:
        return False
    for route in routes:
        match, _ = route.matches(scope)
        if match == Match.FULL:
            return True
    return False


class SharedPortApp:
    """ASGI app for the dashboard port: API requests to ``api``, the rest to ``dashboard``."""

    def __init__(self, dashboard: ASGIApp, api: FastAPI) -> None:
        self.dashboard = dashboard
        self.api = api
        self._routes = api_routes(api)

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if routes_to_api(scope, self._routes):
            await self.api(scope, receive, send)
        else:
            await self.dashboard(scope, receive, send)
