"""Serve the LLM APIs and the worker socket on the dashboard port too.

With ``LLAMA_ROUTER_API_ON_DASHBOARD_PORT`` set, the dashboard port (80)
answers Ollama, OpenAI, Anthropic, decision and worker-link requests as well
as the UI. The Ollama and OpenAI ports keep listening either way.

A request goes to the first API app with a route that matches its path and
method in full: the Ollama app (which also serves most ``/v1`` routes), then
the OpenAI app, which alone has reranking and ``/v1/messages``. Any other
path under ``/v1/`` or ``/v2/`` goes to the OpenAI app too, so a client
gets a JSON 404 or 405 (logged when it carries an API key) rather than a
redirect to the dashboard login. Everything else, including ``/health``,
stays with the dashboard. The apps share no paths except ``/decisions``,
which the dashboard serves as a page (GET) and the APIs as an endpoint
(POST), so matching on the method keeps both working.
"""

from __future__ import annotations

from fastapi import FastAPI
from starlette.routing import BaseRoute, Match
from starlette.types import ASGIApp, Receive, Scope, Send

# Answered by the dashboard, which reports more detail than the API apps.
_DASHBOARD_ONLY = {"/health"}

# Paths only an API serves: one no route matches still goes to an API app.
_API_PREFIXES = ("/v1/", "/v2/")


def api_routes(api_app: FastAPI) -> list[BaseRoute]:
    """The API app's routes, without FastAPI's docs and schema pages."""
    # A page turned off (None) must not match routes that have no path.
    docs = {
        url
        for url in (
            api_app.openapi_url,
            api_app.docs_url,
            api_app.redoc_url,
            api_app.swagger_ui_oauth2_redirect_url,
        )
        if url is not None
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


def api_only_path(scope: Scope) -> bool:
    """A path under an API prefix, which the dashboard never serves."""
    return scope["type"] == "http" and str(scope["path"]).startswith(_API_PREFIXES)


class SharedPortApp:
    """ASGI app for the dashboard port: API requests to the API apps, the
    rest to the dashboard.

    *apis* are tried in order; the last one also answers ``/v1/`` and
    ``/v2/`` paths none of them has a route for.
    """

    def __init__(self, dashboard: ASGIApp, *apis: FastAPI) -> None:
        if not apis:
            raise ValueError("SharedPortApp needs at least one API app")
        self.dashboard = dashboard
        self._apis = [(api, api_routes(api)) for api in apis]

    def app_for(self, scope: Scope) -> ASGIApp:
        for api, routes in self._apis:
            if routes_to_api(scope, routes):
                return api
        if api_only_path(scope):
            return self._apis[-1][0]
        return self.dashboard

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        await self.app_for(scope)(scope, receive, send)
