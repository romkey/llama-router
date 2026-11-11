"""Reranking: score documents against a query.

``POST /v1/rerank`` takes the body llama.cpp, vLLM, Jina and Cohere use
(``model``, ``query``, ``documents``, optional ``top_n``) and answers with
each document's relevance score. ``/rerank``, ``/v1/reranking``,
``/reranking`` and ``/v2/rerank`` are the same route. The request goes to
whichever of those routes the routed provider's last probe found, and to
``/v1/rerank`` when its probe found none.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import Response

from ..endpoints import RERANK_PATH
from ..models import Provider
from ..proxy import proxy_json
from ..routing_table import rerank_upstream_path
from ..v1_client import V1Client

router = APIRouter()


def check_rerank_body(body: dict[str, Any]) -> None:
    """Refuse a body with no query or documents before it is routed."""
    query = body.get("query")
    if not isinstance(query, str) or not query:
        raise HTTPException(status_code=400, detail="query is required")
    documents = body.get("documents")
    if not isinstance(documents, list) or not documents:
        raise HTTPException(
            status_code=400, detail="documents must be a non-empty list"
        )


@router.post("/v1/rerank")
@router.post("/rerank")
@router.post("/v1/reranking")
@router.post("/reranking")
@router.post("/v2/rerank")
async def rerank(request: Request) -> Response:
    upstream = {"path": RERANK_PATH}

    def check(provider: Provider, body: dict[str, Any]) -> None:
        check_rerank_body(body)
        upstream["path"] = rerank_upstream_path(provider) or RERANK_PATH

    async def call(client: V1Client, body: dict[str, Any]) -> dict[str, Any]:
        return await client.rerank(body, upstream["path"])

    return await proxy_json(
        request,
        protocol="v1",
        endpoint=request.url.path,
        call=call,
        stream_call=None,
        generation=False,
        check=check,
    )
