from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.responses import Response

from ..proxy import proxy_json

router = APIRouter()


@router.post("/api/embeddings")
async def embeddings(request: Request) -> Response:
    return await proxy_json(
        request,
        protocol="ollama",
        endpoint="/api/embeddings",
        call=lambda client, body: client.embeddings(body),
        stream_call=None,
        generation=False,
    )


@router.post("/api/embed")
async def embed(request: Request) -> Response:
    return await proxy_json(
        request,
        protocol="ollama",
        endpoint="/api/embed",
        call=lambda client, body: client.embed(body),
        stream_call=None,
        generation=False,
    )
