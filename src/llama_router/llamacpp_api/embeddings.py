from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.responses import Response

from ..proxy import proxy_json

router = APIRouter()


@router.post("/v1/embeddings")
async def embeddings(request: Request) -> Response:
    return await proxy_json(
        request,
        protocol="v1",
        endpoint="/v1/embeddings",
        call=lambda client, body: client.embeddings(body),
        stream_call=None,
        generation=False,
    )
