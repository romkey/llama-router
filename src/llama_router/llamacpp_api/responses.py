from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.responses import Response

from ..proxy import proxy_json

router = APIRouter()


@router.post("/v1/responses")
async def responses(request: Request) -> Response:
    return await proxy_json(
        request,
        protocol="v1",
        endpoint="/v1/responses",
        call=lambda client, body: client.responses(body),
        stream_call=lambda client, body: client.responses_stream(body),
    )
