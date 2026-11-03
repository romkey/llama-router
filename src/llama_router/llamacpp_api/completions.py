from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.responses import Response

from ..proxy import proxy_json

router = APIRouter()


@router.post("/v1/completions")
async def completions(request: Request) -> Response:
    return await proxy_json(
        request,
        protocol="v1",
        endpoint="/v1/completions",
        call=lambda client, body: client.completions(body),
        stream_call=lambda client, body: client.completions_stream(body),
        stream_usage=True,
    )
