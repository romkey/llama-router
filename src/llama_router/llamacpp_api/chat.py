from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.responses import Response

from ..proxy import proxy_json

router = APIRouter()


@router.post("/v1/chat/completions")
@router.post("/chat/completions")
async def chat_completions(request: Request) -> Response:
    return await proxy_json(
        request,
        protocol="v1",
        endpoint=request.url.path,
        call=lambda client, body: client.chat_completions(body),
        stream_call=lambda client, body: client.chat_completions_stream(body),
        stream_usage=True,
    )
