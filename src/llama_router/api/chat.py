from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.responses import Response

from ..proxy import proxy_json

router = APIRouter()


@router.post("/api/chat")
async def chat(request: Request) -> Response:
    return await proxy_json(
        request,
        protocol="ollama",
        endpoint="/api/chat",
        call=lambda client, body: client.chat(body),
        stream_call=lambda client, body: client.chat_stream(body),
    )
