from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import Response

from ..models import Provider
from ..proxy import proxy_json
from ..translate import OllamaOverOpenAI, TranslationError

router = APIRouter()


def _check_translatable(provider: Provider, body: dict[str, Any]) -> None:
    if not provider.is_ollama:
        # Served through OllamaOverOpenAI: refuse what it cannot translate
        # before a response starts.
        try:
            OllamaOverOpenAI.validate("generate", body)
        except TranslationError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/api/generate")
async def generate(request: Request) -> Response:
    return await proxy_json(
        request,
        protocol="ollama",
        endpoint="/api/generate",
        call=lambda client, body: client.generate(body),
        stream_call=lambda client, body: client.generate_stream(body),
        check=_check_translatable,
    )
