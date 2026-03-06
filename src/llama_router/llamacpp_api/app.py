"""llama.cpp / OpenAI-compatible API served on port 8080."""

from __future__ import annotations

from fastapi import FastAPI

from .audio import router as audio_router
from .chat import router as chat_router
from .completions import router as completions_router
from .embeddings import router as embeddings_router
from .images import router as images_router
from .models import router as models_router
from .responses import router as responses_router

app = FastAPI(title="llama-router llama.cpp API")

app.include_router(audio_router)
app.include_router(chat_router)
app.include_router(completions_router)
app.include_router(embeddings_router)
app.include_router(images_router)
app.include_router(models_router)
app.include_router(responses_router)
