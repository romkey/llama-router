"""Ollama-compatible API served on port 11434."""

from __future__ import annotations

from fastapi import FastAPI

from .chat import router as chat_router
from .embeddings import router as embeddings_router
from .tags import router as tags_router

app = FastAPI(title="llama-router Ollama API")

app.include_router(chat_router)
app.include_router(embeddings_router)
app.include_router(tags_router)
