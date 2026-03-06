"""Ollama-compatible API served on port 11434."""

from __future__ import annotations

from fastapi import FastAPI

from .chat import router as chat_router
from .embeddings import router as embeddings_router
from .tags import router as tags_router

from ..llamacpp_api.audio import router as v1_audio_router
from ..llamacpp_api.chat import router as v1_chat_router
from ..llamacpp_api.completions import router as v1_completions_router
from ..llamacpp_api.embeddings import router as v1_embeddings_router
from ..llamacpp_api.images import router as v1_images_router
from ..llamacpp_api.models import router as v1_models_router
from ..llamacpp_api.responses import router as v1_responses_router

app = FastAPI(title="llama-router Ollama API")

app.include_router(chat_router)
app.include_router(embeddings_router)
app.include_router(tags_router)

app.include_router(v1_audio_router)
app.include_router(v1_chat_router)
app.include_router(v1_completions_router)
app.include_router(v1_embeddings_router)
app.include_router(v1_images_router)
app.include_router(v1_models_router)
app.include_router(v1_responses_router)
