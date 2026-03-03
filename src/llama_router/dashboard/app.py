"""Dashboard web application served on port 80."""

from __future__ import annotations

from fastapi import FastAPI

from .routes import router

app = FastAPI(title="llama-router Dashboard")
app.include_router(router)
