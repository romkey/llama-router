"""Dashboard web application served on port 80."""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.templating import Jinja2Templates

from .routes import router

TEMPLATE_DIR = Path(__file__).resolve().parent.parent / "templates"

app = FastAPI(title="llama-router Dashboard")
templates = Jinja2Templates(directory=str(TEMPLATE_DIR))

app.include_router(router)
