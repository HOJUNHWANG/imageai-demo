"""
ImageAI Studio — FastAPI Backend
"""
import sys
import os

# Ensure project root is on path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from .routers import generate, edit, mask, system
from .core.pipeline import get_inpaint_pipe


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup/shutdown lifecycle. Models lazy-load on first request."""
    print("[BOOT] ImageAI Studio backend ready (models will lazy-load on first request).")
    yield
    print("[SHUTDOWN] Cleaning up...")


app = FastAPI(
    title="ImageAI Studio",
    version="2.0.0",
    lifespan=lifespan,
)

# CORS — allow Next.js dev server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API routers
app.include_router(generate.router, prefix="/api", tags=["generate"])
app.include_router(edit.router, prefix="/api", tags=["edit"])
app.include_router(mask.router, prefix="/api", tags=["mask"])
app.include_router(system.router, prefix="/api", tags=["system"])


@app.get("/api/health")
async def health():
    return {"status": "ok"}
