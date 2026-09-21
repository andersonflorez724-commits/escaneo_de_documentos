"""Punto de entrada de la aplicacion FastAPI."""

from __future__ import annotations

import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.v1.router import api_router
from app.core.config import get_settings

logger = logging.getLogger(__name__)


def create_app() -> FastAPI:
    """Construye y configura la instancia de FastAPI."""
    settings = get_settings()
    logging.basicConfig(level=getattr(logging, settings.log_level, logging.INFO))

    app = FastAPI(
        title=settings.app_name,
        description=settings.description,
        version=settings.app_version,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        contact={
            "name": "Anderson Florez",
            "email": "andersonflorez724@gmail.com",
        },
        license_info={"name": "MIT"},
    )

    if settings.allowed_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=list(settings.allowed_origins),
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    app.include_router(api_router, prefix=settings.api_v1_prefix)
    logger.info("FastAPI listo: %s v%s", settings.app_name, settings.app_version)
    return app


app = create_app()

__all__ = ["app", "create_app"]
