"""Punto de entrada de la aplicacion FastAPI."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware

from app.api.v1.router import api_router
from app.core.config import get_settings
from app.core.handlers import register_exception_handlers
from app.core.middleware import RequestContextMiddleware
from app.services.ocr_engine import get_ocr_engine, warmup_ocr_engine

logger = logging.getLogger(__name__)

# Respuestas mas pequenas que esto no se comprimen: el coste no compensa.
GZIP_MINIMUM_SIZE = 1024

OPENAPI_TAGS = [
    {"name": "Sistema", "description": "Salud y disponibilidad del servicio."},
    {
        "name": "Documentos",
        "description": (
            "Lectura e inspeccion de documentos de identificacion: OCR con "
            "modelo preentrenado, deteccion de rostro y validacion de la MRZ."
        ),
    },
]


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:  # noqa: ARG001 - firma de FastAPI
    """Carga el modelo preentrenado al arrancar el proceso.

    Los pesos de OCR tardan varios segundos en cargarse; hacerlo aqui
    evita que la primera peticion del usuario pague ese coste.
    """
    warmup_ocr_engine()
    yield
    logger.info("Deteniendo el servicio")


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
        lifespan=lifespan,
        openapi_tags=OPENAPI_TAGS,
        contact={"name": "Anderson Florez", "email": "andersonflorez724@gmail.com"},
        license_info={"name": "MIT"},
    )

    # ------------------------------- Middleware ----------------------------
    # El orden importa: el ultimo anadido es el mas externo.
    if settings.allowed_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=list(settings.allowed_origins),
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    if settings.enable_gzip:
        app.add_middleware(GZipMiddleware, minimum_size=GZIP_MINIMUM_SIZE)

    app.add_middleware(RequestContextMiddleware)

    # --------------------------- Errores uniformes -------------------------
    register_exception_handlers(app)

    # -------------------------------- Rutas --------------------------------
    app.include_router(api_router, prefix=settings.api_v1_prefix)

    logger.info("FastAPI listo: %s v%s", settings.app_name, settings.app_version)
    return app


app = create_app()

__all__ = ["app", "create_app"]
