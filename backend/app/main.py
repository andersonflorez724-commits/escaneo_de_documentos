"""Punto de entrada de la aplicacion FastAPI."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.v1.router import api_router
from app.core.config import get_settings
from app.services.ocr_engine import get_ocr_engine, warmup_ocr_engine
from app.services.users import get_user_repository

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:  # noqa: ARG001 - firma de FastAPI
    """Carga el modelo preentrenado al arrancar el proceso.

    Los pesos de EasyOCR tardan varios segundos en cargarse. Hacerlo aqui
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
        contact={
            "name": "Anderson Florez",
            "email": "andersonflorez724@gmail.com",
        },
        license_info={"name": "MIT"},
        openapi_tags=[
            {
                "name": "Sistema",
                "description": "Salud y disponibilidad del servicio.",
            },
            {
                "name": "Autenticacion",
                "description": (
                    "Registro e inicio de sesion. Devuelve el **JWT** que hay que "
                    "enviar en la cabecera `Authorization: Bearer <token>`."
                ),
            },
            {
                "name": "Documentos",
                "description": (
                    "Lectura e inspeccion de documentos de identificacion: OCR con "
                    "modelo preentrenado, deteccion de rostro y validacion de la MRZ."
                ),
            },
        ],
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

    # Usuario semilla para poder probar el flujo completo sin base de datos.
    seed_user = get_user_repository().ensure_seed_user()
    logger.info("Usuario semilla disponible: %s", seed_user.email)
    logger.info(
        "Motor de OCR configurado: %s (se cargara en el arranque)",
        get_ocr_engine().name,
    )
    logger.info("FastAPI listo: %s v%s", settings.app_name, settings.app_version)
    return app


app = create_app()

__all__ = ["app", "create_app"]
