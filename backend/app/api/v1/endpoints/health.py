"""Endpoints de sistema: salud y metadatos del servicio."""

from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel, Field

from app.core.config import get_settings

router = APIRouter(tags=["Sistema"])


class HealthResponse(BaseModel):
    """Respuesta del chequeo de salud del servicio."""

    status: str = Field(description="`ok` cuando el servicio responde.", examples=["ok"])
    service: str = Field(description="Nombre del servicio.", examples=["Lector e Inspector de Documentos"])
    version: str = Field(description="Version desplegada.", examples=["1.0.0"])


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Estado del servicio",
    description="Permite verificar que la funcion serverless esta viva.",
)
async def health() -> HealthResponse:
    settings = get_settings()
    return HealthResponse(status="ok", service=settings.app_name, version=settings.app_version)
