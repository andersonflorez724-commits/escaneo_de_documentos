"""Enrutador principal de la version 1 de la API."""

from fastapi import APIRouter

from app.api.v1.endpoints import auth, documents, health

api_router = APIRouter()
api_router.include_router(health.router)
api_router.include_router(auth.router)
api_router.include_router(documents.router)

__all__ = ["api_router"]
