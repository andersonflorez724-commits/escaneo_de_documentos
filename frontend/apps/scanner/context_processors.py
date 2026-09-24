"""Variables de plantilla compartidas por todas las vistas."""

from __future__ import annotations

from typing import Any

from django.conf import settings
from django.http import HttpRequest


def api_config(request: HttpRequest) -> dict[str, Any]:
    """Expone la configuracion de la API consumida por el navegador.

    Si ``FASTAPI_BASE_URL`` esta vacio se asume same-origin y el enlace a
    ``/docs`` se construye con una ruta relativa.
    """
    return {
        "fastapi_base_url": settings.FASTAPI_BASE_URL,
        "max_upload_mb": settings.MAX_UPLOAD_MB,
        "app_name": "Lector e Inspector de Documentos",
        "app_version": "1.0.0",
    }
