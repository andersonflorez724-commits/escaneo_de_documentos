"""Punto de entrada unico para Vercel Serverless Functions.

Antes habia dos builds ``@vercel/python`` (backend y frontend) que compartian
el mismo venv en ``.vercel/python/.venv``: una instalacion pisaba metadatos de
la otra (p. ej. ``annotated_types``) y el build moria con ``ENOENT``.

Aqui hay **una sola** funcion Python que enruta por path:

* ``/api/v1/*``, ``/docs``, ``/redoc``, ``/openapi.json`` -> FastAPI
* el resto -> Django

Dependencias: ``api/requirements.txt`` (FastAPI + Django + RapidOCR + OpenCV).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Awaitable, Callable, MutableMapping, Optional, Union

# api/index.py -> raiz del proyecto
ROOT = Path(__file__).resolve().parent.parent
BACKEND_DIR = ROOT / "backend"
FRONTEND_DIR = ROOT / "frontend"

for _path in (str(BACKEND_DIR), str(FRONTEND_DIR)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

# Defaults serverless (setdefault no pisa variables de Vercel).
# auto -> RapidOCR si el paquete esta en el bundle; si no, heuristico.
os.environ.setdefault("OCR_ENGINE", "auto")
os.environ.setdefault("OCR_USE_GPU", "false")
os.environ.setdefault("MAX_UPLOAD_MB", "4")
os.environ.setdefault("MAX_IMAGE_DIMENSION", "1280")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings")

from app.main import create_app  # noqa: E402

api_app = create_app()

from django.core.asgi import get_asgi_application  # noqa: E402

web_app = get_asgi_application()

Scope = MutableMapping[str, object]
Receive = Callable[[], Awaitable[MutableMapping[str, object]]]
Send = Callable[[MutableMapping[str, object]], Awaitable[None]]


def _is_api_path(path: str) -> bool:
    # Solo la API de vision de FastAPI. El proxy Django del frontend
    # vive en /api/escanear/ y /api/validar-mrz/ (sin prefijo v1).
    if path.startswith("/api/v1"):
        return True
    return path in {"/docs", "/redoc", "/openapi.json"} or path.startswith("/docs/")


class App:
    """ASGI simple: FastAPI en rutas de API, Django en el resto."""

    async def __call__(
        self,
        scope: Scope,
        receive: Receive,
        send: Send,
    ) -> None:
        if scope["type"] == "lifespan":
            # Solo FastAPI necesita lifespan (warmup OCR). Django no usa BD.
            await api_app(scope, receive, send)
            return

        path = str(scope.get("path", ""))
        target = api_app if _is_api_path(path) else web_app
        await target(scope, receive, send)


app = App()
application = app

__all__ = ["app", "application"]
