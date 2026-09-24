"""Punto de entrada de la API para **Vercel Serverless Functions**.

Vercel busca el objeto ``app`` (o ``application``) del modulo. Al ser una
aplicacion ASGI (FastAPI), el runtime de Python la monta como tal, de modo
que Swagger sigue disponible en ``/docs`` y las rutas mantienen el prefijo
``/api/v1``.

Las dependencias de esta funcion se declaran en ``backend/api/requirements.txt``
(el runtime busca el archivo en la misma carpeta que este punto de entrada).

El punto de entrada real en produccion es ``api/index.py`` (raiz del repo),
que monta Django y FastAPI en una sola funcion. Aqui ``OCR_ENGINE=auto``
elige RapidOCR si el paquete esta en el bundle; si no, cae al motor
heuristico. Para aislar el OCR del limite de la funcion se puede desplegar
``backend/`` como contenedor RapidOCR (Dockerfile de la raiz) y apuntar
``FASTAPI_BASE_URL`` hacia esa URL.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# backend/api/index.py -> backend/ -> raiz
BACKEND_DIR = Path(__file__).resolve().parent.parent

if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

# Valores por defecto pensados para el entorno serverless. Las variables que
# existan en Vercel tienen prioridad porque `setdefault` no las sobreescribe.
# MAX_UPLOAD_MB=4: el limite de cuerpo de Vercel es ~4.5 MB.
os.environ.setdefault("OCR_ENGINE", "auto")
os.environ.setdefault("OCR_USE_GPU", "false")
os.environ.setdefault("MAX_UPLOAD_MB", "4")
os.environ.setdefault("MAX_IMAGE_DIMENSION", "1280")
os.environ.setdefault("OMP_NUM_THREADS", "1")

from app.main import create_app  # noqa: E402

app = create_app()
application = app

__all__ = ["app", "application"]
