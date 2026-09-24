"""Punto de entrada del cliente Django para **Vercel Serverless Functions**.

Vercel ejecuta el objeto ``app`` del modulo; aqui es la aplicacion WSGI de
Django. Las dependencias se declaran en ``frontend/api/requirements.txt``.

Antes del primer despliegue hay que configurar en Vercel:

* ``DJANGO_SECRET_KEY``
* ``DJANGO_DEBUG=false``

No hay base de datos ni sesiones. ``FASTAPI_BASE_URL`` puede quedar vacio
para same-origin (Django y FastAPI comparten el dominio en Vercel).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# frontend/api/index.py -> frontend/ -> raiz
FRONTEND_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = FRONTEND_DIR.parent

if str(FRONTEND_DIR) not in sys.path:
    sys.path.insert(0, str(FRONTEND_DIR))

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings")

from django.core.wsgi import get_wsgi_application  # noqa: E402

app = get_wsgi_application()

__all__ = ["app"]
