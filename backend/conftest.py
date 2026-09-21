"""Configuracion global de pytest para el backend.

Agrega el directorio ``backend/`` al ``sys.path`` para que las pruebas puedan
importar el paquete ``app`` sin instalar el proyecto.
"""

from __future__ import annotations

import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))
