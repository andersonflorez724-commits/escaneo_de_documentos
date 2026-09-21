"""Configuracion de la aplicacion.

Toda la configuracion se lee de variables de entorno (ver ``.env.example``)
y queda expuesta a traves de :func:`get_settings`, que cachea la instancia
resultante para que el coste de parseo se pague una sola vez por proceso.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

# backend/app/core/config.py -> backend/ -> raiz del proyecto
BACKEND_DIR = Path(__file__).resolve().parents[2]
PROJECT_ROOT = BACKEND_DIR.parent

# Valores por defecto. Se declaran como constantes de modulo y no como
# atributos de la dataclass porque con ``slots=True`` el acceso via clase
# devuelve el descriptor del slot y no el valor.
DEFAULT_APP_NAME = "Lector e Inspector de Documentos"
DEFAULT_APP_VERSION = "1.0.0"
DEFAULT_SECRET_KEY = "dev-secret-key-cambiar-en-produccion"
DEFAULT_SEED_EMAIL = "admin@escaneo.local"
DEFAULT_SEED_PASSWORD = "Admin123*"
DEFAULT_SEED_NAME = "Administrador"

DESCRIPTION = """
API de **lectura e inspeccion de documentos de identificacion**.

Combina un modelo preentrenado de *EasyOCR* con *OpenCV* para detectar las
zonas de texto y el rostro del documento, extraer los campos relevantes y
validar la consistencia de la **MRZ** (Machine Readable Zone) segun el
estandar ICAO 9293 (digitos de control modulo 10, pesos 7-3-1).

### Flujo tipico
1. `POST /api/v1/auth/login` &rarr; obtiene el token JWT.
2. Autoriza con el boton **Authorize** (`Bearer <token>`).
3. `POST /api/v1/scan-document` con la foto del documento.
4. `POST /api/v1/validate-mrz` para la verificacion de consistencia.
"""


def _load_env() -> None:
    """Carga `.env` desde la raiz del proyecto o desde ``backend/``."""
    for candidate in (PROJECT_ROOT / ".env", BACKEND_DIR / ".env"):
        if candidate.is_file():
            load_dotenv(candidate, override=False)


def _as_bool(value: str | None, default: bool = False) -> bool:
    if value is None or value == "":
        return default
    return value.strip().lower() in {"1", "true", "yes", "on", "si", "sí"}


def _as_int(value: str | None, default: int) -> int:
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return default


def _as_float(value: str | None, default: float) -> float:
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return default


def _as_tuple(value: str | None, default: tuple[str, ...] = ()) -> tuple[str, ...]:
    if not value:
        return default
    parts = [item.strip() for item in value.split(",")]
    return tuple(item for item in parts if item)


@dataclass(frozen=True, slots=True)
class Settings:
    """Configuracion inmutable del servicio."""

    # ------------------------------ Metadatos ------------------------------
    app_name: str = DEFAULT_APP_NAME
    app_version: str = DEFAULT_APP_VERSION
    description: str = DESCRIPTION
    api_v1_prefix: str = "/api/v1"
    log_level: str = "INFO"

    # --------------------------- Autenticacion -----------------------------
    secret_key: str = DEFAULT_SECRET_KEY
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 120
    seed_user_email: str = DEFAULT_SEED_EMAIL
    seed_user_password: str = DEFAULT_SEED_PASSWORD
    seed_user_name: str = DEFAULT_SEED_NAME

    # -------------------------------- CORS ---------------------------------
    allowed_origins: tuple[str, ...] = ()

    # ------------------------- Motor OCR / ML ------------------------------
    ocr_engine: str = "auto"
    ocr_languages: tuple[str, ...] = ("es", "en")
    ocr_use_gpu: bool = False
    ocr_min_confidence: float = 0.30
    max_image_dimension: int = 1600
    face_detector: str = "auto"

    # --------------------------- Subidas -----------------------------------
    max_upload_mb: int = 8

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024


def _build_settings() -> Settings:
    _load_env()
    env = os.environ

    return Settings(
        app_name=env.get("APP_NAME") or DEFAULT_APP_NAME,
        app_version=env.get("APP_VERSION") or DEFAULT_APP_VERSION,
        log_level=(env.get("LOG_LEVEL") or "INFO").upper(),
        secret_key=env.get("SECRET_KEY") or env.get("DJANGO_SECRET_KEY") or DEFAULT_SECRET_KEY,
        algorithm=env.get("JWT_ALGORITHM") or "HS256",
        access_token_expire_minutes=_as_int(env.get("ACCESS_TOKEN_EXPIRE_MINUTES"), 120),
        seed_user_email=env.get("SEED_USER_EMAIL") or DEFAULT_SEED_EMAIL,
        seed_user_password=env.get("SEED_USER_PASSWORD") or DEFAULT_SEED_PASSWORD,
        seed_user_name=env.get("SEED_USER_NAME") or DEFAULT_SEED_NAME,
        allowed_origins=_as_tuple(env.get("ALLOWED_ORIGINS")),
        ocr_engine=(env.get("OCR_ENGINE") or "auto").strip().lower(),
        ocr_languages=_as_tuple(env.get("OCR_LANGUAGES"), ("es", "en")),
        ocr_use_gpu=_as_bool(env.get("OCR_USE_GPU"), False),
        ocr_min_confidence=_as_float(env.get("OCR_MIN_CONFIDENCE"), 0.30),
        max_image_dimension=_as_int(env.get("MAX_IMAGE_DIMENSION"), 1600),
        face_detector=(env.get("FACE_DETECTOR") or "auto").strip().lower(),
        max_upload_mb=_as_int(env.get("MAX_UPLOAD_MB"), 8),
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Devuelve la configuracion cacheada del proceso."""
    return _build_settings()
