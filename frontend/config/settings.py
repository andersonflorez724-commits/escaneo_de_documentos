"""Configuracion del proyecto Django (frontend del escaner).

Los valores se leen de `.env` en la raiz del repositorio cuando existe, de
modo que el mismo codigo sirve para local y para el despliegue en Vercel.

No hay base de datos, sesiones ni historial: la app solo renderiza la pagina
y hace de proxy hacia FastAPI.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# frontend/config/settings.py -> frontend/ -> raiz del proyecto
BASE_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = BASE_DIR.parent

for _candidate in (PROJECT_ROOT / ".env", BASE_DIR / ".env"):
    if _candidate.is_file():
        load_dotenv(_candidate, override=False)


def env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    return value.strip().lower() in {"1", "true", "yes", "on", "si", "sí"}


def env_list(name: str, default: list[str] | None = None) -> list[str]:
    value = os.getenv(name)
    if not value:
        return list(default or [])
    return [item.strip() for item in value.split(",") if item.strip()]


ON_VERCEL = bool(os.getenv("VERCEL"))

# ---------------------------------------------------------------------------
# Seguridad
# ---------------------------------------------------------------------------
SECRET_KEY = os.getenv("DJANGO_SECRET_KEY") or os.getenv("SECRET_KEY", "django-insecure-solo-para-desarrollo")
DEBUG = env_bool("DJANGO_DEBUG", not ON_VERCEL)

ALLOWED_HOSTS = env_list("DJANGO_ALLOWED_HOSTS", ["localhost", "127.0.0.1"])
if ON_VERCEL and ".vercel.app" not in ALLOWED_HOSTS:
    ALLOWED_HOSTS.append(".vercel.app")

# `VERCEL_URL` no incluye el esquema (por ejemplo `mi-app.vercel.app`).
_vercel_url = os.getenv("VERCEL_URL")
if _vercel_url and _vercel_url not in ALLOWED_HOSTS:
    ALLOWED_HOSTS.append(_vercel_url)

CSRF_TRUSTED_ORIGINS = env_list(
    "DJANGO_CSRF_TRUSTED_ORIGINS",
    ["http://localhost:8000", "http://127.0.0.1:8000", "https://*.vercel.app"],
)
if _vercel_url:
    CSRF_TRUSTED_ORIGINS.append(f"https://{_vercel_url}")

# Vercel termina el TLS en su borde y reenvia la peticion por HTTP.
if not DEBUG:
    SECURE_PROXY_SSL_HEADER = ("HTTP_X_FORWARDED_PROTO", "https")
    USE_X_FORWARDED_HOST = True

# ---------------------------------------------------------------------------
# Aplicaciones  (sin auth, sesiones ni mensajes: no hay base de datos)
# ---------------------------------------------------------------------------
INSTALLED_APPS = [
    "django.contrib.staticfiles",
    # Apps del proyecto
    "apps.scanner",
]

MIDDLEWARE = [
    "django.middleware.security.SecurityMiddleware",
    "django.middleware.common.CommonMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
    "django.middleware.clickjacking.XFrameOptionsMiddleware",
]

ROOT_URLCONF = "config.urls"

TEMPLATES = [
    {
        "BACKEND": "django.template.backends.django.DjangoTemplates",
        "DIRS": [BASE_DIR / "templates"],
        "APP_DIRS": True,
        "OPTIONS": {
            "context_processors": [
                "django.template.context_processors.request",
                "apps.scanner.context_processors.api_config",
            ],
        },
    },
]

WSGI_APPLICATION = "config.wsgi.application"
ASGI_APPLICATION = "config.asgi.application"

# ---------------------------------------------------------------------------
# Base de datos: no se usa. Django no necesita ENGINE ni NAME porque ninguna
# app del proyecto define modelos ni contribuye tablas.
# ---------------------------------------------------------------------------
DATABASES = {}

# ---------------------------------------------------------------------------
# Internacionalizacion
# ---------------------------------------------------------------------------
LANGUAGE_CODE = "es-co"
TIME_ZONE = "America/Bogota"
USE_I18N = True
USE_TZ = True

# ---------------------------------------------------------------------------
# Archivos estaticos
# ---------------------------------------------------------------------------
STATIC_URL = "/static/"

_static_dirs = [BASE_DIR / "static"]
STATICFILES_DIRS = [d for d in _static_dirs if d.exists()]

# En Vercel los archivos recolectados se publican desde `public/`, que el CDN
# sirve directamente en la raiz del dominio. Asi `/static/css/styles.css` se
# resuelve sin llegar a la funcion serverless. El collectstatic debe correrse
# con VERCEL=1 (o DJANGO_STATIC_ROOT) para que quede en `public/static/`.
if os.getenv("DJANGO_STATIC_ROOT"):
    STATIC_ROOT = Path(os.environ["DJANGO_STATIC_ROOT"])
elif ON_VERCEL:
    STATIC_ROOT = PROJECT_ROOT / "public" / "static"
else:
    STATIC_ROOT = BASE_DIR / "staticfiles"

STORAGES = {
    "default": {"BACKEND": "django.core.files.storage.FileSystemStorage"},
    "staticfiles": {"BACKEND": "django.contrib.staticfiles.storage.StaticFilesStorage"},
}

MEDIA_URL = "/media/"
MEDIA_ROOT = BASE_DIR / "media"

DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"

# ---------------------------------------------------------------------------
# Integracion con el backend FastAPI
# ---------------------------------------------------------------------------
# Vacio = mismo origen (despliegue Vercel: Django y FastAPI comparten dominio).
# En local se define FASTAPI_BASE_URL en `.env` (por ejemplo http://127.0.0.1:8001).
if ON_VERCEL:
    FASTAPI_BASE_URL = os.getenv("FASTAPI_BASE_URL", "").rstrip("/")
else:
    FASTAPI_BASE_URL = os.getenv("FASTAPI_BASE_URL", "http://127.0.0.1:8001").rstrip("/")
FASTAPI_TIMEOUT = float(os.getenv("FASTAPI_TIMEOUT", "45"))

# Tope de subida aceptado por el navegador (debe coincidir con MAX_UPLOAD_MB).
# En Vercel el limite de cuerpo es ~4.5 MB, por eso el maximo es 4.
if ON_VERCEL:
    MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "4"))
else:
    MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "8"))

# Limites de subida de Django. El cuerpo de la peticion debe admitir la imagen
# completa (con un margen para el resto de campos del formulario); a partir de
# FILE_UPLOAD_MAX_MEMORY_SIZE Django guarda el archivo en disco en lugar de
# mantenerlo en memoria.
DATA_UPLOAD_MAX_MEMORY_SIZE = MAX_UPLOAD_MB * 1024 * 1024 + 1024 * 1024
FILE_UPLOAD_MAX_MEMORY_SIZE = 2 * 1024 * 1024
FILE_UPLOAD_MAX_NUMBER_FILES = 5

# ---------------------------------------------------------------------------
# Cookies
# ---------------------------------------------------------------------------
CSRF_COOKIE_SAMESITE = "Lax"
CSRF_COOKIE_SECURE = env_bool("DJANGO_SECURE_COOKIES", not DEBUG)
X_FRAME_OPTIONS = "SAMEORIGIN"

LOGGING = {
    "version": 1,
    "disable_existing_loggers": False,
    "handlers": {"console": {"class": "logging.StreamHandler"}},
    "root": {"handlers": ["console"], "level": os.getenv("DJANGO_LOG_LEVEL", "INFO")},
}
