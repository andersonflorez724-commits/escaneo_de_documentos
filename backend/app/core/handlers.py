"""Manejadores globales de errores: respuestas uniformes para toda la API.

Todas las respuestas de error comparten la misma forma, de modo que el
cliente Django (y cualquier consumidor) pueda tratarlas igual:

```json
{"detail": "...", "code": "invalid_image", "request_id": "..."}
```
"""

from __future__ import annotations

import logging

from fastapi import FastAPI, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.core.errors import AppError

logger = logging.getLogger(__name__)


def _request_id(request: Request) -> str | None:
    return getattr(request.state, "request_id", None)


def _body(detail: object, code: str, request: Request, context: object = None) -> dict[str, object]:
    payload: dict[str, object] = {"detail": detail, "code": code}
    request_id = _request_id(request)
    if request_id:
        payload["request_id"] = request_id
    if context:
        payload["context"] = context
    return payload


def register_exception_handlers(app: FastAPI) -> None:
    """Registra los manejadores de error de la aplicacion."""

    @app.exception_handler(AppError)
    async def handle_app_error(request: Request, exc: AppError) -> JSONResponse:
        """Errores de dominio: se devuelven con su codigo y contexto."""
        logger.info(
            "Error de dominio %s en %s %s: %s",
            exc.code,
            request.method,
            request.url.path,
            exc.message,
        )
        return JSONResponse(
            status_code=exc.status_code,
            content=_body(exc.message, exc.code, request, exc.context or None),
        )

    @app.exception_handler(RequestValidationError)
    async def handle_validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
        """Errores de validacion de Pydantic con un resumen legible."""
        # `exc.errors()` puede traer objetos no serializables en `ctx`
        # (por ejemplo la excepcion original de un validador), por eso se
        # normaliza antes de responder.
        errors = jsonable_encoder(exc.errors(), exclude={"ctx", "url", "input"})
        summary = ". ".join(
            f"{'.'.join(str(part) for part in error.get('loc', [])[1:])}: {error.get('msg', 'dato invalido')}"
            for error in errors[:5]
        )
        return JSONResponse(
            status_code=422,
            content=_body(
                errors,
                "validation_error",
                request,
                {"summary": summary or "Datos invalidos."},
            ),
        )

    @app.exception_handler(StarletteHTTPException)
    async def handle_http_exception(request: Request, exc: StarletteHTTPException) -> JSONResponse:
        """Mantiene los mensajes de FastAPI con la forma uniforme."""
        return JSONResponse(
            status_code=exc.status_code,
            content=_body(exc.detail, "http_error", request),
            headers=getattr(exc, "headers", None),
        )

    @app.exception_handler(Exception)
    async def handle_unexpected_error(request: Request, exc: Exception) -> JSONResponse:
        """Red de seguridad: nunca se filtra un traceback al cliente."""
        logger.exception(
            "Error inesperado en %s %s",
            request.method,
            request.url.path,
            exc_info=exc,
        )
        return JSONResponse(
            status_code=500,
            content=_body(
                "Ocurrio un error interno. Vuelve a intentarlo en unos segundos.",
                "internal_error",
                request,
            ),
        )


__all__ = ["register_exception_handlers"]
