"""Errores de dominio de la aplicacion.

Definirlos aqui permite que las capas de servicio no dependan de FastAPI y
que exista un unico lugar donde se decide el codigo HTTP y el formato de la
respuesta de error.
"""

from __future__ import annotations

from typing import Any


class AppError(Exception):
    """Error esperado del dominio, con codigo HTTP asociado."""

    status_code: int = 400
    code: str = "application_error"
    default_message: str = "Ocurrio un error al procesar la solicitud."

    def __init__(
        self,
        message: str | None = None,
        *,
        code: str | None = None,
        status_code: int | None = None,
        context: dict[str, Any] | None = None,
    ) -> None:
        self.message = message or self.default_message
        if code:
            self.code = code
        if status_code:
            self.status_code = status_code
        self.context = context or {}
        super().__init__(self.message)

    def as_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"detail": self.message, "code": self.code}
        if self.context:
            payload["context"] = self.context
        return payload


class InvalidImageError(AppError):
    """La imagen recibida no se pudo decodificar o no es utilizable."""

    status_code = 422
    code = "invalid_image"
    default_message = "No se pudo leer la imagen enviada."


class DocumentScanError(AppError):
    """Fallo interno durante el procesamiento del documento."""

    status_code = 500
    code = "scan_failed"
    default_message = "No se pudo procesar el documento."


class PayloadTooLargeError(AppError):
    """El archivo supera el limite permitido."""

    status_code = 413
    code = "payload_too_large"
    default_message = "El archivo enviado es demasiado grande."


class UnsupportedMediaError(AppError):
    """El tipo de archivo no esta soportado."""

    status_code = 415
    code = "unsupported_media_type"
    default_message = "El tipo de archivo no esta soportado."


class OCRAvailabilityError(AppError):
    """El motor de OCR no esta disponible en este entorno."""

    status_code = 503
    code = "ocr_unavailable"
    default_message = "El motor de lectura no esta disponible en este momento."


__all__ = [
    "AppError",
    "DocumentScanError",
    "InvalidImageError",
    "OCRAvailabilityError",
    "PayloadTooLargeError",
    "UnsupportedMediaError",
]
