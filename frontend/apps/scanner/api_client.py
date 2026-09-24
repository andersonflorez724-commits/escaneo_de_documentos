"""Cliente HTTP del frontend hacia la API FastAPI.

El navegador nunca habla directamente con el backend de vision: Django actua
como *backend for frontend*.

    Navegador ──cookies──► Django ──HTTP──► FastAPI

No hay sesiones ni base de datos: cada peticion es independiente. Si
``FASTAPI_BASE_URL`` esta vacio (despliegue same-origin en Vercel), la URL se
deriva de la peticion entrante.
"""

from __future__ import annotations

import logging
from typing import Any, Mapping

import requests
from django.conf import settings
from django.http import HttpRequest

logger = logging.getLogger(__name__)

# Errores del backend que se devuelven sin reinterpretar.
_PASS_THROUGH_STATUSES = frozenset({400, 403, 404, 409, 413, 415, 422, 429, 502, 503, 504})


class ApiError(Exception):
    """Error controlado al comunicarse con la API FastAPI."""

    def __init__(self, message: str, *, status_code: int | None = None, payload: Any = None) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.payload = payload

    @property
    def http_status(self) -> int:
        """Codigo HTTP que debe devolverse al navegador."""
        status = self.status_code

        if status in _PASS_THROUGH_STATUSES:
            return status
        return 502


class FastAPIClient:
    """Envoltorio minimo y sin estado sobre la API de vision."""

    def __init__(
        self,
        *,
        base_url: str | None = None,
        timeout: float | None = None,
    ) -> None:
        self.base_url = (base_url or settings.FASTAPI_BASE_URL).rstrip("/")
        self.timeout = timeout or settings.FASTAPI_TIMEOUT

    # ---------------------------------------------------------------- HTTP
    def _headers(self, extra: Mapping[str, str] | None = None) -> dict[str, str]:
        headers = {"Accept": "application/json"}
        if extra:
            headers.update(extra)
        return headers

    def _request(
        self,
        method: str,
        path: str,
        *,
        headers: Mapping[str, str] | None = None,
        **kwargs: Any,
    ) -> Any:
        url = f"{self.base_url}{path}"

        try:
            response = requests.request(
                method,
                url,
                headers=self._headers(headers),
                timeout=self.timeout,
                **kwargs,
            )
        except requests.Timeout as exc:
            logger.warning("Timeout llamando a %s %s", method, url)
            raise ApiError(
                "El servidor de vision tardo demasiado en responder. Intentalo de nuevo.",
                status_code=504,
            ) from exc
        except requests.RequestException as exc:
            logger.warning("Fallo de red llamando a %s %s: %s", method, url, exc)
            raise ApiError(
                "No se pudo contactar con el servidor de vision. Verifica que el backend este activo.",
                status_code=503,
            ) from exc

        payload = self._decode(response)

        if response.status_code >= 400:
            raise ApiError(
                self._extract_detail(payload, response.status_code),
                status_code=response.status_code,
                payload=payload,
            )

        return payload

    @staticmethod
    def _decode(response: requests.Response) -> Any:
        if not response.content:
            return {}
        try:
            return response.json()
        except ValueError:
            logger.warning("Respuesta no JSON (%s): %.200s", response.status_code, response.text)
            return {}

    @staticmethod
    def _extract_detail(payload: Any, status_code: int) -> str:
        detail = payload.get("detail") if isinstance(payload, dict) else None

        if isinstance(detail, str):
            return detail

        if isinstance(detail, list):
            messages = [str(item.get("msg", "dato invalido")) for item in detail if isinstance(item, dict)]
            if messages:
                return ". ".join(messages)

        if status_code == 422:
            return "La imagen no se pudo procesar. Intenta con otra fotografia."
        return "El servidor de vision devolvio un error."

    # ----------------------------------------------------------- endpoints
    def health(self) -> dict[str, Any]:
        return self._request("GET", "/api/v1/health")

    def scan_document(
        self,
        *,
        content: bytes,
        filename: str = "documento.jpg",
        content_type: str = "image/jpeg",
        back_content: bytes | None = None,
        back_filename: str = "documento-reverso.jpg",
        back_content_type: str = "image/jpeg",
        include_preview: bool = False,
    ) -> dict[str, Any]:
        """Envia la foto del documento y devuelve los campos extraidos.

        La cara frontal viaja en el campo ``file`` y, si se recibe, el reverso
        en ``back_file``: el backend fusiona el texto de ambas antes de extraer
        los campos.
        """
        params = {"include_preview": "true"} if include_preview else None

        files: dict[str, tuple[str, bytes, str]] = {
            "file": (filename, content, content_type),
        }
        if back_content:
            files["back_file"] = (back_filename, back_content, back_content_type)

        return self._request(
            "POST",
            "/api/v1/scan-document",
            params=params,
            files=files,
        )

    def validate_mrz(
        self,
        mrz_lines: list[str] | None = None,
        text: str | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {}
        if mrz_lines:
            payload["mrz_lines"] = mrz_lines
        if text:
            payload["text"] = text
        return self._request("POST", "/api/v1/validate-mrz", json=payload)


def get_service_client(request: HttpRequest | None = None) -> FastAPIClient:
    """Devuelve el cliente HTTP hacia FastAPI (sin autenticacion).

    Si ``FASTAPI_BASE_URL`` esta vacio se usa el origen de la peticion
    entrante (same-origin en Vercel: Django y FastAPI comparten dominio).
    """
    base_url = settings.FASTAPI_BASE_URL
    if not base_url and request is not None:
        base_url = request.build_absolute_uri("/").rstrip("/")
    return FastAPIClient(base_url=base_url)


__all__ = [
    "ApiError",
    "FastAPIClient",
    "get_service_client",
]
