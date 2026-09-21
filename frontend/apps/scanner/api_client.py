"""Cliente HTTP del frontend hacia la API FastAPI.

El navegador nunca habla directamente con el backend de vision: Django actua
como *backend for frontend*. El usuario se autentica con la sesion de Django y
Django obtiene (y renueva) un **JWT** con la cuenta de servicio, de modo que
ninguna credencial de la API viaja al navegador.

    Navegador ──sesion/cookies──► Django ──JWT (Bearer)──► FastAPI
"""

from __future__ import annotations

import logging
import time
from typing import Any, Mapping

import requests
from django.conf import settings

logger = logging.getLogger(__name__)

# Claves dentro de la sesion de Django.
TOKEN_SESSION_KEY = "fastapi_token"
TOKEN_EXPIRY_SESSION_KEY = "fastapi_token_expires_at"
HISTORY_SESSION_KEY = "scan_history"

# Margen de seguridad para renovar el token antes de que caduque.
TOKEN_REFRESH_MARGIN_SECONDS = 30

MAX_HISTORY_ENTRIES = 20


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
        """Codigo HTTP que debe devolverse al navegador.

        Los errores de validacion se propagan tal cual para que la interfaz
        pueda explicar que paso. Un 401 se traduce a 502 a proposito: es un
        fallo de la credencial de servicio, no de la sesion del usuario, y no
        debe hacer que el navegador lo interprete como sesion expirada.
        """
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
        token: str | None = None,
    ) -> None:
        self.base_url = (base_url or settings.FASTAPI_BASE_URL).rstrip("/")
        self.timeout = timeout or settings.FASTAPI_TIMEOUT
        self.token = token

    # ---------------------------------------------------------------- HTTP
    def _headers(self, extra: Mapping[str, str] | None = None) -> dict[str, str]:
        headers = {"Accept": "application/json"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
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

        if status_code == 401:
            return "No se pudo autenticar con el servidor de vision."
        if status_code == 422:
            return "La imagen no se pudo procesar. Intenta con otra fotografia."
        return "El servidor de vision devolvio un error."

    # ----------------------------------------------------------- endpoints
    def health(self) -> dict[str, Any]:
        return self._request("GET", "/api/v1/health")

    def login(self, email: str, password: str) -> dict[str, Any]:
        """Autentica contra FastAPI y devuelve el token con su vigencia."""
        return self._request(
            "POST",
            "/api/v1/auth/login",
            json={"email": email, "password": password},
        )

    def register(self, email: str, password: str, full_name: str) -> dict[str, Any]:
        return self._request(
            "POST",
            "/api/v1/auth/register",
            json={"email": email, "password": password, "full_name": full_name},
        )

    def me(self) -> dict[str, Any]:
        return self._request("GET", "/api/v1/auth/me")

    def scan_document(
        self,
        *,
        content: bytes,
        filename: str = "documento.jpg",
        content_type: str = "image/jpeg",
        include_preview: bool = False,
    ) -> dict[str, Any]:
        """Envia la foto del documento y devuelve los campos extraidos."""
        params = {"include_preview": "true"} if include_preview else None

        return self._request(
            "POST",
            "/api/v1/scan-document",
            params=params,
            files={"file": (filename, content, content_type)},
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


# ---------------------------------------------------------------------------
# Gestion del token en la sesion de Django
# ---------------------------------------------------------------------------
def store_token(session: Any, token_payload: Mapping[str, Any]) -> None:
    """Guarda el JWT y su instante de caducidad en la sesion."""
    session[TOKEN_SESSION_KEY] = token_payload["access_token"]
    session[TOKEN_EXPIRY_SESSION_KEY] = time.time() + float(token_payload.get("expires_in", 0))


def clear_token(session: Any) -> None:
    session.pop(TOKEN_SESSION_KEY, None)
    session.pop(TOKEN_EXPIRY_SESSION_KEY, None)


def session_token_is_valid(session: Any) -> bool:
    """Indica si el token guardado sigue vigente (con margen de seguridad)."""
    token = session.get(TOKEN_SESSION_KEY)
    expires_at = session.get(TOKEN_EXPIRY_SESSION_KEY)

    if not token or not expires_at:
        return False

    return float(expires_at) - TOKEN_REFRESH_MARGIN_SECONDS > time.time()


def authenticate_service_account(client: FastAPIClient) -> dict[str, Any]:
    """Inicia sesion con la cuenta de servicio del backend.

    Si el backend se reinicio (su almacen de usuarios esta en memoria) la
    cuenta habra desaparecido: en ese caso se registra y se reintenta, de modo
    que el frontend se recupere solo sin intervencion manual.
    """
    email = settings.FASTAPI_SERVICE_EMAIL
    password = settings.FASTAPI_SERVICE_PASSWORD

    if not email or not password:
        raise ApiError(
            "El frontend no tiene configurada la cuenta de servicio de la API. "
            "Define FASTAPI_SERVICE_EMAIL y FASTAPI_SERVICE_PASSWORD.",
            status_code=503,
        )

    try:
        return client.login(email, password)
    except ApiError as exc:
        if exc.status_code != 401:
            raise

        logger.info("La cuenta de servicio no existe en el backend: se registra.")
        try:
            client.register(email, password, full_name="Cuenta de servicio")
        except ApiError as register_exc:
            # Otro proceso pudo registrarla entre ambas llamadas.
            if register_exc.status_code != 409:
                raise exc from register_exc

        return client.login(email, password)


def get_service_client(session: Any) -> FastAPIClient:
    """Devuelve un cliente autenticado, reutilizando el token de la sesion."""
    if session_token_is_valid(session):
        return FastAPIClient(token=session[TOKEN_SESSION_KEY])

    client = FastAPIClient()
    payload = authenticate_service_account(client)
    store_token(session, payload)

    logger.debug("Token de servicio renovado para la sesion %s", session.session_key)
    return FastAPIClient(token=payload["access_token"])


# ---------------------------------------------------------------------------
# Historial de escaneos (solo metadatos, nunca la imagen)
# ---------------------------------------------------------------------------
def record_scan(session: Any, result: Mapping[str, Any]) -> None:
    """Agrega el resumen de un escaneo al historial de la sesion."""
    from django.utils import timezone

    fields = result.get("fields") or {}
    mrz = result.get("mrz") or {}

    entry = {
        "scanned_at": timezone.localtime().strftime("%d/%m/%Y %H:%M"),
        "document_number": result.get("document_number"),
        "name": result.get("name"),
        "document_type": result.get("document_type"),
        "valid_photo": bool(result.get("valid_photo")),
        "mrz_present": bool(mrz.get("detected")),
        "mrz_valid": bool(mrz.get("valid")),
        "confidence": result.get("confidence"),
        "birth_date": fields.get("birth_date"),
    }

    history = list(session.get(HISTORY_SESSION_KEY, []))
    history.insert(0, entry)
    session[HISTORY_SESSION_KEY] = history[:MAX_HISTORY_ENTRIES]


__all__ = [
    "ApiError",
    "FastAPIClient",
    "HISTORY_SESSION_KEY",
    "TOKEN_EXPIRY_SESSION_KEY",
    "TOKEN_SESSION_KEY",
    "authenticate_service_account",
    "clear_token",
    "get_service_client",
    "record_scan",
    "session_token_is_valid",
    "store_token",
]
