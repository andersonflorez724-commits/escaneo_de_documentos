"""Vistas de la app del escaner."""

from __future__ import annotations

import json
import logging

from django.conf import settings
from django.http import HttpRequest, HttpResponse, JsonResponse
from django.shortcuts import render
from django.views.decorators.cache import never_cache
from django.views.decorators.csrf import ensure_csrf_cookie
from django.views.decorators.http import require_POST

from apps.scanner.api_client import ApiError, get_service_client

logger = logging.getLogger(__name__)

ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/jpg", "image/png", "image/webp", "image/bmp", "image/tiff"}
TRUTHY = {"1", "true", "on", "yes", "si", "sí"}


@ensure_csrf_cookie
@never_cache
def index(request: HttpRequest) -> HttpResponse:
    """Pagina principal: captura con la camara y resultados del escaneo."""
    return render(request, "scanner/index.html", {"active": "scanner"})


def _json_error(message: str, status: int) -> JsonResponse:
    return JsonResponse({"detail": message}, status=status)


def _validate_upload(upload) -> JsonResponse | None:
    """Comprueba el tipo y el tamano de una imagen subida.

    Returns:
        La respuesta de error que se debe devolver, o ``None`` si la imagen
        es aceptable.
    """
    content_type = (upload.content_type or "").lower()
    if content_type and content_type not in ALLOWED_CONTENT_TYPES:
        return _json_error(
            f"El tipo de archivo '{upload.content_type}' no esta soportado. "
            "Envia una imagen JPEG, PNG o WEBP.",
            415,
        )

    if upload.size > settings.MAX_UPLOAD_MB * 1024 * 1024:
        return _json_error(f"La imagen supera el limite de {settings.MAX_UPLOAD_MB} MB.", 413)

    return None


@require_POST
def scan_proxy(request: HttpRequest) -> JsonResponse:
    """Recibe las fotos desde el navegador y las reenvia al backend FastAPI.

    Valida la subida y delega el analisis en FastAPI. No se guarda historial
    ni estado de sesion: cada escaneo es independiente.

    Se aceptan las dos caras del documento: la frontal en ``file`` y el reverso
    en ``back_file``. Ambas son opcionales por separado, pero al menos una debe
    llegar.
    """
    upload = request.FILES.get("file")
    if upload is None:
        return _json_error("No se recibio ninguna imagen.", 400)

    error = _validate_upload(upload)
    if error is not None:
        return error

    back_upload = request.FILES.get("back_file")
    if back_upload is not None:
        error = _validate_upload(back_upload)
        if error is not None:
            return error

    include_preview = str(request.POST.get("include_preview", "")).strip().lower() in TRUTHY

    try:
        client = get_service_client(request)
    except ApiError as exc:
        logger.error("No se pudo contactar con FastAPI: %s", exc.message)
        return _json_error(exc.message, 503)

    try:
        result = client.scan_document(
            content=upload.read(),
            filename=upload.name or "documento.jpg",
            content_type=upload.content_type or "image/jpeg",
            back_content=back_upload.read() if back_upload is not None else None,
            back_filename=(back_upload.name if back_upload is not None else "documento-reverso.jpg"),
            back_content_type=(back_upload.content_type if back_upload is not None else None) or "image/jpeg",
            include_preview=include_preview,
        )
    except ApiError as exc:
        logger.warning("Fallo el escaneo: %s", exc.message)
        return _json_error(exc.message, exc.http_status)

    logger.info(
        "Se proceso un documento (caras=%s, confianza=%s)",
        result.get("sides_processed"),
        result.get("confidence"),
    )
    return JsonResponse(result)


@require_POST
def validate_mrz_proxy(request: HttpRequest) -> JsonResponse:
    """Valida la consistencia de una MRZ a traves del backend.

    Acepta JSON con `mrz_lines` o `text`. Se expone sobre todo para poder
    probar el endpoint de la API desde el propio frontend.
    """
    try:
        payload = json.loads(request.body or b"{}")
    except json.JSONDecodeError:
        return _json_error("El cuerpo de la peticion no es JSON valido.", 400)

    if not isinstance(payload, dict):
        return _json_error("El cuerpo de la peticion debe ser un objeto JSON.", 400)

    try:
        client = get_service_client(request)
    except ApiError as exc:
        logger.error("No se pudo contactar con FastAPI: %s", exc.message)
        return _json_error(exc.message, 503)

    try:
        result = client.validate_mrz(payload.get("mrz_lines") or [])
    except ApiError as exc:
        return _json_error(exc.message, exc.http_status)

    return JsonResponse(result)
