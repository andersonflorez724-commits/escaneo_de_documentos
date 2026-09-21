"""Vistas de la app del escaner."""

from __future__ import annotations

import json
import logging

from django.conf import settings
from django.contrib.auth.decorators import login_required
from django.http import HttpRequest, HttpResponse, JsonResponse
from django.shortcuts import render
from django.views.decorators.cache import never_cache
from django.views.decorators.csrf import ensure_csrf_cookie
from django.views.decorators.http import require_POST

from apps.scanner.api_client import ApiError, get_service_client, record_scan

logger = logging.getLogger(__name__)

ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/jpg", "image/png", "image/webp", "image/bmp", "image/tiff"}
TRUTHY = {"1", "true", "on", "yes", "si", "sí"}


@login_required
@ensure_csrf_cookie
@never_cache
def index(request: HttpRequest) -> HttpResponse:
    """Pagina principal: captura con la camara y resultados del escaneo."""
    return render(request, "scanner/index.html", {"active": "scanner"})


@login_required
def history(request: HttpRequest) -> HttpResponse:
    """Historial de documentos escaneados en la sesion actual."""
    return render(
        request,
        "scanner/history.html",
        {"active": "history", "scans": request.session.get("scan_history", [])},
    )


def _json_error(message: str, status: int) -> JsonResponse:
    return JsonResponse({"detail": message}, status=status)


@login_required
@require_POST
def scan_proxy(request: HttpRequest) -> JsonResponse:
    """Recibe la foto desde el navegador y la reenvia al backend FastAPI.

    Hace de puente entre la sesion de Django y el JWT de la API: valida la
    subida, delega el analisis en FastAPI y guarda un resumen del escaneo en
    el historial de la sesion (nunca la imagen).
    """
    upload = request.FILES.get("file")
    if upload is None:
        return _json_error("No se recibio ninguna imagen.", 400)

    content_type = (upload.content_type or "").lower()
    if content_type and content_type not in ALLOWED_CONTENT_TYPES:
        return _json_error(
            f"El tipo de archivo '{upload.content_type}' no esta soportado. "
            "Envia una imagen JPEG, PNG o WEBP.",
            415,
        )

    max_bytes = settings.MAX_UPLOAD_MB * 1024 * 1024
    if upload.size > max_bytes:
        return _json_error(f"La imagen supera el limite de {settings.MAX_UPLOAD_MB} MB.", 413)

    include_preview = str(request.POST.get("include_preview", "")).strip().lower() in TRUTHY

    # 1. Token de la API (se reutiliza el de la sesion mientras siga vigente).
    try:
        client = get_service_client(request.session)
    except ApiError as exc:
        logger.error("No se pudo autenticar contra FastAPI: %s", exc.message)
        return _json_error(exc.message, 503)

    # 2. Analisis del documento.
    try:
        result = client.scan_document(
            content=upload.read(),
            filename=upload.name or "documento.jpg",
            content_type=upload.content_type or "image/jpeg",
            include_preview=include_preview,
        )
    except ApiError as exc:
        logger.warning("Fallo el escaneo para %s: %s", request.user.get_username(), exc.message)
        return _json_error(exc.message, exc.http_status)

    # 3. Resumen en el historial de la sesion.
    record_scan(request.session, result)

    logger.info(
        "Usuario %s proceso un documento (confianza=%s)",
        request.user.get_username(),
        result.get("confidence"),
    )
    return JsonResponse(result)


@login_required
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
        client = get_service_client(request.session)
    except ApiError as exc:
        logger.error("No se pudo autenticar contra FastAPI: %s", exc.message)
        return _json_error(exc.message, 503)

    try:
        result = client.validate_mrz(payload.get("mrz_lines") or [])
    except ApiError as exc:
        return _json_error(exc.message, exc.http_status)

    return JsonResponse(result)
