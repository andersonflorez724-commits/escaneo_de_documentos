"""Vistas de la app del escaner."""

from __future__ import annotations

import logging

from django.contrib.auth.decorators import login_required
from django.http import HttpRequest, HttpResponse, JsonResponse
from django.shortcuts import render
from django.views.decorators.cache import never_cache
from django.views.decorators.csrf import ensure_csrf_cookie
from django.views.decorators.http import require_POST

logger = logging.getLogger(__name__)


@login_required
@ensure_csrf_cookie
@never_cache
def index(request: HttpRequest) -> HttpResponse:
    """Pagina principal: captura con la camara y resultados del escaneo."""
    return render(request, "scanner/index.html", {"active": "scanner"})


@login_required
def history(request: HttpRequest) -> HttpResponse:
    """Historial de documentos escaneados en la sesion actual."""
    scans = request.session.get("scan_history", [])
    return render(
        request,
        "scanner/history.html",
        {"active": "history", "scans": scans},
    )


@login_required
@require_POST
def scan_proxy(request: HttpRequest) -> JsonResponse:
    """Recibe la foto desde el navegador y la reenvia al backend FastAPI.

    La integracion HTTP se completa en el paso siguiente.
    """
    return JsonResponse(
        {
            "detail": (
                "La integracion con el servidor de vision todavia no esta habilitada. "
                "Configura FASTAPI_BASE_URL para activarla."
            )
        },
        status=501,
    )
