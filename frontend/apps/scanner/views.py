"""Vistas de la app del escaner."""

from __future__ import annotations

from django.contrib.auth.decorators import login_required
from django.http import HttpRequest, HttpResponse
from django.shortcuts import render


@login_required
def index(request: HttpRequest) -> HttpResponse:
    """Pagina principal: punto de entrada al escaneo de documentos."""
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
