"""Vistas de la app del escaner."""

from __future__ import annotations

from django.http import HttpRequest, HttpResponse
from django.shortcuts import render


def index(request: HttpRequest) -> HttpResponse:
    """Pagina principal: presenta el sistema y el punto de entrada al escaneo."""
    return render(request, "scanner/index.html", {"active": "scanner"})
