"""URLs de la app de cuentas (se completan en el modulo de autenticacion)."""

from django.urls import URLPattern, path

app_name = "accounts"

urlpatterns: list[URLPattern] = []

__all__ = ["app_name", "urlpatterns", "path"]
