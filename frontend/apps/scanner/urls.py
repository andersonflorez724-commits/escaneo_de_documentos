"""URLs de la app del escaner."""

from django.urls import path

from apps.scanner import views

app_name = "scanner"

urlpatterns = [
    path("", views.index, name="index"),
    path("api/escanear/", views.scan_proxy, name="scan"),
    path("api/validar-mrz/", views.validate_mrz_proxy, name="validate-mrz"),
]
