"""URLs de la app del escaner."""

from django.urls import path

from apps.scanner import views

app_name = "scanner"

urlpatterns = [
    path("", views.index, name="index"),
    path("historial/", views.history, name="history"),
    path("api/escanear/", views.scan_proxy, name="scan"),
]
