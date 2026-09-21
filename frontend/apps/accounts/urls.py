"""URLs de la app de cuentas."""

from django.urls import path

from apps.accounts import views

app_name = "accounts"

urlpatterns = [
    path("ingresar/", views.AppLoginView.as_view(), name="login"),
    path("salir/", views.logout_view, name="logout"),
    path("registro/", views.register_view, name="register"),
    path("perfil/", views.profile_view, name="profile"),
]
