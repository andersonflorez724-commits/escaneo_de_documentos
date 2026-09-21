"""Vistas del modulo de cuentas: login, logout, registro y perfil."""

from __future__ import annotations

from typing import Any

from django.contrib import messages
from django.contrib.auth import login as auth_login, logout as auth_logout
from django.contrib.auth.decorators import login_required
from django.contrib.auth.views import LoginView
from django.http import HttpRequest, HttpResponse
from django.shortcuts import redirect, render
from django.urls import reverse_lazy
from django.views.decorators.http import require_http_methods

from apps.accounts.forms import LoginForm, RegisterForm


class AppLoginView(LoginView):
    """Inicio de sesion con plantilla propia y mensaje de bienvenida."""

    template_name = "accounts/login.html"
    authentication_form = LoginForm
    redirect_authenticated_user = True
    next_page = reverse_lazy("scanner:index")

    def get_context_data(self, **kwargs: Any) -> dict[str, Any]:
        context = super().get_context_data(**kwargs)
        context["active"] = "login"
        return context

    def form_valid(self, form: Any) -> HttpResponse:
        response = super().form_valid(form)
        messages.success(
            self.request,
            f"Bienvenido, {self.request.user.get_full_name() or self.request.user.username}.",
        )
        return response


@require_http_methods(["POST"])
def logout_view(request: HttpRequest) -> HttpResponse:
    """Cierra la sesion del usuario (solo por POST, protegido con CSRF)."""
    if request.user.is_authenticated:
        auth_logout(request)
        messages.info(request, "Has cerrado la sesion correctamente.")
    return redirect("accounts:login")


def register_view(request: HttpRequest) -> HttpResponse:
    """Registro de una cuenta nueva y acceso inmediato."""
    if request.user.is_authenticated:
        return redirect("scanner:index")

    form = RegisterForm(request.POST or None)
    if request.method == "POST" and form.is_valid():
        user = form.save()
        auth_login(request, user)
        messages.success(request, "Cuenta creada correctamente. Ya puedes escanear documentos.")
        return redirect("scanner:index")

    return render(request, "accounts/register.html", {"form": form, "active": "register"})


@login_required
def profile_view(request: HttpRequest) -> HttpResponse:
    """Ficha del usuario con el estado de la conexion con la API."""
    return render(
        request,
        "accounts/profile.html",
        {
            "active": "profile",
            "api_token_present": bool(request.session.get("fastapi_token")),
        },
    )
