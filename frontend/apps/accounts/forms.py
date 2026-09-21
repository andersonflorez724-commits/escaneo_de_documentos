"""Formularios de autenticacion del cliente Django."""

from __future__ import annotations

from typing import Any

from django import forms
from django.contrib.auth import get_user_model
from django.contrib.auth.forms import AuthenticationForm, UserCreationForm

User = get_user_model()

_INPUT = "input"
_PLACEHOLDER_STYLE = {"class": _INPUT, "autocomplete": "off"}


class LoginForm(AuthenticationForm):
    """Login por correo electronico.

    El `username` de Django se sincroniza con el correo, de modo que el usuario
    final solo necesita recordar una credencial.
    """

    username = forms.EmailField(
        label="Correo electronico",
        widget=forms.EmailInput(
            attrs={
                "class": _INPUT,
                "autofocus": True,
                "autocomplete": "username",
                "placeholder": "tu@correo.com",
                "id": "id_username",
            }
        ),
    )
    password = forms.CharField(
        label="Contrasena",
        strip=False,
        widget=forms.PasswordInput(
            attrs={
                "class": _INPUT,
                "autocomplete": "current-password",
                "placeholder": "********",
                "id": "id_password",
            }
        ),
    )
    error_messages = {
        **AuthenticationForm.error_messages,
        "invalid_login": "Correo o contrasena incorrectos. Verifica tus datos e intentalo de nuevo.",
        "inactive": "Esta cuenta esta deshabilitada.",
    }

    def clean_username(self) -> str:
        return self.cleaned_data["username"].strip().lower()


class RegisterForm(UserCreationForm):
    """Registro de un usuario nuevo del frontend."""

    email = forms.EmailField(
        label="Correo electronico",
        widget=forms.EmailInput(attrs={**_PLACEHOLDER_STYLE, "placeholder": "tu@correo.com"}),
    )
    first_name = forms.CharField(
        label="Nombres",
        max_length=60,
        widget=forms.TextInput(attrs={**_PLACEHOLDER_STYLE, "placeholder": "Ana Maria"}),
    )
    last_name = forms.CharField(
        label="Apellidos",
        max_length=60,
        widget=forms.TextInput(attrs={**_PLACEHOLDER_STYLE, "placeholder": "Gomez Perez"}),
    )

    class Meta(UserCreationForm.Meta):
        model = User
        fields = ("email", "first_name", "last_name")

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        # El identificador de acceso es el correo, no hace falta pedir username.
        self.fields.pop("username", None)
        for name in ("password1", "password2"):
            self.fields[name].widget.attrs.update({"class": _INPUT, "placeholder": "********"})
            self.fields[name].help_text = ""

    def clean_email(self) -> str:
        email = self.cleaned_data["email"].strip().lower()
        if User.objects.filter(username__iexact=email).exists():
            raise forms.ValidationError("Ya existe una cuenta con este correo electronico.")
        return email

    def save(self, commit: bool = True) -> Any:
        user = super().save(commit=False)
        email = self.cleaned_data["email"]
        user.email = email
        user.username = email
        user.first_name = self.cleaned_data["first_name"]
        user.last_name = self.cleaned_data["last_name"]
        if commit:
            user.save()
        return user
