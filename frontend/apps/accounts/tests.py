"""Pruebas del modulo de autenticacion del frontend."""

from __future__ import annotations

from django.contrib.auth import get_user_model
from django.test import TestCase
from django.urls import reverse

User = get_user_model()


class AccessControlTests(TestCase):
    """El escaner debe exigir una sesion iniciada."""

    def test_index_redirects_anonymous_user_to_login(self) -> None:
        response = self.client.get(reverse("scanner:index"))
        self.assertEqual(response.status_code, 302)
        self.assertIn(reverse("accounts:login"), response["Location"])

    def test_history_redirects_anonymous_user_to_login(self) -> None:
        response = self.client.get(reverse("scanner:history"))
        self.assertEqual(response.status_code, 302)
        self.assertIn(reverse("accounts:login"), response["Location"])

    def test_index_renders_for_authenticated_user(self) -> None:
        User.objects.create_user(username="ana@escaneo.com", email="ana@escaneo.com", password="Segura123")
        self.assertTrue(self.client.login(username="ana@escaneo.com", password="Segura123"))

        response = self.client.get(reverse("scanner:index"))

        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, "scanner/index.html")


class LoginViewTests(TestCase):
    def setUp(self) -> None:
        self.password = "Segura123"
        self.user = User.objects.create_user(
            username="ana@escaneo.com",
            email="ana@escaneo.com",
            password=self.password,
            first_name="Ana",
            last_name="Gomez",
        )

    def test_login_page_renders(self) -> None:
        response = self.client.get(reverse("accounts:login"))
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, "accounts/login.html")

    def test_login_with_valid_credentials(self) -> None:
        response = self.client.post(
            reverse("accounts:login"),
            {"username": "ana@escaneo.com", "password": self.password},
        )
        self.assertRedirects(response, reverse("scanner:index"))

    def test_login_accepts_uppercase_email(self) -> None:
        response = self.client.post(
            reverse("accounts:login"),
            {"username": "ANA@escaneo.com", "password": self.password},
        )
        self.assertRedirects(response, reverse("scanner:index"))

    def test_login_with_wrong_password_shows_error(self) -> None:
        response = self.client.post(
            reverse("accounts:login"),
            {"username": "ana@escaneo.com", "password": "incorrecta"},
        )
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Correo o contrasena incorrectos")


class RegisterViewTests(TestCase):
    def test_register_creates_user_and_starts_session(self) -> None:
        response = self.client.post(
            reverse("accounts:register"),
            {
                "email": "nuevo@escaneo.com",
                "first_name": "Nuevo",
                "last_name": "Usuario",
                "password1": "ClaveSegura123",
                "password2": "ClaveSegura123",
            },
        )
        self.assertRedirects(response, reverse("scanner:index"))

        user = User.objects.get(username="nuevo@escaneo.com")
        self.assertEqual(user.email, "nuevo@escaneo.com")
        self.assertEqual(user.first_name, "Nuevo")

    def test_register_rejects_duplicated_email(self) -> None:
        User.objects.create_user(username="dup@escaneo.com", email="dup@escaneo.com", password="Segura123")

        response = self.client.post(
            reverse("accounts:register"),
            {
                "email": "DUP@escaneo.com",
                "first_name": "Otra",
                "last_name": "Persona",
                "password1": "ClaveSegura123",
                "password2": "ClaveSegura123",
            },
        )

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Ya existe una cuenta con este correo electronico")
        self.assertEqual(User.objects.filter(email__iexact="dup@escaneo.com").count(), 1)


class LogoutViewTests(TestCase):
    def setUp(self) -> None:
        User.objects.create_user(username="ana@escaneo.com", email="ana@escaneo.com", password="Segura123")
        self.client.login(username="ana@escaneo.com", password="Segura123")

    def test_logout_requires_post(self) -> None:
        response = self.client.get(reverse("accounts:logout"))
        self.assertEqual(response.status_code, 405)

    def test_logout_closes_session(self) -> None:
        response = self.client.post(reverse("accounts:logout"))
        self.assertRedirects(response, reverse("accounts:login"))

        follow_up = self.client.get(reverse("scanner:index"))
        self.assertEqual(follow_up.status_code, 302)
