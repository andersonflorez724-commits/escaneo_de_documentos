"""Pruebas de la interfaz del escaner."""

from __future__ import annotations

from django.contrib.auth import get_user_model
from django.test import TestCase
from django.urls import reverse

User = get_user_model()

PASSWORD = "Segura123"


class ScannerPageTests(TestCase):
    def setUp(self) -> None:
        User.objects.create_user(username="ana@escaneo.com", email="ana@escaneo.com", password=PASSWORD)
        self.client.login(username="ana@escaneo.com", password=PASSWORD)

    def test_page_requires_login(self) -> None:
        self.client.logout()
        response = self.client.get(reverse("scanner:index"))

        self.assertEqual(response.status_code, 302)
        self.assertIn(reverse("accounts:login"), response["Location"])

    def test_page_renders_the_camera_viewport(self) -> None:
        response = self.client.get(reverse("scanner:index"))

        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, "scanner/index.html")
        self.assertContains(response, 'id="camera-video"')
        self.assertContains(response, 'id="btn-capture"')
        self.assertContains(response, 'id="btn-process"')

    def test_page_loads_the_camera_script_as_a_module(self) -> None:
        response = self.client.get(reverse("scanner:index"))

        self.assertContains(response, "js/scanner.js")
        self.assertContains(response, 'type="module"')

    def test_page_exposes_the_scan_endpoint_to_javascript(self) -> None:
        response = self.client.get(reverse("scanner:index"))

        self.assertContains(response, f'data-scan-url="{reverse("scanner:scan")}"')

    def test_page_sets_the_csrf_cookie_for_fetch_requests(self) -> None:
        response = self.client.get(reverse("scanner:index"))

        self.assertIn("csrftoken", response.cookies)

    def test_page_is_not_cached(self) -> None:
        response = self.client.get(reverse("scanner:index"))

        self.assertIn("no-cache", response["Cache-Control"])

    def test_page_shows_the_upload_limit(self) -> None:
        response = self.client.get(reverse("scanner:index"))
        self.assertContains(response, "data-max-upload-mb")


class HistoryPageTests(TestCase):
    def setUp(self) -> None:
        User.objects.create_user(username="ana@escaneo.com", email="ana@escaneo.com", password=PASSWORD)
        self.client.login(username="ana@escaneo.com", password=PASSWORD)

    def test_requires_login(self) -> None:
        self.client.logout()
        response = self.client.get(reverse("scanner:history"))
        self.assertEqual(response.status_code, 302)

    def test_renders_empty_state(self) -> None:
        response = self.client.get(reverse("scanner:history"))

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Todavia no has escaneado")

    def test_renders_the_scans_stored_in_the_session(self) -> None:
        session = self.client.session
        session["scan_history"] = [
            {
                "scanned_at": "21/09/2026 10:30",
                "document_number": "12345678",
                "name": "JUAN PEREZ",
                "document_type": "CEDULA DE CIUDADANIA",
                "valid_photo": True,
                "mrz_valid": True,
                "mrz_present": True,
            }
        ]
        session.save()

        response = self.client.get(reverse("scanner:history"))

        self.assertContains(response, "12345678")
        self.assertContains(response, "JUAN PEREZ")


class ScanProxyAccessTests(TestCase):
    def setUp(self) -> None:
        User.objects.create_user(username="ana@escaneo.com", email="ana@escaneo.com", password=PASSWORD)
        self.client.login(username="ana@escaneo.com", password=PASSWORD)

    def test_requires_login(self) -> None:
        self.client.logout()
        response = self.client.post(reverse("scanner:scan"))

        self.assertEqual(response.status_code, 302)
        self.assertIn(reverse("accounts:login"), response["Location"])

    def test_rejects_get_requests(self) -> None:
        self.assertEqual(self.client.get(reverse("scanner:scan")).status_code, 405)

    def test_answers_json(self) -> None:
        response = self.client.post(reverse("scanner:scan"))

        self.assertEqual(response["Content-Type"], "application/json")
