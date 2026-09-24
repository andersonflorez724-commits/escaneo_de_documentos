"""Pruebas de la interfaz del escaner."""

from __future__ import annotations

from django.test import SimpleTestCase
from django.urls import reverse


class ScannerPageTests(SimpleTestCase):
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

    def test_page_offers_copying_the_extracted_data(self) -> None:
        response = self.client.get(reverse("scanner:index"))
        self.assertContains(response, 'id="btn-copy"')

    def test_page_documents_the_keyboard_shortcuts(self) -> None:
        response = self.client.get(reverse("scanner:index"))
        self.assertContains(response, "<kbd>Espacio</kbd>")
        self.assertContains(response, "<kbd>Esc</kbd>")

    def test_page_includes_the_toast_region(self) -> None:
        response = self.client.get(reverse("scanner:index"))
        self.assertContains(response, 'id="toast"')

    def test_page_marks_the_results_region_for_screen_readers(self) -> None:
        response = self.client.get(reverse("scanner:index"))
        self.assertContains(response, 'aria-live="polite"')


class ScanProxyAccessTests(SimpleTestCase):
    def test_rejects_get_requests(self) -> None:
        self.assertEqual(self.client.get(reverse("scanner:scan")).status_code, 405)

    def test_answers_json(self) -> None:
        response = self.client.post(reverse("scanner:scan"))

        self.assertEqual(response["Content-Type"], "application/json")
