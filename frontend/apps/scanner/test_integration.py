"""Pruebas de la integracion HTTP entre el cliente Django y FastAPI.

FastAPI no se levanta: se sustituye el cliente por un doble para comprobar la
traduccion de errores, el guardado del historial y el puente de sesion/JWT.
"""

from __future__ import annotations

from typing import Any
from unittest import mock

from django.contrib.auth import get_user_model
from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import TestCase
from django.urls import reverse

from apps.scanner.api_client import (
    ApiError,
    clear_token,
    record_scan,
    session_token_is_valid,
    store_token,
)

User = get_user_model()
PASSWORD = "Segura123"

SCAN_PAYLOAD: dict[str, Any] = {
    "success": True,
    "message": "Documento leido y validado correctamente.",
    "document_number": "12345678",
    "name": "JUAN PEREZ",
    "valid_photo": True,
    "document_type": "CEDULA DE CIUDADANIA",
    "confidence": 0.91,
    "fields": {"birth_date": "1974-08-12", "expiry_date": "2022-04-15"},
    "mrz": {
        "detected": True,
        "format": "TD3",
        "valid": True,
        "passed_checks": 5,
        "total_checks": 5,
        "checks": {},
    },
    "face": {"detected": True, "valid_photo": True},
    "quality": {"score": 0.8},
    "engine": "easyocr",
    "processing_ms": 1234.5,
    "text_lines": ["JUAN PEREZ"],
    "warnings": [],
    "preview_base64": None,
}


class FakeClient:
    """Doble del cliente de FastAPI."""

    def __init__(self, *, result: dict[str, Any] | None = None, error: ApiError | None = None) -> None:
        self.result = result if result is not None else SCAN_PAYLOAD
        self.error = error
        self.calls: list[dict[str, Any]] = []

    def scan_document(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(kwargs)
        if self.error:
            raise self.error
        return self.result


def image_upload(name: str = "documento.jpg", content_type: str = "image/jpeg", size: int = 1024) -> SimpleUploadedFile:
    return SimpleUploadedFile(name, b"\xff\xd8\xff" + b"0" * size, content_type=content_type)


class ScanProxyTests(TestCase):
    def setUp(self) -> None:
        User.objects.create_user(username="ana@escaneo.com", email="ana@escaneo.com", password=PASSWORD)
        self.client.login(username="ana@escaneo.com", password=PASSWORD)
        self.url = reverse("scanner:scan")

    def patch_client(self, client: Any) -> Any:
        return mock.patch("apps.scanner.views.get_service_client", return_value=client)

    # ----------------------------- validaciones ---------------------------
    def test_rejects_a_request_without_file(self) -> None:
        response = self.client.post(self.url)

        self.assertEqual(response.status_code, 400)
        self.assertIn("ninguna imagen", response.json()["detail"])

    def test_rejects_an_unsupported_content_type(self) -> None:
        with self.patch_client(FakeClient()):
            response = self.client.post(self.url, {"file": image_upload("doc.txt", "text/plain")})

        self.assertEqual(response.status_code, 415)

    def test_rejects_a_file_over_the_size_limit(self) -> None:
        oversized = image_upload(size=9 * 1024 * 1024)

        with self.patch_client(FakeClient()):
            response = self.client.post(self.url, {"file": oversized})

        self.assertEqual(response.status_code, 413)

    # ------------------------------- exito --------------------------------
    def test_returns_the_payload_from_fastapi(self) -> None:
        fake = FakeClient()

        with self.patch_client(fake):
            response = self.client.post(self.url, {"file": image_upload()})

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["document_number"], "12345678")
        self.assertEqual(body["name"], "JUAN PEREZ")
        self.assertTrue(body["valid_photo"])

    def test_forwards_the_image_and_flags_to_fastapi(self) -> None:
        fake = FakeClient()

        with self.patch_client(fake):
            self.client.post(self.url, {"file": image_upload("foto.png", "image/png"), "include_preview": "true"})

        call = fake.calls[0]
        self.assertEqual(call["filename"], "foto.png")
        self.assertEqual(call["content_type"], "image/png")
        self.assertTrue(call["include_preview"])
        self.assertTrue(call["content"].startswith(b"\xff\xd8\xff"))

    def test_include_preview_is_false_by_default(self) -> None:
        fake = FakeClient()

        with self.patch_client(fake):
            self.client.post(self.url, {"file": image_upload()})

        self.assertFalse(fake.calls[0]["include_preview"])

    def test_records_the_scan_in_the_session_history(self) -> None:
        with self.patch_client(FakeClient()):
            self.client.post(self.url, {"file": image_upload()})

        history = self.client.session.get("scan_history")
        self.assertEqual(len(history), 1)
        self.assertEqual(history[0]["document_number"], "12345678")
        self.assertEqual(history[0]["name"], "JUAN PEREZ")
        self.assertTrue(history[0]["mrz_valid"])
        # La imagen nunca se guarda en la sesion.
        self.assertNotIn("preview_base64", history[0])

    # ------------------------------- errores ------------------------------
    def test_maps_backend_validation_errors(self) -> None:
        error = ApiError("La imagen no se pudo decodificar.", status_code=422)

        with self.patch_client(FakeClient(error=error)):
            response = self.client.post(self.url, {"file": image_upload()})

        self.assertEqual(response.status_code, 422)
        self.assertIn("decodificar", response.json()["detail"])

    def test_returns_502_when_the_backend_fails(self) -> None:
        error = ApiError("Error interno.", status_code=500)

        with self.patch_client(FakeClient(error=error)):
            response = self.client.post(self.url, {"file": image_upload()})

        self.assertEqual(response.status_code, 502)

    def test_returns_503_when_the_service_account_fails(self) -> None:
        error = ApiError("Sin cuenta de servicio.", status_code=503)

        with mock.patch("apps.scanner.views.get_service_client", side_effect=error):
            response = self.client.post(self.url, {"file": image_upload()})

        self.assertEqual(response.status_code, 503)

    def test_returns_504_on_backend_timeout(self) -> None:
        error = ApiError("Tardo demasiado.", status_code=504)

        with self.patch_client(FakeClient(error=error)):
            response = self.client.post(self.url, {"file": image_upload()})

        self.assertEqual(response.status_code, 504)


class ValidateMrzProxyTests(TestCase):
    def setUp(self) -> None:
        User.objects.create_user(username="ana@escaneo.com", email="ana@escaneo.com", password=PASSWORD)
        self.client.login(username="ana@escaneo.com", password=PASSWORD)
        self.url = reverse("scanner:validate-mrz")

    def test_requires_login(self) -> None:
        self.client.logout()
        self.assertEqual(self.client.post(self.url).status_code, 302)

    def test_rejects_invalid_json(self) -> None:
        response = self.client.post(self.url, data="{no json", content_type="application/json")
        self.assertEqual(response.status_code, 400)

    def test_rejects_a_non_object_payload(self) -> None:
        response = self.client.post(self.url, data="[1, 2]", content_type="application/json")
        self.assertEqual(response.status_code, 400)

    def test_returns_the_validation_result(self) -> None:
        class FakeMRZClient:
            def validate_mrz(self, mrz_lines: list[str]) -> dict[str, Any]:
                return {"valid": True, "message": "MRZ consistente.", "mrz": {"format": "TD3"}}

        with mock.patch("apps.scanner.views.get_service_client", return_value=FakeMRZClient()):
            response = self.client.post(
                self.url,
                data='{"mrz_lines": ["A", "B"]}',
                content_type="application/json",
            )

        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.json()["valid"])

    def test_maps_backend_errors(self) -> None:
        class FailingClient:
            def validate_mrz(self, mrz_lines: list[str]) -> dict[str, Any]:
                raise ApiError("MRZ invalida.", status_code=422)

        with mock.patch("apps.scanner.views.get_service_client", return_value=FailingClient()):
            response = self.client.post(self.url, data="{}", content_type="application/json")

        self.assertEqual(response.status_code, 422)


class TokenSessionTests(TestCase):
    """El JWT de la API vive en la sesion de Django, no en el navegador."""

    def setUp(self) -> None:
        self.session = self.client.session

    def test_token_is_stored_with_its_expiry(self) -> None:
        store_token(self.session, {"access_token": "abc123", "expires_in": 3600})

        self.assertEqual(self.session["fastapi_token"], "abc123")
        self.assertTrue(session_token_is_valid(self.session))

    def test_expired_token_is_rejected(self) -> None:
        store_token(self.session, {"access_token": "abc123", "expires_in": 0})

        self.assertFalse(session_token_is_valid(self.session))

    def test_missing_token_is_rejected(self) -> None:
        self.assertFalse(session_token_is_valid(self.session))

    def test_clear_token_removes_the_credentials(self) -> None:
        store_token(self.session, {"access_token": "abc123", "expires_in": 3600})
        clear_token(self.session)

        self.assertFalse(session_token_is_valid(self.session))
        self.assertNotIn("fastapi_token", self.session)


class HistoryRecordingTests(TestCase):
    def setUp(self) -> None:
        self.session = self.client.session

    def test_keeps_the_most_recent_first(self) -> None:
        for index in range(3):
            record_scan(self.session, {**SCAN_PAYLOAD, "document_number": f"000{index}"})

        history = self.session["scan_history"]
        self.assertEqual(history[0]["document_number"], "0002")
        self.assertEqual(len(history), 3)

    def test_caps_the_history_length(self) -> None:
        for index in range(30):
            record_scan(self.session, {**SCAN_PAYLOAD, "document_number": f"{index:08d}"})

        self.assertEqual(len(self.session["scan_history"]), 20)

    def test_handles_a_payload_without_optional_blocks(self) -> None:
        record_scan(self.session, {"document_number": None, "name": None})

        entry = self.session["scan_history"][0]
        self.assertFalse(entry["mrz_present"])
        self.assertFalse(entry["valid_photo"])


class FastAPIClientTests(TestCase):
    """Comprobaciones del cliente HTTP sin levantar el backend."""

    def test_timeout_is_reported_as_504(self) -> None:
        import requests

        from apps.scanner.api_client import FastAPIClient

        with mock.patch("apps.scanner.api_client.requests.request", side_effect=requests.Timeout):
            with self.assertRaises(ApiError) as ctx:
                FastAPIClient().health()

        self.assertEqual(ctx.exception.status_code, 504)

    def test_connection_error_is_reported_as_503(self) -> None:
        import requests

        from apps.scanner.api_client import FastAPIClient

        with mock.patch(
            "apps.scanner.api_client.requests.request",
            side_effect=requests.ConnectionError("caido"),
        ):
            with self.assertRaises(ApiError) as ctx:
                FastAPIClient().health()

        self.assertEqual(ctx.exception.status_code, 503)

    def test_backend_detail_message_is_propagated(self) -> None:
        from apps.scanner.api_client import FastAPIClient

        response = mock.Mock()
        response.status_code = 422
        response.content = b'{"detail": "La imagen esta vacia."}'
        response.json.return_value = {"detail": "La imagen esta vacia."}

        with mock.patch("apps.scanner.api_client.requests.request", return_value=response):
            with self.assertRaises(ApiError) as ctx:
                FastAPIClient().health()

        self.assertEqual(ctx.exception.message, "La imagen esta vacia.")
        self.assertEqual(ctx.exception.http_status, 422)

    def test_pydantic_validation_errors_become_a_readable_message(self) -> None:
        from apps.scanner.api_client import FastAPIClient

        response = mock.Mock()
        response.status_code = 422
        response.content = b"{}"
        response.json.return_value = {"detail": [{"msg": "value is not a valid email address"}]}

        with mock.patch("apps.scanner.api_client.requests.request", return_value=response):
            with self.assertRaises(ApiError) as ctx:
                FastAPIClient().health()

        self.assertIn("valid email", ctx.exception.message)

    def test_authorization_header_is_used_when_a_token_is_present(self) -> None:
        from apps.scanner.api_client import FastAPIClient

        response = mock.Mock()
        response.status_code = 200
        response.content = b'{"status": "ok"}'
        response.json.return_value = {"status": "ok"}

        with mock.patch("apps.scanner.api_client.requests.request", return_value=response) as patched:
            FastAPIClient(token="jwt-de-prueba").health()

        headers = patched.call_args.kwargs["headers"]
        self.assertEqual(headers["Authorization"], "Bearer jwt-de-prueba")

    def test_service_account_is_registered_when_missing(self) -> None:
        from apps.scanner import api_client

        client = mock.Mock()
        client.login.side_effect = [
            ApiError("Credenciales incorrectas.", status_code=401),
            {"access_token": "nuevo", "expires_in": 3600},
        ]

        with mock.patch.object(api_client.settings, "FASTAPI_SERVICE_EMAIL", "svc@escaneo.com"):
            with mock.patch.object(api_client.settings, "FASTAPI_SERVICE_PASSWORD", "Segura123"):
                payload = api_client.authenticate_service_account(client)

        self.assertEqual(payload["access_token"], "nuevo")
        client.register.assert_called_once()


class ResponseHelperTests(TestCase):
    def test_image_upload_helper_builds_a_valid_file(self) -> None:
        upload = image_upload()

        self.assertEqual(upload.content_type, "image/jpeg")
        self.assertGreater(upload.size, 0)
        self.assertTrue(callable(getattr(upload, "read", None)))
        self.assertTrue(upload.read().startswith(b"\xff\xd8\xff"))
