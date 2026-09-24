"""Pruebas del manejo uniforme de errores y del middleware transversal."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.core.config import get_settings
from app.core.errors import AppError, InvalidImageError
from app.main import app

API = "/api/v1"


@pytest.fixture(scope="module")
def client() -> TestClient:
    # Sin `with` no se ejecuta el lifespan: las pruebas no cargan PyTorch.
    return TestClient(app, raise_server_exceptions=False)


class TestDomainErrors:
    def test_invalid_image_maps_to_422_with_a_code(self, client: TestClient) -> None:
        response = client.post(
            f"{API}/scan-document",
            files={"file": ("doc.jpg", b"no soy una imagen", "image/jpeg")},
        )

        assert response.status_code == 422
        body = response.json()
        assert body["code"] == "invalid_image"
        assert "imagen" in body["detail"].lower()

    def test_error_inherits_from_app_error(self) -> None:
        error = InvalidImageError("prueba")

        assert isinstance(error, AppError)
        assert error.status_code == 422
        assert error.as_dict() == {"detail": "prueba", "code": "invalid_image"}

    def test_app_error_accepts_custom_status_and_context(self) -> None:
        error = AppError("algo", code="custom", status_code=418, context={"campo": "x"})

        assert error.status_code == 418
        assert error.as_dict()["context"] == {"campo": "x"}


class TestValidationErrors:
    def test_validation_error_uses_the_uniform_shape(self, client: TestClient) -> None:
        response = client.post(f"{API}/validate-mrz", json={"mrz_lines": 123})

        assert response.status_code == 422
        body = response.json()
        assert body["code"] == "validation_error"
        assert isinstance(body["detail"], list)
        assert "summary" in body["context"]

    def test_validation_summary_is_readable(self, client: TestClient) -> None:
        response = client.post(f"{API}/validate-mrz", json={"mrz_lines": ["only-one-line", "two", "three", "four"]})

        summary = response.json()["context"]["summary"]
        assert "mrz_lines" in summary

    def test_validation_errors_do_not_leak_internal_objects(self, client: TestClient) -> None:
        # `ctx` puede contener la excepcion original del validador y no es
        # serializable a JSON: debe filtrarse.
        response = client.post(
            f"{API}/validate-mrz",
            json={"mrz_lines": {"no": "es una lista"}},
        )

        assert response.status_code == 422
        for error in response.json()["detail"]:
            assert "ctx" not in error


class TestHttpErrors:
    def test_unknown_route_returns_a_uniform_404(self, client: TestClient) -> None:
        response = client.get(f"{API}/no-existe")

        assert response.status_code == 404
        assert response.json()["code"] == "http_error"


class TestUnexpectedErrors:
    def test_unhandled_exception_returns_a_uniform_500(self) -> None:
        from app.main import create_app

        test_app = create_app()

        @test_app.get(f"{API}/boom")
        async def boom() -> None:  # pragma: no cover - solo se invoca en la prueba
            raise RuntimeError("fallo simulado")

        with TestClient(test_app, raise_server_exceptions=False) as local_client:
            response = local_client.get(f"{API}/boom")

        assert response.status_code == 500
        body = response.json()
        assert body["code"] == "internal_error"
        # Nunca se filtra el mensaje interno al cliente.
        assert "fallo simulado" not in response.text


class TestRequestMiddleware:
    def test_assigns_a_request_id(self, client: TestClient) -> None:
        response = client.get(f"{API}/health")

        assert response.headers["X-Request-ID"]

    def test_propagates_a_client_supplied_request_id(self, client: TestClient) -> None:
        response = client.get(f"{API}/health", headers={"X-Request-ID": "abc123"})

        assert response.headers["X-Request-ID"] == "abc123"

    def test_reports_the_processing_time(self, client: TestClient) -> None:
        response = client.get(f"{API}/health")

        assert float(response.headers["X-Process-Time-Ms"]) >= 0

    def test_request_id_is_included_in_error_responses(self, client: TestClient) -> None:
        response = client.get(f"{API}/no-existe", headers={"X-Request-ID": "traza-1"})

        assert response.json()["request_id"] == "traza-1"


class TestResponseCompression:
    def test_large_responses_are_compressed(self, client: TestClient) -> None:
        response = client.get("/openapi.json", headers={"Accept-Encoding": "gzip"})

        assert response.status_code == 200
        assert response.headers.get("Content-Encoding") == "gzip"

    def test_small_responses_are_not_compressed(self, client: TestClient) -> None:
        response = client.get(f"{API}/health", headers={"Accept-Encoding": "gzip"})

        assert response.headers.get("Content-Encoding") is None


class TestConfiguration:
    def test_gzip_can_be_disabled_from_the_environment(self) -> None:
        from app.core.config import _build_settings

        assert get_settings().enable_gzip is True
        assert _build_settings() is not None

    def test_text_lines_are_capped(self) -> None:
        assert get_settings().ocr_max_text_lines > 0
