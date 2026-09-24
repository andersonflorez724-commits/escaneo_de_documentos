"""Pruebas de los endpoints HTTP de la API."""

from __future__ import annotations

import dataclasses
from typing import Iterator

import pytest
from fastapi.testclient import TestClient

from app.core.config import get_settings
from app.main import app
from app.services.face_detector import HeuristicFaceDetector
from app.services.scanner import DocumentScanner, get_document_scanner
from tests.fixtures import ICAO_TD3_LINES, build_td3_mrz, document_bytes
from tests.stubs import StubOCREngine

API = "/api/v1"

# Texto que simularia el OCR: campos impresos + una MRZ coherente con ellos.
SCAN_TEXTS = [
    "REPUBLICA DE COLOMBIA",
    "CEDULA DE CIUDADANIA",
    "NUMERO",
    "CC 12345678",
    "NOMBRES",
    "JUAN PEREZ",
    "FECHA DE NACIMIENTO",
    "12/08/1974",
    "SEXO M",
    "FECHA DE VENCIMIENTO",
    "15/04/2022",
    *build_td3_mrz(),
]


@pytest.fixture(scope="module")
def client() -> Iterator[TestClient]:
    """Cliente de pruebas sin lifespan: no carga PyTorch."""
    with TestClient(app, raise_server_exceptions=False) as test_client:
        # El contexto si dispara el lifespan, por eso se usa el motor simulado
        # a traves de la sustitucion de dependencias.
        yield test_client


@pytest.fixture(scope="module", autouse=True)
def stub_scanner() -> Iterator[None]:
    """Sustituye el escaner real por uno deterministico y rapido."""
    scanner = DocumentScanner(
        engine=StubOCREngine(SCAN_TEXTS),
        face_detector=HeuristicFaceDetector(),
        settings=get_settings(),
    )
    app.dependency_overrides[get_document_scanner] = lambda: scanner
    yield
    app.dependency_overrides.clear()


# ---------------------------------------------------------------------------
# Sistema
# ---------------------------------------------------------------------------
class TestSystemEndpoints:
    def test_health(self, client: TestClient) -> None:
        response = client.get(f"{API}/health")

        assert response.status_code == 200
        assert response.json()["status"] == "ok"
        assert response.json()["version"]

    def test_openapi_schema_documents_every_endpoint(self, client: TestClient) -> None:
        schema = client.get("/openapi.json").json()

        assert set(schema["paths"]) == {
            f"{API}/health",
            f"{API}/scan-document",
            f"{API}/validate-mrz",
        }

    def test_openapi_groups_endpoints_by_tag(self, client: TestClient) -> None:
        schema = client.get("/openapi.json").json()
        tags = {tag["name"] for tag in schema["tags"]}

        assert {"Sistema", "Documentos"} <= tags

    def test_swagger_ui_is_served(self, client: TestClient) -> None:
        response = client.get("/docs")

        assert response.status_code == 200
        assert "swagger-ui" in response.text.lower()

    def test_redoc_is_served(self, client: TestClient) -> None:
        assert client.get("/redoc").status_code == 200

    def test_openapi_declares_no_security_scheme(self, client: TestClient) -> None:
        schema = client.get("/openapi.json").json()
        assert "securitySchemes" not in schema.get("components", {})


# ---------------------------------------------------------------------------
# Escaneo de documentos
# ---------------------------------------------------------------------------
class TestScanDocumentEndpoint:
    def test_extracts_the_required_fields(self, client: TestClient) -> None:
        response = client.post(
            f"{API}/scan-document",
            files={"file": ("documento.jpg", document_bytes(), "image/jpeg")},
        )

        assert response.status_code == 200, response.text
        body = response.json()

        # Contrato minimo exigido por el taller.
        assert body["document_number"] == "12345678"
        assert body["name"] == "JUAN PEREZ"
        assert body["valid_photo"] is True

        assert body["success"] is True
        assert 0.0 <= body["confidence"] <= 1.0
        assert body["engine"] == "stub"

    def test_reports_the_mrz_analysis(self, client: TestClient) -> None:
        response = client.post(
            f"{API}/scan-document",
            files={"file": ("documento.jpg", document_bytes(), "image/jpeg")},
        )
        mrz = response.json()["mrz"]

        assert mrz["detected"] is True
        assert mrz["format"] == "TD3"
        assert mrz["valid"] is True
        assert mrz["checks"]["document_number"] is True
        assert mrz["total_checks"] == mrz["passed_checks"]

    def test_accepts_the_back_side_of_the_document(self, client: TestClient) -> None:
        response = client.post(
            f"{API}/scan-document",
            files={
                "file": ("documento-frontal.jpg", document_bytes(), "image/jpeg"),
                "back_file": ("documento-reverso.jpg", document_bytes(), "image/jpeg"),
            },
        )

        assert response.status_code == 200, response.text
        body = response.json()

        assert body["sides_processed"] == 2
        assert body["both_sides"] is True
        assert [side["side"] for side in body["sides"]] == ["front", "back"]
        assert [side["label"] for side in body["sides"]] == ["cara frontal", "reverso"]
        # Los campos extraidos siguen siendo los del contrato minimo.
        assert body["document_number"] == "12345678"
        assert body["name"] == "JUAN PEREZ"

    def test_keeps_the_back_side_optional(self, client: TestClient) -> None:
        response = client.post(
            f"{API}/scan-document",
            files={"file": ("documento-frontal.jpg", document_bytes(), "image/jpeg")},
        )

        assert response.status_code == 200
        body = response.json()
        assert body["sides_processed"] == 1
        assert body["both_sides"] is False
        assert "reverso" in body["message"]

    def test_rejects_an_unsupported_back_side(self, client: TestClient) -> None:
        response = client.post(
            f"{API}/scan-document",
            files={
                "file": ("documento-frontal.jpg", document_bytes(), "image/jpeg"),
                "back_file": ("reverso.txt", b"hola", "text/plain"),
            },
        )

        assert response.status_code == 415

    def test_reports_the_document_fields_of_the_response(self, client: TestClient) -> None:
        response = client.post(
            f"{API}/scan-document",
            files={"file": ("documento.jpg", document_bytes(), "image/jpeg")},
        )
        fields = response.json()["fields"]

        assert fields["birth_date"] == "1974-08-12"
        assert fields["expiry_date"] == "2022-04-15"
        assert "first_surname" in fields
        assert "birth_place" in fields
        assert "blood_type" in fields

    def test_reports_the_face_and_quality_blocks(self, client: TestClient) -> None:
        response = client.post(
            f"{API}/scan-document",
            files={"file": ("documento.jpg", document_bytes(), "image/jpeg")},
        )
        body = response.json()

        assert body["face"]["detected"] is True
        assert body["face"]["valid_photo"] is True
        assert body["quality"]["width"] > 0
        assert 0 <= body["quality"]["score"] <= 1

    def test_omits_the_preview_by_default(self, client: TestClient) -> None:
        response = client.post(
            f"{API}/scan-document",
            files={"file": ("documento.jpg", document_bytes(), "image/jpeg")},
        )
        assert response.json()["preview_base64"] is None

    def test_includes_the_preview_when_requested(self, client: TestClient) -> None:
        response = client.post(
            f"{API}/scan-document?include_preview=true",
            files={"file": ("documento.jpg", document_bytes(), "image/jpeg")},
        )

        assert response.json()["preview_base64"]

    def test_rejects_a_non_image_file(self, client: TestClient) -> None:
        response = client.post(
            f"{API}/scan-document",
            files={"file": ("documento.jpg", b"esto no es una imagen", "image/jpeg")},
        )

        assert response.status_code == 422
        assert "imagen" in response.json()["detail"].lower()

    def test_rejects_an_unsupported_content_type(self, client: TestClient) -> None:
        response = client.post(
            f"{API}/scan-document",
            files={"file": ("documento.txt", b"hola", "text/plain")},
        )

        assert response.status_code == 415

    def test_rejects_an_empty_file(self, client: TestClient) -> None:
        response = client.post(
            f"{API}/scan-document",
            files={"file": ("documento.jpg", b"", "image/jpeg")},
        )

        assert response.status_code == 422

    def test_rejects_a_file_over_the_size_limit(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import app.api.v1.endpoints.documents as documents_module

        monkeypatch.setattr(
            documents_module,
            "settings",
            dataclasses.replace(get_settings(), max_upload_mb=0),
        )

        response = client.post(
            f"{API}/scan-document",
            files={"file": ("documento.jpg", document_bytes(), "image/jpeg")},
        )

        assert response.status_code == 413


# ---------------------------------------------------------------------------
# Validacion de MRZ
# ---------------------------------------------------------------------------
class TestValidateMrzEndpoint:
    def test_validates_a_consistent_passport(self, client: TestClient) -> None:
        response = client.post(
            f"{API}/validate-mrz",
            json={"mrz_lines": list(ICAO_TD3_LINES)},
        )

        assert response.status_code == 200, response.text
        body = response.json()

        assert body["valid"] is True
        assert body["inconsistent_fields"] == []
        assert body["mrz"]["format"] == "TD3"
        assert body["mrz"]["document_number"] == "L898902C3"
        assert body["mrz"]["full_name"] == "ANNA MARIA ERIKSSON"
        assert body["mrz"]["birth_date"] == "1974-08-12"
        assert "consistente" in body["message"]

    def test_detects_a_tampered_document_number(self, client: TestClient) -> None:
        tampered = [ICAO_TD3_LINES[0], ICAO_TD3_LINES[1].replace("L898902C3", "L898902C4")]

        response = client.post(f"{API}/validate-mrz", json={"mrz_lines": tampered})

        body = response.json()
        assert body["valid"] is False
        assert "document_number" in body["inconsistent_fields"]
        assert "composite" in body["inconsistent_fields"]
        assert "inconsistente" in body["message"]

    def test_extracts_the_mrz_from_raw_ocr_text(self, client: TestClient) -> None:
        text = "\n".join(["REPUBLICA DE COLOMBIA", "CEDULA", *ICAO_TD3_LINES])

        response = client.post(f"{API}/validate-mrz", json={"text": text})

        assert response.status_code == 200
        assert response.json()["valid"] is True

    def test_validates_td1_documents(self, client: TestClient) -> None:
        td1 = [
            "I<UTOD231458907<<<<<<<<<<<<<<<",
            "7408122F1204159UTO<<<<<<<<<<<6",
            "ERIKSSON<<ANNA<MARIA<<<<<<<<<<",
        ]

        response = client.post(f"{API}/validate-mrz", json={"mrz_lines": td1})

        body = response.json()
        assert body["valid"] is True
        assert body["mrz"]["format"] == "TD1"

    def test_reports_when_no_mrz_is_found(self, client: TestClient) -> None:
        response = client.post(f"{API}/validate-mrz", json={"text": "JUAN PEREZ\nBogota"})

        body = response.json()
        assert body["valid"] is False
        assert body["mrz"]["detected"] is False

    def test_requires_some_input(self, client: TestClient) -> None:
        response = client.post(f"{API}/validate-mrz", json={})

        assert response.status_code == 422
        assert "mrz_lines" in response.text

    def test_rejects_an_over_long_payload(self, client: TestClient) -> None:
        response = client.post(
            f"{API}/validate-mrz",
            json={"mrz_lines": ["A" * 40, "B" * 40, "C" * 40, "D" * 40]},
        )
        assert response.status_code == 422
