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
SEED_EMAIL = "admin@escaneo.com"
SEED_PASSWORD = "Admin123*"

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
    "EXPEDICION",
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


@pytest.fixture(scope="module")
def token(client: TestClient) -> str:
    response = client.post(f"{API}/auth/login", json={"email": SEED_EMAIL, "password": SEED_PASSWORD})
    assert response.status_code == 200, response.text
    return response.json()["access_token"]


@pytest.fixture(scope="module")
def auth(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


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
            f"{API}/auth/register",
            f"{API}/auth/login",
            f"{API}/auth/token",
            f"{API}/auth/me",
            f"{API}/scan-document",
            f"{API}/validate-mrz",
        }

    def test_openapi_groups_endpoints_by_tag(self, client: TestClient) -> None:
        schema = client.get("/openapi.json").json()
        tags = {tag["name"] for tag in schema["tags"]}

        assert {"Sistema", "Autenticacion", "Documentos"} <= tags

    def test_swagger_ui_is_served(self, client: TestClient) -> None:
        response = client.get("/docs")

        assert response.status_code == 200
        assert "swagger-ui" in response.text.lower()

    def test_redoc_is_served(self, client: TestClient) -> None:
        assert client.get("/redoc").status_code == 200

    def test_openapi_declares_security_scheme(self, client: TestClient) -> None:
        schema = client.get("/openapi.json").json()
        assert "OAuth2PasswordBearer" in schema["components"]["securitySchemes"]


# ---------------------------------------------------------------------------
# Autenticacion
# ---------------------------------------------------------------------------
class TestAuthEndpoints:
    def test_register_returns_created_user(self, client: TestClient) -> None:
        response = client.post(
            f"{API}/auth/register",
            json={"email": "nueva@escaneo.com", "full_name": "Nueva Usuaria", "password": "Segura123"},
        )

        assert response.status_code == 201, response.text
        body = response.json()
        assert body["email"] == "nueva@escaneo.com"
        assert body["full_name"] == "Nueva Usuaria"
        assert "password" not in body
        assert "hashed_password" not in body

    def test_register_rejects_duplicated_email(self, client: TestClient) -> None:
        payload = {"email": "repetida@escaneo.com", "full_name": "Repetida Uno", "password": "Segura123"}
        assert client.post(f"{API}/auth/register", json=payload).status_code == 201

        payload["full_name"] = "Repetida Dos"
        assert client.post(f"{API}/auth/register", json=payload).status_code == 409

    def test_register_rejects_weak_password(self, client: TestClient) -> None:
        response = client.post(
            f"{API}/auth/register",
            json={"email": "debil@escaneo.com", "full_name": "Clave Debil", "password": "solamenteletras"},
        )

        assert response.status_code == 422
        assert "numero" in response.text

    def test_register_rejects_invalid_email(self, client: TestClient) -> None:
        response = client.post(
            f"{API}/auth/register",
            json={"email": "no-es-un-correo", "full_name": "Correo Malo", "password": "Segura123"},
        )
        assert response.status_code == 422

    def test_login_returns_token(self, client: TestClient) -> None:
        response = client.post(f"{API}/auth/login", json={"email": SEED_EMAIL, "password": SEED_PASSWORD})

        assert response.status_code == 200
        body = response.json()
        assert body["token_type"] == "bearer"
        assert body["expires_in"] > 0
        assert body["user"]["email"] == SEED_EMAIL

    def test_login_is_case_insensitive_on_email(self, client: TestClient) -> None:
        response = client.post(f"{API}/auth/login", json={"email": SEED_EMAIL.upper(), "password": SEED_PASSWORD})
        assert response.status_code == 200

    def test_login_rejects_wrong_password(self, client: TestClient) -> None:
        response = client.post(f"{API}/auth/login", json={"email": SEED_EMAIL, "password": "incorrecta1"})

        assert response.status_code == 401
        assert response.headers["WWW-Authenticate"] == "Bearer"

    def test_oauth2_token_endpoint_accepts_form_data(self, client: TestClient) -> None:
        response = client.post(f"{API}/auth/token", data={"username": SEED_EMAIL, "password": SEED_PASSWORD})

        assert response.status_code == 200
        assert response.json()["access_token"]

    def test_me_returns_current_user(self, client: TestClient, auth: dict[str, str]) -> None:
        response = client.get(f"{API}/auth/me", headers=auth)

        assert response.status_code == 200
        assert response.json()["email"] == SEED_EMAIL

    def test_me_requires_token(self, client: TestClient) -> None:
        assert client.get(f"{API}/auth/me").status_code == 401

    def test_me_rejects_invalid_token(self, client: TestClient) -> None:
        response = client.get(f"{API}/auth/me", headers={"Authorization": "Bearer no-es-un-jwt"})
        assert response.status_code == 401


# ---------------------------------------------------------------------------
# Escaneo de documentos
# ---------------------------------------------------------------------------
class TestScanDocumentEndpoint:
    def test_requires_authentication(self, client: TestClient) -> None:
        response = client.post(f"{API}/scan-document", files={"file": ("doc.jpg", document_bytes(), "image/jpeg")})
        assert response.status_code == 401

    def test_extracts_the_required_fields(self, client: TestClient, auth: dict[str, str]) -> None:
        response = client.post(
            f"{API}/scan-document",
            headers=auth,
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

    def test_reports_the_mrz_analysis(self, client: TestClient, auth: dict[str, str]) -> None:
        response = client.post(
            f"{API}/scan-document",
            headers=auth,
            files={"file": ("documento.jpg", document_bytes(), "image/jpeg")},
        )
        mrz = response.json()["mrz"]

        assert mrz["detected"] is True
        assert mrz["format"] == "TD3"
        assert mrz["valid"] is True
        assert mrz["checks"]["document_number"] is True
        assert mrz["total_checks"] == mrz["passed_checks"]

    def test_reports_the_face_and_quality_blocks(self, client: TestClient, auth: dict[str, str]) -> None:
        response = client.post(
            f"{API}/scan-document",
            headers=auth,
            files={"file": ("documento.jpg", document_bytes(), "image/jpeg")},
        )
        body = response.json()

        assert body["face"]["detected"] is True
        assert body["face"]["valid_photo"] is True
        assert body["quality"]["width"] > 0
        assert 0 <= body["quality"]["score"] <= 1

    def test_omits_the_preview_by_default(self, client: TestClient, auth: dict[str, str]) -> None:
        response = client.post(
            f"{API}/scan-document",
            headers=auth,
            files={"file": ("documento.jpg", document_bytes(), "image/jpeg")},
        )
        assert response.json()["preview_base64"] is None

    def test_includes_the_preview_when_requested(self, client: TestClient, auth: dict[str, str]) -> None:
        response = client.post(
            f"{API}/scan-document?include_preview=true",
            headers=auth,
            files={"file": ("documento.jpg", document_bytes(), "image/jpeg")},
        )

        assert response.json()["preview_base64"]

    def test_rejects_a_non_image_file(self, client: TestClient, auth: dict[str, str]) -> None:
        response = client.post(
            f"{API}/scan-document",
            headers=auth,
            files={"file": ("documento.jpg", b"esto no es una imagen", "image/jpeg")},
        )

        assert response.status_code == 422
        assert "imagen" in response.json()["detail"].lower()

    def test_rejects_an_unsupported_content_type(self, client: TestClient, auth: dict[str, str]) -> None:
        response = client.post(
            f"{API}/scan-document",
            headers=auth,
            files={"file": ("documento.txt", b"hola", "text/plain")},
        )

        assert response.status_code == 415

    def test_rejects_an_empty_file(self, client: TestClient, auth: dict[str, str]) -> None:
        response = client.post(
            f"{API}/scan-document",
            headers=auth,
            files={"file": ("documento.jpg", b"", "image/jpeg")},
        )

        assert response.status_code == 422

    def test_rejects_a_file_over_the_size_limit(
        self, client: TestClient, auth: dict[str, str], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import app.api.v1.endpoints.documents as documents_module

        monkeypatch.setattr(
            documents_module,
            "settings",
            dataclasses.replace(get_settings(), max_upload_mb=0),
        )

        response = client.post(
            f"{API}/scan-document",
            headers=auth,
            files={"file": ("documento.jpg", document_bytes(), "image/jpeg")},
        )

        assert response.status_code == 413


# ---------------------------------------------------------------------------
# Validacion de MRZ
# ---------------------------------------------------------------------------
class TestValidateMrzEndpoint:
    def test_requires_authentication(self, client: TestClient) -> None:
        response = client.post(f"{API}/validate-mrz", json={"mrz_lines": list(ICAO_TD3_LINES)})
        assert response.status_code == 401

    def test_validates_a_consistent_passport(self, client: TestClient, auth: dict[str, str]) -> None:
        response = client.post(
            f"{API}/validate-mrz",
            headers=auth,
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

    def test_detects_a_tampered_document_number(self, client: TestClient, auth: dict[str, str]) -> None:
        tampered = [ICAO_TD3_LINES[0], ICAO_TD3_LINES[1].replace("L898902C3", "L898902C4")]

        response = client.post(f"{API}/validate-mrz", headers=auth, json={"mrz_lines": tampered})

        body = response.json()
        assert body["valid"] is False
        assert "document_number" in body["inconsistent_fields"]
        assert "composite" in body["inconsistent_fields"]
        assert "inconsistente" in body["message"]

    def test_extracts_the_mrz_from_raw_ocr_text(self, client: TestClient, auth: dict[str, str]) -> None:
        text = "\n".join(["REPUBLICA DE COLOMBIA", "CEDULA", *ICAO_TD3_LINES])

        response = client.post(f"{API}/validate-mrz", headers=auth, json={"text": text})

        assert response.status_code == 200
        assert response.json()["valid"] is True

    def test_validates_td1_documents(self, client: TestClient, auth: dict[str, str]) -> None:
        td1 = [
            "I<UTOD231458907<<<<<<<<<<<<<<<",
            "7408122F1204159UTO<<<<<<<<<<<6",
            "ERIKSSON<<ANNA<MARIA<<<<<<<<<<",
        ]

        response = client.post(f"{API}/validate-mrz", headers=auth, json={"mrz_lines": td1})

        body = response.json()
        assert body["valid"] is True
        assert body["mrz"]["format"] == "TD1"

    def test_reports_when_no_mrz_is_found(self, client: TestClient, auth: dict[str, str]) -> None:
        response = client.post(f"{API}/validate-mrz", headers=auth, json={"text": "JUAN PEREZ\nBogota"})

        body = response.json()
        assert body["valid"] is False
        assert body["mrz"]["detected"] is False

    def test_requires_some_input(self, client: TestClient, auth: dict[str, str]) -> None:
        response = client.post(f"{API}/validate-mrz", headers=auth, json={})

        assert response.status_code == 422
        assert "mrz_lines" in response.text

    def test_rejects_an_over_long_payload(self, client: TestClient, auth: dict[str, str]) -> None:
        response = client.post(
            f"{API}/validate-mrz",
            headers=auth,
            json={"mrz_lines": ["A" * 40, "B" * 40, "C" * 40, "D" * 40]},
        )
        assert response.status_code == 422
