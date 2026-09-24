"""Pruebas del pipeline completo de escaneo de documentos."""

from __future__ import annotations

import dataclasses

import cv2
import numpy as np
import pytest

from app.core.config import get_settings
from app.core.errors import PayloadTooLargeError
from app.services.face_detector import HeuristicFaceDetector
from app.services.image_utils import InvalidImageError
from app.services.scanner import DocumentScanner, ScanResult, mask_document_number
from app.services.ocr_engine import get_ocr_engine
from tests.fixtures import (
    ICAO_TD3_LINES,
    TARJETA_IDENTIDAD_BACK_TEXT,
    TARJETA_IDENTIDAD_FRONT_TEXT,
    build_td3_mrz,
    document_bytes,
    encode,
    render_synthetic_document,
)
from tests.stubs import (
    FailingOCREngine,
    RecordingHeuristicEngine,
    StubOCREngine,
    TwoSidedStubOCREngine,
)

COLOMBIAN_ID_TEXT = [
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
]


def build_scanner(texts: list[str] | None = None, **settings_overrides: object) -> DocumentScanner:
    """Escaner rapido: OCR simulado + detector heuristico de rostro."""
    settings = dataclasses.replace(get_settings(), **settings_overrides)  # type: ignore[arg-type]
    return DocumentScanner(
        engine=StubOCREngine(texts if texts is not None else COLOMBIAN_ID_TEXT),
        face_detector=HeuristicFaceDetector(),
        settings=settings,
    )


class TestMasking:
    """El numero de documento no debe quedar en claro en los registros."""

    def test_masks_all_but_last_four_digits(self) -> None:
        assert mask_document_number("12345678") == "****5678"

    def test_handles_short_values(self) -> None:
        assert mask_document_number("12") == "**"

    def test_handles_none(self) -> None:
        assert mask_document_number(None) is None


class TestScanPipeline:
    def test_extracts_document_number_and_name(self) -> None:
        result = build_scanner().scan(document_bytes())

        assert result.document_number == "12345678"
        assert result.name == "JUAN PEREZ"
        assert result.fields.birth_date == "1974-08-12"
        assert result.fields.document_type == "CEDULA DE CIUDADANIA"

    def test_validates_a_consistent_mrz(self) -> None:
        result = build_scanner(list(ICAO_TD3_LINES)).scan(document_bytes())

        assert result.mrz.detected is True
        assert result.mrz.mrz_format == "TD3"
        assert result.mrz.valid is True
        assert result.mrz.passed_checks == result.mrz.total_checks

    def test_flags_an_inconsistent_mrz(self) -> None:
        tampered = [ICAO_TD3_LINES[0], ICAO_TD3_LINES[1].replace("L898902C3", "L898902C4")]
        result = build_scanner(tampered).scan(document_bytes())

        assert result.mrz.detected is True
        assert result.mrz.valid is False
        assert result.mrz.checks["document_number"] is False

    def test_detects_the_face_and_validates_the_photo(self) -> None:
        result = build_scanner().scan(document_bytes())

        assert result.face.detected is True
        assert result.valid_photo is True

    def test_reports_missing_face(self) -> None:
        result = build_scanner().scan(document_bytes(with_face=False))

        assert result.face.detected is False
        assert result.valid_photo is False

    def test_detects_and_crops_the_document(self) -> None:
        result = build_scanner().scan(document_bytes())
        assert result.document_detected is True

    def test_engine_name_is_reported(self) -> None:
        assert build_scanner().scan(document_bytes()).engine == "stub"

    def test_confidence_is_bounded(self) -> None:
        result = build_scanner().scan(document_bytes())
        assert 0.0 <= result.confidence <= 1.0

    def test_processing_time_is_measured(self) -> None:
        assert build_scanner().scan(document_bytes()).processing_ms > 0


class TestPreprocessing:
    def test_preview_is_generated_on_request(self) -> None:
        result = build_scanner().scan(document_bytes(), include_preview=True)

        assert result.preview_jpeg is not None
        assert result.preview_jpeg.startswith(b"\xff\xd8")  # cabecera JPEG
        assert result.preview_base64 is not None

    def test_preview_is_omitted_by_default(self) -> None:
        result = build_scanner().scan(document_bytes())

        assert result.preview_jpeg is None
        assert result.preview_base64 is None

    def test_accepts_png_input(self) -> None:
        png = encode(render_synthetic_document(), ".png")
        assert build_scanner().scan(png).document_number == "12345678"

    def test_warns_on_a_blurry_photo(self) -> None:
        flat = np.full((640, 1000, 3), 190, dtype=np.uint8)
        result = build_scanner().scan(encode(flat))

        assert result.quality.is_blurry is True
        assert any("movida" in warning or "desenfocada" in warning for warning in result.warnings)

    def test_reports_quality_metrics(self) -> None:
        quality = build_scanner().scan(document_bytes()).quality

        assert quality.width > 0 and quality.height > 0
        assert 0.0 <= quality.score <= 1.0

    def test_noisy_photo_still_reports_quality(self) -> None:
        noisy = document_bytes(noise=18.0)
        result = build_scanner().scan(noisy)

        assert result.quality.blur_score > 0

    def test_corrects_a_rotated_document(self) -> None:
        image = render_synthetic_document()
        rotated = cv2.warpAffine(
            image,
            cv2.getRotationMatrix2D((image.shape[1] / 2, image.shape[0] / 2), 3.0, 1.0),
            (image.shape[1], image.shape[0]),
            borderMode=cv2.BORDER_REPLICATE,
        )

        result = build_scanner().scan(encode(rotated))

        assert abs(result.deskew_angle) <= 20.0


class TestTwoSidedScan:
    """El documento se fotografia por caras y el escaner las fusiona."""

    def build(self) -> DocumentScanner:
        return DocumentScanner(
            engine=TwoSidedStubOCREngine(TARJETA_IDENTIDAD_FRONT_TEXT, TARJETA_IDENTIDAD_BACK_TEXT),
            face_detector=HeuristicFaceDetector(),
            settings=get_settings(),
        )

    def test_merges_the_text_of_both_sides(self) -> None:
        result = self.build().scan_sides(
            [("front", document_bytes()), ("back", document_bytes())]
        )

        # Ninguna de las dos caras trae por si sola todos los campos: la
        # frontal tiene el numero y el nombre, el reverso las fechas y el
        # grupo sanguineo.
        assert result.document_number == "1033186199"
        assert result.name == "JUAN JOSE OCAMPO ARTEAGA"
        assert result.fields.birth_date == "2008-09-20"
        assert result.fields.birth_place == "MEDELLIN (ANTIOQUIA)"
        assert result.fields.expiry_date == "2026-09-20"
        assert result.fields.blood_type == "O+"

    def test_reports_each_side(self) -> None:
        result = self.build().scan_sides(
            [("front", document_bytes()), ("back", document_bytes())]
        )

        assert [side.side for side in result.sides] == ["front", "back"]
        assert all(side.ok for side in result.sides)
        assert result.both_sides is True
        assert result.sides[0].width > 0

    def test_warns_when_only_one_side_is_scanned(self) -> None:
        result = build_scanner().scan(document_bytes())

        assert result.both_sides is False
        assert len(result.sides) == 1
        assert any("una sola cara" in warning for warning in result.warnings)

    def test_keeps_the_readable_side_when_the_other_fails(self) -> None:
        result = self.build().scan_sides(
            [("front", b"esto no es una imagen"), ("back", document_bytes())]
        )

        assert result.sides[0].ok is False
        assert result.sides[0].error
        assert result.sides[1].ok is True
        assert result.both_sides is False

    def test_fails_when_no_side_can_be_decoded(self) -> None:
        with pytest.raises(InvalidImageError):
            self.build().scan_sides(
                [("front", b"esto no es una imagen"), ("back", b"tampoco")]
            )

    def test_labels_the_warnings_with_the_side(self) -> None:
        blurry = encode(np.full((640, 1000, 3), 190, dtype=np.uint8))

        result = self.build().scan_sides([("front", document_bytes()), ("back", blurry)])

        assert any(warning.startswith("[reverso]") for warning in result.warnings)


class TestErrorHandling:
    def test_rejects_a_file_that_is_not_an_image(self) -> None:
        with pytest.raises(InvalidImageError):
            build_scanner().scan(b"esto no es una imagen")

    def test_rejects_an_empty_payload(self) -> None:
        with pytest.raises(InvalidImageError):
            build_scanner().scan(b"")

    def test_rejects_an_image_over_the_size_limit(self) -> None:
        scanner = build_scanner(max_upload_mb=0)

        with pytest.raises(PayloadTooLargeError, match="limite") as ctx:
            scanner.scan(document_bytes())

        # El manejador global debe traducirlo a un 413.
        assert ctx.value.status_code == 413

    def test_propagates_engine_failures(self) -> None:
        scanner = DocumentScanner(
            engine=FailingOCREngine(),
            face_detector=HeuristicFaceDetector(),
            settings=get_settings(),
        )

        with pytest.raises(RuntimeError, match="fallo simulado"):
            scanner.scan(document_bytes())


class TestResponseOptimization:
    """Ajustes de rendimiento del pipeline."""

    def test_skips_the_extra_mrz_passes_when_the_first_read_is_enough(self) -> None:
        # La lectura de la pagina completa ya trae una MRZ valida y completa:
        # no hace falta invertir dos inferencias mas.
        engine = StubOCREngine([*COLOMBIAN_ID_TEXT, *build_td3_mrz()])
        scanner = DocumentScanner(engine=engine, face_detector=HeuristicFaceDetector())

        result = scanner.scan(document_bytes())

        assert result.mrz.valid is True
        assert result.used_mrz_passes is False
        assert len(engine.calls) == 1

    def test_runs_the_extra_passes_when_the_mrz_is_missing(self) -> None:
        engine = StubOCREngine(COLOMBIAN_ID_TEXT)
        scanner = DocumentScanner(engine=engine, face_detector=HeuristicFaceDetector())

        result = scanner.scan(document_bytes())

        assert result.used_mrz_passes is True
        assert len(engine.calls) == 3

    def test_caps_the_amount_of_raw_text_returned(self) -> None:
        many_lines = [f"LINEA DE TEXTO {index}" for index in range(200)]
        scanner = build_scanner(many_lines, ocr_max_text_lines=10)

        result = scanner.scan(document_bytes())

        assert len(result.text_lines) <= 10

    def test_keeps_every_line_when_under_the_cap(self) -> None:
        scanner = build_scanner(COLOMBIAN_ID_TEXT, ocr_max_text_lines=80)

        result = scanner.scan(document_bytes())

        assert len(result.text_lines) == len(COLOMBIAN_ID_TEXT)


class TestEngineWithoutRecognition:
    def test_warns_that_characters_cannot_be_read(self) -> None:
        engine = RecordingHeuristicEngine()
        scanner = DocumentScanner(engine=engine, face_detector=HeuristicFaceDetector(), settings=get_settings())

        result = scanner.scan(document_bytes())

        assert result.document_number is None
        assert any("reconocimiento" in warning for warning in result.warnings)

    def test_runs_the_mrz_band_passes(self) -> None:
        engine = RecordingHeuristicEngine()
        scanner = DocumentScanner(engine=engine, face_detector=HeuristicFaceDetector(), settings=get_settings())

        scanner.scan(document_bytes())

        # Una pasada de la pagina completa y otra de la franja MRZ. Como este
        # motor no reconoce caracteres, no tiene sentido encadenar la segunda
        # variante: el pipeline se corta.
        assert engine.calls == 2

    def test_stops_early_when_a_full_mrz_is_already_available(self) -> None:
        engine = StubOCREngine(["REPUBLICA DE COLOMBIA", *build_td3_mrz()])
        scanner = DocumentScanner(engine=engine, face_detector=HeuristicFaceDetector())

        result = scanner.scan(document_bytes())

        assert result.mrz.valid is True
        assert result.used_mrz_passes is False
        assert len(engine.calls) == 1


@pytest.mark.slow
class TestRealOCREngine:
    """Inferencia real con el motor configurado (RapidOCR o EasyOCR).

    Es lenta (carga el modelo y hace tres pasadas por documento), por eso se
    marca como `slow` y **comparte un unico escaneo** entre todas las
    comprobaciones de la clase.

    La exactitud del lector de MRZ (digitos de control, formatos TD1/TD2/TD3)
    se verifica por separado en `test_mrz.py` con los vectores oficiales del
    estandar, y de extremo a extremo en `TestScanPipeline` con un OCR
    deterministico. Aqui se comprueba que el modelo real lea el documento.
    """

    @pytest.fixture(scope="class")
    def scan_result(self) -> ScanResult:
        scanner = DocumentScanner(engine=get_ocr_engine(), face_detector=HeuristicFaceDetector())
        return scanner.scan(document_bytes(), include_preview=True)

    def test_uses_the_vision_model(self, scan_result: ScanResult) -> None:
        assert scan_result.engine in {"rapidocr", "easyocr"}
        assert scan_result.text_lines, "El OCR no reconocio ningun texto"

    def test_reads_the_printed_document_number(self, scan_result: ScanResult) -> None:
        assert scan_result.document_number == "12345678"

    def test_reads_the_holder_name(self, scan_result: ScanResult) -> None:
        assert scan_result.name == "JUAN PEREZ"

    def test_reads_the_printed_dates(self, scan_result: ScanResult) -> None:
        assert scan_result.fields.birth_date == "1974-08-12"
        assert scan_result.fields.expiry_date == "2022-04-15"

    def test_classifies_the_document_type(self, scan_result: ScanResult) -> None:
        assert scan_result.fields.document_type == "CEDULA DE CIUDADANIA"

    def test_locates_the_mrz_zone(self, scan_result: ScanResult) -> None:
        assert scan_result.mrz.detected is True
        assert scan_result.mrz.normalized_lines

    def test_validates_the_photo_of_the_holder(self, scan_result: ScanResult) -> None:
        assert scan_result.valid_photo is True

    def test_confidence_is_meaningful(self, scan_result: ScanResult) -> None:
        assert scan_result.confidence > 0.5

    def test_generates_the_preview(self, scan_result: ScanResult) -> None:
        assert scan_result.preview_base64

    def test_does_not_invent_fields_on_an_empty_document(self) -> None:
        scanner = DocumentScanner(engine=get_ocr_engine(), face_detector=HeuristicFaceDetector())
        blank = encode(np.full((880, 1400, 3), 235, dtype=np.uint8))

        result = scanner.scan(blank)

        assert result.document_number is None
        assert result.name is None
        assert result.valid_photo is False
