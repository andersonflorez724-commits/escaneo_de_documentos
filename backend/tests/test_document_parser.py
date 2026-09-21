"""Pruebas de la extraccion de campos del documento."""

from __future__ import annotations

from app.services.document_parser import (
    build_lines,
    detect_document_type,
    extract_document_number,
    extract_name,
    extract_sex,
    normalize_text,
    parse_document,
)
from app.services.mrz import parse_mrz
from tests.fixtures import ICAO_TD3_LINES

COLOMBIAN_ID_LINES = [
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
]


class TestNormalization:
    def test_removes_accents_and_uppercases(self) -> None:
        assert normalize_text("Cédula de Ciudadanía") == "CEDULA DE CIUDADANIA"

    def test_collapses_whitespace(self) -> None:
        assert normalize_text("  JUAN   PEREZ ") == "JUAN PEREZ"

    def test_build_lines_discards_noise(self) -> None:
        assert build_lines(["A", "JUAN PEREZ", "  ", "Bogota"]) == ["JUAN PEREZ", "Bogota"]


class TestDocumentNumber:
    def test_finds_labelled_number(self) -> None:
        value, score = extract_document_number(COLOMBIAN_ID_LINES)
        assert value == "12345678"
        assert score > 0.5

    def test_prefers_labelled_over_bare_number(self) -> None:
        lines = ["TEL 3001234567", "CEDULA 87654321"]
        value, _ = extract_document_number(lines)
        assert value == "87654321"

    def test_ignores_dates(self) -> None:
        value, _ = extract_document_number(["FECHA 12/08/1974"])
        assert value != "12081974"

    def test_returns_none_without_candidates(self) -> None:
        value, score = extract_document_number(["Bogota D.C.", "REPUBLICA"])
        assert value is None
        assert score == 0.0

    def test_accepts_alphanumeric_passport_number(self) -> None:
        value, _ = extract_document_number(["PASAPORTE L898902C3"])
        assert value == "L898902C3"


class TestName:
    def test_finds_name_after_label(self) -> None:
        value, score = extract_name(COLOMBIAN_ID_LINES)
        assert value == "JUAN PEREZ"
        assert score > 0.5

    def test_value_on_following_line(self) -> None:
        value, _ = extract_name(["NOMBRES", "MARIA FERNANDA RUIZ"])
        assert value == "MARIA FERNANDA RUIZ"

    def test_rejects_form_words(self) -> None:
        value, _ = extract_name(["REPUBLICA DE COLOMBIA", "CEDULA DE CIUDADANIA"])
        assert value is None

    def test_prefers_mrz_name(self) -> None:
        mrz = parse_mrz(list(ICAO_TD3_LINES))
        value, score = extract_name(COLOMBIAN_ID_LINES, mrz)
        assert value == "ANNA MARIA ERIKSSON"
        assert score > 0.9


class TestSex:
    def test_detects_sex_from_text(self) -> None:
        assert extract_sex(["SEXO M"]) == "M"
        assert extract_sex(["SEXO  F"]) == "F"

    def test_detects_long_form(self) -> None:
        assert extract_sex(["MASCULINO"]) == "M"
        assert extract_sex(["FEMENINO"]) == "F"

    def test_none_when_absent(self) -> None:
        assert extract_sex(["REPUBLICA DE COLOMBIA"]) is None


class TestDocumentType:
    def test_detects_colombian_id(self) -> None:
        assert detect_document_type(COLOMBIAN_ID_LINES) == "CEDULA DE CIUDADANIA"

    def test_detects_passport(self) -> None:
        assert detect_document_type(["PASAPORTE", "REPUBLICA DE COLOMBIA"]) == "PASAPORTE"

    def test_uses_mrz_code_when_no_hint(self) -> None:
        mrz = parse_mrz(list(ICAO_TD3_LINES))
        assert detect_document_type(["REPUBLICA DE X"], mrz) == "PASAPORTE"


class TestParseDocument:
    def test_extracts_everything_from_visible_text(self) -> None:
        fields = parse_document(COLOMBIAN_ID_LINES)

        assert fields.document_number == "12345678"
        assert fields.name == "JUAN PEREZ"
        assert fields.birth_date == "1974-08-12"
        assert fields.expiry_date == "2022-04-15"
        assert fields.sex == "M"
        assert fields.document_type == "CEDULA DE CIUDADANIA"
        assert fields.confidence > 0.5
        assert fields.sources["document_number"] == "visual"

    def test_mrz_has_priority_over_visible_text(self) -> None:
        texts = list(ICAO_TD3_LINES) + ["NUMERO 99999999", "NOMBRES OTRA PERSONA"]
        mrz = parse_mrz(list(ICAO_TD3_LINES))

        fields = parse_document(texts, mrz)

        assert fields.document_number == "L898902C3"
        assert fields.name == "ANNA MARIA ERIKSSON"
        assert fields.sources["document_number"] == "mrz"
        assert fields.sources["name"] == "mrz"
        assert fields.sex == "F"

    def test_mrz_lines_are_not_used_as_visible_text(self) -> None:
        # Sin MRZ, las lineas con rellenos no deben producir un "nombre".
        fields = parse_document(list(ICAO_TD3_LINES))
        assert fields.name is None

    def test_warns_when_nothing_is_readable(self) -> None:
        fields = parse_document(["..", "--"])

        assert fields.document_number is None
        assert fields.name is None
        assert fields.warnings

    def test_warns_when_mrz_is_inconsistent(self) -> None:
        tampered = [ICAO_TD3_LINES[0], ICAO_TD3_LINES[1].replace("L898902C3", "L898902C4")]
        mrz = parse_mrz(tampered)

        fields = parse_document(list(tampered), mrz)

        assert any("digitos de control" in warning for warning in fields.warnings)

    def test_handles_empty_input(self) -> None:
        fields = parse_document([])
        assert fields.document_number is None
        assert fields.as_dict()["confidence"] == 0.0
