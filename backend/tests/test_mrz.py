"""Pruebas del lector y validador de MRZ (ICAO 9293)."""

from __future__ import annotations

from datetime import date

import pytest

from app.services.mrz import (
    compute_check_digit,
    detect_format,
    find_mrz_lines,
    is_verification_code_line,
    normalize_line,
    parse_mrz,
    parse_verification_code,
    repair_line,
    verify_check_digit,
)

# ---------------------------------------------------------------------------
# Vectores oficiales del ejemplo trabajado de ICAO Doc 9303
# ---------------------------------------------------------------------------
TD3_LINES = [
    "P<UTOERIKSSON<<ANNA<MARIA<<<<<<<<<<<<<<<<<<<",
    "L898902C36UTO7408122F1204159ZE184226B<<<<<10",
]

TD2_LINES = [
    "I<UTOERIKSSON<<ANNA<MARIA<<<<<<<<<<<",
    "D231458907UTO7408122F1204159<<<<<<<6",
]

TD1_LINES = [
    "I<UTOD231458907<<<<<<<<<<<<<<<",
    "7408122F1204159UTO<<<<<<<<<<<6",
    "ERIKSSON<<ANNA<MARIA<<<<<<<<<<",
]


class TestCheckDigit:
    """Algoritmo modulo 10 con pesos 7-3-1."""

    @pytest.mark.parametrize(
        ("sequence", "expected"),
        [
            ("L898902C3", "6"),  # numero de documento del pasaporte de ejemplo
            ("740812", "2"),  # fecha de nacimiento
            ("120415", "9"),  # fecha de caducidad
            ("Z", "5"),  # 35 * 7 = 245 -> 5
            ("AB", "3"),  # 10*7 + 11*3 = 103 -> 3
            ("<", "0"),
            ("", "0"),
        ],
    )
    def test_compute_check_digit(self, sequence: str, expected: str) -> None:
        assert compute_check_digit(sequence) == expected

    def test_verify_check_digit(self) -> None:
        assert verify_check_digit("L898902C3", "6")
        assert not verify_check_digit("L898902C3", "5")

    def test_filler_is_valid_for_empty_field(self) -> None:
        assert verify_check_digit("<<<<<<<<<<", "<")
        assert not verify_check_digit("ABC<<<<<<<", "<")

    def test_lowercase_is_accepted(self) -> None:
        assert compute_check_digit("l898902c3") == "6"


class TestNormalization:
    def test_strips_accents_and_spaces(self) -> None:
        assert normalize_line(" Pérez  G<mez ") == "PEREZG<MEZ"

    def test_maps_ocr_confusions_to_filler(self) -> None:
        assert normalize_line("ABC««DEF") == "ABC<<DEF"
        assert normalize_line("ABC||DEF") == "ABC<<DEF"

    def test_repair_pads_short_line(self) -> None:
        repaired, repairs = repair_line("L898902C36", 44)
        assert len(repaired) == 44
        assert repaired.startswith("L898902C36")
        assert repairs

    def test_repair_truncates_long_line(self) -> None:
        repaired, repairs = repair_line("A" * 50, 44)
        assert len(repaired) == 44
        assert repairs


class TestFormatDetection:
    def test_td3(self) -> None:
        assert detect_format([normalize_line(line) for line in TD3_LINES]) == "TD3"

    def test_td2(self) -> None:
        assert detect_format([normalize_line(line) for line in TD2_LINES]) == "TD2"

    def test_td1(self) -> None:
        assert detect_format([normalize_line(line) for line in TD1_LINES]) == "TD1"

    def test_empty(self) -> None:
        assert detect_format([]) is None


class TestParseTD3:
    """Pasaporte: 2 lineas de 44 caracteres."""

    @pytest.fixture()
    def result(self):  # noqa: ANN201 - fixture
        return parse_mrz(TD3_LINES)

    def test_detected_and_valid(self, result) -> None:
        assert result.detected
        assert result.mrz_format == "TD3"
        assert result.all_checks_passed, result.checks
        assert result.valid

    def test_fields(self, result) -> None:
        assert result.document_number == "L898902C3"
        assert result.issuing_country == "UTO"
        assert result.nationality == "UTO"
        assert result.document_type == "P"
        assert result.surname == "ERIKSSON"
        assert result.given_names == "ANNA MARIA"
        assert result.full_name == "ANNA MARIA ERIKSSON"
        assert result.sex == "F"

    def test_dates(self, result) -> None:
        assert result.birth_date == date(1974, 8, 12)
        assert result.expiry_date == date(2012, 4, 15)

    def test_as_dict_is_json_serializable(self, result) -> None:
        payload = result.as_dict()
        assert payload["valid"] is True
        assert payload["total_checks"] == len(payload["checks"])
        assert payload["birth_date"] == "1974-08-12"


class TestParseTD2:
    @pytest.fixture()
    def result(self):  # noqa: ANN201 - fixture
        return parse_mrz(TD2_LINES)

    def test_detected_and_valid(self, result) -> None:
        assert result.mrz_format == "TD2"
        assert result.all_checks_passed, result.checks

    def test_fields(self, result) -> None:
        # En TD2 el numero de documento ocupa 9 caracteres y el '7' que le
        # sigue es su digito de control.
        assert result.document_number == "D23145890"
        assert result.surname == "ERIKSSON"
        assert result.birth_date == date(1974, 8, 12)
        assert result.expiry_date == date(2012, 4, 15)


class TestParseTD1:
    @pytest.fixture()
    def result(self):  # noqa: ANN201 - fixture
        return parse_mrz(TD1_LINES)

    def test_detected_and_valid(self, result) -> None:
        assert result.mrz_format == "TD1"
        assert result.all_checks_passed, result.checks

    def test_fields(self, result) -> None:
        # En TD1 el numero de documento ocupa 9 caracteres (el '7' final es
        # el digito de control).
        assert result.document_number == "D23145890"
        assert result.issuing_country == "UTO"
        assert result.surname == "ERIKSSON"
        assert result.given_names == "ANNA MARIA"

    def test_colombian_cc_strip_is_read_as_td1(self) -> None:
        # Franja tipo TD1 de la Cedula colombiana: el numero va en la linea 1
        # con el control impreso en '<' (documento que no usa ese check) y el
        # NUIP en el campo opcional de la linea 2.
        lines = [
            "1CC0L000000012<<<<<<<<<<<<<<<<",
            "8808213F3101300C0L1234567890<9",
        ]
        result = parse_mrz(lines)

        assert result.mrz_format == "TD1"
        assert result.birth_date == date(1988, 8, 21)
        assert result.expiry_date == date(2031, 1, 30)
        assert result.sex == "F"
        assert result.checks["birth_date"] is True
        assert result.checks["expiry_date"] is True
        # El '<' de la casilla de control no se cuenta como falla.
        assert "document_number" not in result.checks
        assert "personal_number" not in result.checks

    def test_detects_td1_even_with_a_long_noise_candidate(self) -> None:
        # Una linea de 44 caracteres de texto visual no debe forzar TD3 cuando
        # hay dos renglones con estructura TD1.
        assert (
            detect_format(
                [
                    "REGISTRADORNACIONALNEZ<GARCIA<<MARIA<DANIELA",
                    "1CC0L000000012<<<<<<<<<<<<<<<<",
                    "8808213F3101300C0L1234567890<9",
                ]
            )
            == "TD1"
        )


class TestInconsistency:
    def test_tampered_document_number_fails_check(self) -> None:
        tampered = [TD3_LINES[0], TD3_LINES[1].replace("L898902C3", "L898902C4")]
        result = parse_mrz(tampered)

        assert result.detected
        assert result.checks["document_number"] is False
        assert result.checks["composite"] is False
        assert result.valid is False
        assert 0.0 < result.score < 1.0

    def test_invalid_birth_date_is_reported(self) -> None:
        tampered = [TD3_LINES[0], TD3_LINES[1][:13] + "741332" + TD3_LINES[1][19:]]
        result = parse_mrz(tampered)

        assert result.birth_date is None
        assert any("nacimiento" in error for error in result.errors)

    def test_empty_input(self) -> None:
        result = parse_mrz([])

        assert result.detected is False
        assert result.errors

    def test_missing_line_is_padded(self) -> None:
        result = parse_mrz([TD3_LINES[0]])

        assert result.detected
        assert len(result.normalized_lines) == 2
        assert result.repairs


class TestFindMrzLines:
    def test_ignores_the_colombian_verification_code(self) -> None:
        # La Tarjeta de Identidad y la Cedula no llevan MRZ: el reverso trae un
        # codigo de verificacion que no debe leerse como TD2.
        code = "P-0100150-00799122-M-1033186199-20160309 0048866905A 45514466"

        assert find_mrz_lines([code]) == []
        assert parse_mrz([code]).detected is False

    def test_ignores_visual_noise_lines(self) -> None:
        # Firma y encabezado de la oficina: longitud compatible con la MRZ pero
        # sin estructura; si entran, la deteccion se inclina a TD3.
        noise = [
            "FIRMAMARTNENACIONALLDADESTATURASEXO",
            "REGISTRADORNACIONALNEZ<GARCIA<<MARIA<DANIELA",
        ]
        assert find_mrz_lines(noise) == []

    def test_ignores_a_long_line_without_structure(self) -> None:
        # Un renglon normal del documento, normalizado, parece una MRZ por su
        # longitud y su falta de rellenos: la estructura lo delata.
        line = "FECHA DE VENCIMIENTO 20-SEP-2026 G S RH SEXO"

        assert normalize_line(line)
        assert find_mrz_lines([line]) == []

    def test_reads_the_colombian_verification_code(self) -> None:
        code = "P-0100150-00799122-M-1033186199-20160309 0048866905A 45514466"

        assert is_verification_code_line(code) is True
        parsed = parse_verification_code(code)
        assert parsed is not None
        assert parsed["sex"] == "M"
        assert parsed["number"] == "1033186199"
        assert parsed["date"] == "20160309"

    def test_a_plain_line_is_not_a_verification_code(self) -> None:
        assert is_verification_code_line("JUAN PEREZ") is False
        assert parse_verification_code("JUAN PEREZ") is None

    def test_does_not_guess_td1_from_a_single_short_line(self) -> None:
        assert detect_format(["P<COLPEREZ<<JUAN<<<<<<<<<<<<<<<<"]) is None

    def test_extracts_from_noisy_ocr_text(self) -> None:
        text = [
            "REPUBLICA DE COLOMBIA",
            "CEDULA DE CIUDADANIA",
            "P<UTOERIKSSON<<ANNA<MARIA<<<<<<<<<<<<<<<<<<<",
            "L898902C36UTO7408122F1204159ZE184226B<<<<<10",
        ]

        lines = find_mrz_lines(text)

        assert lines == [normalize_line(line) for line in TD3_LINES]

    def test_ignores_regular_text(self) -> None:
        assert find_mrz_lines(["JUAN PEREZ", "Bogota D.C.", "12345678"]) == []

    def test_handles_multiline_block(self) -> None:
        block = "\n".join(["P<UTOERIKSSON<<ANNA<MARIA<<<<<<<<<<<<<<<<<<<", TD3_LINES[1]])
        assert len(find_mrz_lines([block])) == 2


class TestNameSplitting:
    def test_multiple_given_names(self) -> None:
        result = parse_mrz(TD3_LINES)
        assert result.given_names == "ANNA MARIA"

    def test_single_name_without_filler(self) -> None:
        lines = list(TD3_LINES)
        lines[0] = "P<UTOSOLIS<<JUAN<<<<<<<<<<<<<<<<<<<<<<<<<<"
        result = parse_mrz(lines)
        assert result.surname == "SOLIS"
        assert result.given_names == "JUAN"
