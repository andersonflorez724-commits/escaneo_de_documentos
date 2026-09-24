"""Pruebas de la extraccion de campos del documento."""

from __future__ import annotations

from app.services.document_parser import (
    BIRTH_LABEL_PATTERN,
    EXPIRY_LABEL_PATTERN,
    build_lines,
    detect_document_type,
    extract_blood_type,
    extract_date,
    extract_document_number,
    extract_issue,
    extract_name,
    extract_sex,
    normalize_text,
    parse_document,
)
from app.services.mrz import parse_mrz
from tests.fixtures import (
    ICAO_TD3_LINES,
    TARJETA_IDENTIDAD_BACK_TEXT,
    TARJETA_IDENTIDAD_FRONT_TEXT,
)

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
    "FECHA DE VENCIMIENTO",
    "15/04/2022",
]

# Texto real del OCR cuando la foto se toma con la camara: los rotulos
# pequenos (APELLIDOS, NOMBRES, FECHA DE ..., SEXO, ...) no se leen y solo
# quedan los valores grandes y algo de ruido.
CAMERA_OCR_LINES = [
    "2 ;",
    "identificac1on",
    "1.020.116.685",
    "FLOREZ",
    "ANDERSON {{1aA",
    "Anderòn",
    "13-Nov-2008",
    "MeDELLIN",
    "Cnnooui",
    "13-NOV-2026 0+ M",
]

# OCR real de la Cedula de Ciudadan\u00eda (franja tipo TD1 en el reverso).
CEDULA_CC_OCR_LINES = [
    "FIRMAMARTNENACIONALLDADESTATURASEXO",
    "REGISTRADORNACIONALNEZ<GARCIA<<MARIA<DANIELA",
    "1CC0L000000012<<<<<<<<<<<<<<<<",
    "8808213F3101300C0L1234567890<9",
    "APELLIDOS MARTINEZ GARCIA",
    "NOMBRES SARCA.M. MARIA DANIELA",
    "NUIP 1.234.567.890",
    "Fecha de nacimiento 30 ENE 2031 21 AGO 1988 G.S. +0",
    "FECHA Y LUGAR DE EXPEDICION 23 SEP 2006 BOGOTA D C",
    "LUGAR DE NACIMIENTO C22TA",
    "FECHA DE VENCIMIENTO",
    "30 ENE 2031",
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


TARJETA_IDENTIDAD_FRONT = TARJETA_IDENTIDAD_FRONT_TEXT
TARJETA_IDENTIDAD_BACK = TARJETA_IDENTIDAD_BACK_TEXT


class TestColombianIdentityCard:
    """Campos reales de la Tarjeta de Identidad (frontal + reverso)."""

    def test_reads_every_field_of_both_sides(self) -> None:
        fields = parse_document([*TARJETA_IDENTIDAD_FRONT, *TARJETA_IDENTIDAD_BACK])

        assert fields.document_number == "1033186199"
        assert fields.document_type == "TARJETA DE IDENTIDAD"
        assert fields.name == "JUAN JOSE OCAMPO ARTEAGA"
        assert fields.first_surname == "OCAMPO"
        assert fields.second_surname == "ARTEAGA"
        assert fields.given_names == "JUAN JOSE"
        assert fields.birth_date == "2008-09-20"
        assert fields.birth_place == "MEDELLIN (ANTIOQUIA)"
        assert fields.issue_date == "2016-02-02"
        assert fields.issue_place == "MEDELLIN"
        assert fields.expiry_date == "2026-09-20"
        assert fields.sex == "M"
        assert fields.blood_type == "O+"
        assert fields.nationality == "COL"

    def test_the_signature_is_not_mistaken_for_the_holder_name(self) -> None:
        fields = parse_document(TARJETA_IDENTIDAD_FRONT)

        assert fields.first_surname == "OCAMPO"
        assert fields.given_names == "JUAN JOSE"
        assert "IRMA" not in (fields.name or "")
        assert "IOSE" not in (fields.name or "")

    def test_reads_the_number_with_thousands_separators(self) -> None:
        value, _ = extract_document_number(["NÚMERO", "1.033.186.199"])
        assert value == "1033186199"

    def test_the_dates_of_the_back_are_not_swapped(self) -> None:
        fields = parse_document(TARJETA_IDENTIDAD_BACK)

        # La caducidad es la del rotulo "FECHA DE VENCIMIENTO" y la expedicion
        # la de "FECHA Y LUGAR DE EXPEDICIÓN".
        assert fields.expiry_date == "2026-09-20"
        assert fields.issue_date == "2016-02-02"

    def test_blood_type_and_sex_share_the_value_line(self) -> None:
        lines = ["20-SEP-2026 0+ M", "FECHA DE VENCIMIENTO G $ RH SEXO"]

        value, _ = extract_blood_type(lines)
        assert value == "O+"
        assert extract_sex(lines) == "M"

    def test_the_verification_code_backs_the_visual_fields(self) -> None:
        # Solo el reverso: el codigo de verificacion aporta el numero y el sexo.
        code = "P-0100150-00799122-M-1033186199-20160309 0048866905A 45514466"
        fields = parse_document([*TARJETA_IDENTIDAD_BACK, code])

        assert fields.document_number == "1033186199"
        assert fields.sources["document_number"] in {"barcode", "visual"}

    def test_warns_when_the_number_does_not_match_the_code(self) -> None:
        tampered = [
            "NÚMERO",
            "1.033.186.100",
            "P-0100150-00799122-M-1033186199-20160309 0048866905A 45514466",
        ]
        fields = parse_document(tampered)

        assert any("codigo de verificacion" in warning for warning in fields.warnings)


class TestDateAssociation:
    """La fecha debe ser la que acompana a su etiqueta, no cualquier fecha."""

    def test_reads_value_after_the_label_on_the_same_line(self) -> None:
        value, score = extract_date(["VENCIMIENTO 15/04/2022"], EXPIRY_LABEL_PATTERN)

        assert value == "2022-04-15"
        assert score > 0.5

    def test_reads_value_on_the_line_below_the_label(self) -> None:
        value, _ = extract_date(
            ["FECHA DE NACIMIENTO", "12/08/1974"], BIRTH_LABEL_PATTERN
        )

        assert value == "1974-08-12"

    def test_ignores_a_date_that_precedes_the_label(self) -> None:
        # Regresion: el OCR puede unir dos zonas del documento y dejar la
        # fecha de caducidad delante de la etiqueta de nacimiento. Esa fecha
        # pertenece a otra zona y no debe usarse como fecha de nacimiento.
        value, _ = extract_date(
            ["15/04/2022 FECHA DE NACIMIENTO", "12/08/1974"], BIRTH_LABEL_PATTERN
        )

        assert value == "1974-08-12"

    def test_does_not_borrow_the_expiry_date_for_the_birth_label(self) -> None:
        lines = [
            "FECHA DE NACIMIENTO",
            "12/08/1974",
            "VENCIMIENTO 15/04/2022",
        ]
        fields = parse_document(lines)

        assert fields.birth_date == "1974-08-12"
        assert fields.expiry_date == "2022-04-15"

    def test_ignores_unrelated_dates_when_the_label_has_no_value(self) -> None:
        value, _ = extract_date(["FECHA DE NACIMIENTO"], BIRTH_LABEL_PATTERN)

        assert value is None

    def test_does_not_confuse_the_expiry_label_with_the_birth_label(self) -> None:
        value, _ = extract_date(["VENCIMIENTO 15/04/2022"], BIRTH_LABEL_PATTERN)

        assert value is None

    def test_reads_the_value_when_the_label_is_below_it(self) -> None:
        # Maquetacion de la Tarjeta de Identidad: el valor va encima del rotulo.
        value, _ = extract_date(
            ["15/04/2022", "FECHA DE VENCIMIENTO"], EXPIRY_LABEL_PATTERN
        )

        assert value == "2022-04-15"

    def test_does_not_take_the_expiry_date_from_the_issue_line(self) -> None:
        # La linea que lleva otro rotulo de fecha pertenece a esa otra fecha.
        lines = [
            "20-SEP-2026 0+ M",
            "FECHA DE VENCIMIENTO G $ RH SEXO",
            "02-FEB-2016 MEDELLIN FECHA Y LUGAR DE EXPEDICION",
        ]
        fields = parse_document(lines)

        assert fields.expiry_date == "2026-09-20"
        assert fields.issue_date == "2016-02-02"

    def test_expiry_ignores_the_issue_date_when_the_label_is_on_the_next_line(self) -> None:
        # RapidOCR (y otros) parten "02-FEB-2016 MEDELLIN" y
        # "FECHA Y LUGAR DE EXPEDICION" en dos renglones: el vencimiento no
        # debe quedarse con la fecha de expedicion solo porque este debajo.
        lines = [
            "20-SEP-2026 O+ M",
            "FECHA DE VENCIMIENTO GS RH SEXO",
            "02-FEB-2016 MEDELLIN",
            "FECHA Y LUGAR DE EXPEDICION",
        ]
        fields = parse_document(lines)

        assert fields.expiry_date == "2026-09-20"
        assert fields.issue_date == "2016-02-02"

    def test_name_rejects_form_labels(self) -> None:
        from app.services.document_parser import _looks_like_name

        assert not _looks_like_name("FECHA Y LUGAR DE EXPEDICION ANDRES")
        assert _looks_like_name("JUAN JOSE OCAMPO")

    def test_reads_dates_with_the_month_in_letters(self) -> None:
        value, _ = extract_date(["FECHA DE NACIMIENTO 15 ABR 2004"], BIRTH_LABEL_PATTERN)
        assert value == "2004-04-15"

        value, _ = extract_date(["FECHA DE NACIMIENTO 20-SEP-2008"], BIRTH_LABEL_PATTERN)
        assert value == "2008-09-20"


class TestPlacesAndIssue:
    """Lugar de nacimiento y expedicion, dos campos que comparten rotulo."""

    def test_rebuilds_a_place_split_in_two_lines(self) -> None:
        fields = parse_document(["MEDELLIN", "(ANTIOQUIA)", "LUGAR DE NACIMIENTO"])

        assert fields.birth_place == "MEDELLIN (ANTIOQUIA)"

    def test_reads_the_place_below_its_label(self) -> None:
        fields = parse_document(["LUGAR DE NACIMIENTO", "CARTAGENA (BOLIVAR)"])

        assert fields.birth_place == "CARTAGENA (BOLIVAR)"

    def test_splits_date_and_place_of_issue(self) -> None:
        fields = parse_document(["20 ABR 2022, CARTAGENA", "FECHA Y LUGAR DE EXPEDICIÓN"])

        assert fields.issue_date == "2022-04-20"
        assert fields.issue_place == "CARTAGENA"

    def test_reads_the_issue_when_the_value_follows_the_label(self) -> None:
        issued, place, _ = extract_issue(["FECHA Y LUGAR DE EXPEDICIÓN 02-FEB-2016 MEDELLIN"])

        assert issued == "2016-02-02"
        assert place == "MEDELLIN"

    def test_does_not_invent_a_place_from_the_barcode(self) -> None:
        fields = parse_document(
            [
                "FECHA Y LUGAR DE EXPEDICIÓN",
                "INDiCE Derecho P-0100150-00799122-M-1033186199-20160309",
            ]
        )

        assert fields.issue_place is None


class TestCameraCaptureWithoutLabels:
    """Fallbacks cuando la camara deja el texto sin los rotulos del documento."""

    def test_reads_every_field_from_value_lines_alone(self) -> None:
        fields = parse_document(CAMERA_OCR_LINES)

        assert fields.document_number == "1020116685"
        assert fields.first_surname == "FLOREZ"
        assert fields.second_surname is None
        assert fields.given_names == "ANDERSON"
        assert fields.name == "ANDERSON FLOREZ"
        assert fields.birth_date == "2008-11-13"
        assert fields.expiry_date == "2026-11-13"
        assert fields.issue_date is None
        assert fields.birth_place == "MEDELLIN"
        assert fields.sex == "M"
        assert fields.blood_type == "O+"
        assert fields.nationality == "COL"
        assert fields.issuing_country == "COL"
        assert fields.confidence > 0.3
        assert fields.sources["first_surname"] == "visual"
        assert fields.sources["birth_date"] == "visual"

    def test_does_not_take_the_signature_as_a_name(self) -> None:
        # Sin numero de documento no hay ventanilla fiable para los nombres.
        fields = parse_document(["FLOREZ", "ANDERSON", "13-Nov-2008"])

        assert fields.first_surname is None
        assert fields.given_names is None

    def test_sex_from_a_date_and_blood_line_without_label(self) -> None:
        assert extract_sex(["13-NOV-2026 0+ M"]) == "M"
        assert extract_sex(["20-SEP-2026 0+ F"]) == "F"

    def test_does_not_invent_an_expiry_from_a_single_birth_date(self) -> None:
        fields = parse_document(["1.033.186.199", "20-SEP-2008", "FLOREZ"])

        assert fields.birth_date == "2008-09-20"
        assert fields.expiry_date is None
        assert fields.issue_date is None


class TestCedulaDeCiudadania:
    """OCR real de la Cedula colombiana con franja tipo TD1."""

    def test_reads_every_field(self) -> None:
        from app.services.mrz import extract_and_validate

        mrz = extract_and_validate(CEDULA_CC_OCR_LINES)
        fields = parse_document(CEDULA_CC_OCR_LINES, mrz)

        assert mrz.mrz_format == "TD1"
        assert fields.document_number == "1234567890"
        assert fields.first_surname == "MARTINEZ"
        assert fields.second_surname == "GARCIA"
        assert fields.given_names == "MARIA DANIELA"
        assert fields.name == "MARIA DANIELA MARTINEZ GARCIA"
        assert fields.birth_date == "1988-08-21"
        assert fields.expiry_date == "2031-01-30"
        assert fields.sex == "F"
        assert fields.blood_type == "O+"
        assert fields.nationality == "COL"
        assert fields.issue_date == "2006-09-23"
        assert fields.issue_place == "BOGOTA D C"
        # La firma no se cuela en los nombres y el ruido no es lugar.
        assert "SARCA" not in (fields.given_names or "")
        assert fields.birth_place is None

    def test_birth_label_with_two_dates_picks_the_oldest(self) -> None:
        value, _ = extract_date(
            ["Fecha de nacimiento 30 ENE 2031 21 AGO 1988"],
            BIRTH_LABEL_PATTERN,
        )
        assert value == "1988-08-21"

    def test_expiry_label_with_two_dates_picks_the_newest(self) -> None:
        value, _ = extract_date(
            ["FECHA DE VENCIMIENTO 30 ENE 2031 21 AGO 1988"],
            EXPIRY_LABEL_PATTERN,
        )
        assert value == "2031-01-30"

    def test_blood_type_label_without_rh_word(self) -> None:
        value, _ = extract_blood_type(["21 AGO 1988 G.S. +0"])
        assert value == "O+"

    def test_signature_token_is_stripped_from_names(self) -> None:
        value, _ = extract_name(["NOMBRES", "SARCA.M. MARIA DANIELA"])
        assert value == "MARIA DANIELA"

    def test_mrz_noise_does_not_overwrite_visual_fields(self) -> None:
        from app.services.mrz import parse_mrz

        # Texto visual detectado por error como MRZ basura.
        mrz = parse_mrz(
            [
                "FIRMAMARTNENACIONALLDADESTATURASEXO<<<<<<<<<",
                "1CC0L000000012<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<",
            ]
        )
        fields = parse_document(
            ["NUIP 1.234.567.890", "NOMBRES", "MARIA DANIELA", "APELLIDOS", "MARTINEZ GARCIA"],
            mrz,
        )

        assert fields.document_number == "1234567890"
        assert fields.name == "MARIA DANIELA MARTINEZ GARCIA"
        assert fields.sources["document_number"] == "visual"
        assert fields.sources["name"] == "visual"
