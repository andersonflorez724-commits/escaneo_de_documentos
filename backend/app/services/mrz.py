"""Lectura y validacion de la MRZ (Machine Readable Zone).

Implementa el estandar **ICAO 9293** (Doc 9303, Parte 3):

* Juego de caracteres restringido: ``0-9 A-Z <``.
* Digito de control *modulo 10* con ponderaciones ``7 3 1`` repetidas.
* Formatos soportados: **TD1** (3x30, cedulas/DNI), **TD2** (2x36) y
  **TD3** (2x44, pasaportes).

La validacion de consistencia de caracteres consiste en recalcular cada
digito de control y compararlo con el impreso en el documento. Un documento
autentico devuelve todos los checks en ``True``; una fotocopia mal leida o un
documento manipulado suele fallar en al menos uno.
"""

from __future__ import annotations

import itertools
import re
import unicodedata
from dataclasses import dataclass, field
from datetime import date
from typing import Any, Iterable

# ---------------------------------------------------------------------------
# Alfabeto MRZ y pesos del digito de control
# ---------------------------------------------------------------------------
FILLER = "<"
MRZ_ALPHABET = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ" + FILLER

# Tabla de valores del estandar (ICAO 9293, tabla 4): los digitos valen su
# propio numero, las letras 10..35 y el relleno '<' vale 0. No se puede usar
# el indice de MRZ_ALPHABET porque '<' ocupa la ultima posicion.
CHAR_VALUES: dict[str, int] = {
    **{str(digit): digit for digit in range(10)},
    **{chr(ord("A") + offset): 10 + offset for offset in range(26)},
    FILLER: 0,
}

CHECK_WEIGHTS = (7, 3, 1)

# Longitudes oficiales por formato.
TD1_LINE_LENGTH = 30
TD2_LINE_LENGTH = 36
TD3_LINE_LENGTH = 44

MRZ_LINE_PATTERN = re.compile(r"^[A-Z0-9<]{28,46}$")
MIN_FILLER_RATIO = 0.06  # un documento real casi siempre trae rellenos '<'

# Tope de lecturas alternativas que se prueban al elegir la mejor combinacion.
MAX_MRZ_CANDIDATES = 6

# Sustituciones tipicas que introduce el OCR sobre la tipografia OCR-B.
LOOKALIKE_MAP: dict[str, str] = {
    "«": "<",
    "»": "<",
    "~": "<",
    "=": "<",
    "*": "<",
    "|": "<",
    "!": "<",
    "(": "<",
    ")": "<",
    "[": "<",
    "]": "<",
    "{": "<",
    "}": "<",
    "\\": "<",
    "/": "<",
    "“": "<",
    "”": "<",
    "’": "<",
    "`": "<",
    "·": "<",
    "•": "<",
    "§": "<",
}


# ---------------------------------------------------------------------------
# Digito de control (ICAO 9293, seccion 4.9)
# ---------------------------------------------------------------------------
def char_value(character: str) -> int:
    """Valor numerico de un caracter MRZ (``<`` vale 0)."""
    return CHAR_VALUES.get(character.upper(), 0)


def compute_check_digit(sequence: str) -> str:
    """Calcula el digito de control de una secuencia MRZ.

    Cada caracter se multiplica por el peso 7, 3 o 1 (ciclico), se suman los
    productos y se toma el modulo 10.

    >>> compute_check_digit("L898902C3")
    '6'
    >>> compute_check_digit("740812")
    '2'
    """
    total = 0
    for index, character in enumerate(sequence):
        total += char_value(character) * CHECK_WEIGHTS[index % len(CHECK_WEIGHTS)]
    return str(total % 10)


def verify_check_digit(sequence: str, expected: str) -> bool:
    """Indica si ``expected`` coincide con el digito de control calculado."""
    if not expected:
        return False
    if expected == FILLER:
        # Un '<' solo es valido cuando el campo opcional esta vacio.
        return not sequence.strip(FILLER)
    return compute_check_digit(sequence) == expected


# ---------------------------------------------------------------------------
# Normalizacion y reparacion
# ---------------------------------------------------------------------------
def normalize_line(line: str) -> str:
    """Deja la linea en mayusculas y limita el alfabeto MRZ."""
    if not line:
        return ""
    # Quita acentos (la MRZ no los admite) y espacios.
    decomposed = unicodedata.normalize("NFKD", line)
    ascii_only = "".join(char for char in decomposed if not unicodedata.combining(char))
    ascii_only = ascii_only.upper()
    ascii_only = "".join(LOOKALIKE_MAP.get(char, char) for char in ascii_only)
    ascii_only = re.sub(r"\s+", "", ascii_only)
    return "".join(char for char in ascii_only if char in MRZ_ALPHABET)


def repair_line(line: str, expected_length: int) -> tuple[str, list[str]]:
    """Ajusta la linea a la longitud oficial rellenando o recortando con ``<``.

    Returns:
        ``(linea_reparada, lista_de_reparaciones)``.
    """
    repairs: list[str] = []
    cleaned = normalize_line(line)

    if len(cleaned) < expected_length:
        missing = expected_length - len(cleaned)
        cleaned = cleaned + FILLER * missing
        repairs.append(f"Se agregaron {missing} rellenos '<' al final de la linea.")
    elif len(cleaned) > expected_length:
        cleaned = cleaned[:expected_length]
        repairs.append(f"Se recortaron {len(line) - expected_length} caracteres sobrantes.")

    return cleaned, repairs


# ---------------------------------------------------------------------------
# Deteccion de las lineas MRZ dentro del texto del OCR
# ---------------------------------------------------------------------------
def _is_plausible_line(cleaned: str) -> bool:
    """Una linea MRZ plausible trae rellenos ``<`` o digitos de control."""
    if len(cleaned) < TD1_LINE_LENGTH:
        return False
    if not MRZ_LINE_PATTERN.match(cleaned):
        return False

    filler_ratio = cleaned.count(FILLER) / len(cleaned)
    has_digit = any(char.isdigit() for char in cleaned)
    return not (filler_ratio < MIN_FILLER_RATIO and not has_digit)


def _split_merged_line(cleaned: str) -> list[str]:
    """Separa bloques donde el OCR unio varias lineas MRZ en una sola.

    El detector de texto tiende a fusionar renglones muy cercanos; si la
    longitud es un multiplo exacto de un formato conocido se trocea.
    """
    pieces: list[str] = []
    for line_length in (TD3_LINE_LENGTH, TD2_LINE_LENGTH, TD1_LINE_LENGTH):
        if len(cleaned) % line_length != 0:
            continue
        parts = [cleaned[i : i + line_length] for i in range(0, len(cleaned), line_length)]
        if len(parts) < 2:
            continue
        if all(_is_plausible_line(part) for part in parts):
            pieces = parts
            break
    return pieces


def find_mrz_lines(texts: Iterable[str]) -> list[str]:
    """Extrae del texto OCR las lineas candidatas a formar la MRZ.

    Se aceptan tanto renglones individuales como bloques que el OCR haya
    unido, que se trocean por multiples de las longitudes oficiales.
    """
    candidates: list[str] = []

    for raw in texts:
        for piece in str(raw).splitlines():
            cleaned = normalize_line(piece)
            if _is_plausible_line(cleaned):
                candidates.append(cleaned)
                continue

            candidates.extend(_split_merged_line(cleaned))

    # Elimina duplicados conservando el orden de aparicion.
    unique: list[str] = []
    for candidate in candidates:
        if candidate not in unique:
            unique.append(candidate)
    return unique


def detect_format(lines: list[str]) -> str | None:
    """Infiere el formato MRZ a partir de las longitudes de las lineas.

    El OCR puede recortar caracteres, por eso el criterio principal es la
    longitud de la linea mas larga y no una coincidencia exacta.
    """
    lengths = [len(line) for line in lines if line]
    if not lengths:
        return None

    longest = max(lengths)

    # Tres lineas cortas solo pueden ser TD1 (cedulas y DNI).
    if len(lengths) >= 3 and longest <= TD1_LINE_LENGTH + 3:
        return "TD1"
    if longest >= TD2_LINE_LENGTH + 2:
        return "TD3"
    if longest >= TD2_LINE_LENGTH - 2:
        return "TD2"
    if longest >= TD1_LINE_LENGTH - 2:
        return "TD1"
    return None


def expected_line_count(mrz_format: str) -> int:
    return 3 if mrz_format == "TD1" else 2


def expected_line_length(mrz_format: str) -> int:
    return {"TD1": TD1_LINE_LENGTH, "TD2": TD2_LINE_LENGTH, "TD3": TD3_LINE_LENGTH}.get(
        mrz_format, TD3_LINE_LENGTH
    )


# ---------------------------------------------------------------------------
# Clasificacion estructural de las lineas
# ---------------------------------------------------------------------------
_TD1_HEADER_RE = re.compile(r"^[A-Z<]{2}[A-Z<]{3}")


def looks_like_header_line(line: str) -> bool:
    """Linea 1 de TD2/TD3: codigo de documento + pais emisor + nombre."""
    if len(line) < TD2_LINE_LENGTH - 2:
        return False
    if not re.match(r"^[A-Z<]{2}[A-Z<]{3}", line):
        return False
    # El nombre se escribe con letras y rellenos: apenas lleva digitos.
    return sum(char.isdigit() for char in line) <= 3


def looks_like_data_line(line: str) -> bool:
    """Linea 2 de TD2/TD3: lleva digitos de control en las posiciones 10 y 20."""
    if len(line) < 20:
        return False
    return line[9].isdigit() and line[19].isdigit()


def looks_like_td1_header(line: str) -> bool:
    """Linea 1 de TD1: codigo de pais seguido del numero de documento."""
    return bool(_TD1_HEADER_RE.match(line)) and any(char.isdigit() for char in line[:15])


def looks_like_td1_middle(line: str) -> bool:
    """Linea 2 de TD1: empieza con la fecha de nacimiento ``YYMMDD``."""
    return len(line) >= 15 and line[0:6].isdigit() and line[6].isdigit()


def looks_like_td1_name(line: str) -> bool:
    """Linea 3 de TD1: ``APELLIDOS<<NOMBRES``, sin digitos."""
    return len(line) >= 25 and FILLER in line and not any(char.isdigit() for char in line)


_SLOT_MATCHERS: dict[str, tuple[Any, ...]] = {
    "TD1": (looks_like_td1_header, looks_like_td1_middle, looks_like_td1_name),
    "TD2": (looks_like_header_line, looks_like_data_line),
    "TD3": (looks_like_header_line, looks_like_data_line),
}


def assign_slots(candidates: list[str], mrz_format: str, line_length: int) -> list[str]:
    """Coloca cada linea leida por el OCR en la posicion que le corresponde.

    Es importante porque el OCR puede perder una linea: si la unica linea
    recuperada fuese la segunda y se tratase como primera, el nombre del
    titular saldria del campo equivocado.
    """
    matchers = _SLOT_MATCHERS.get(mrz_format, _SLOT_MATCHERS["TD3"])
    padded = [repair_line(candidate, line_length)[0] for candidate in candidates]
    slots: list[str | None] = [None] * len(matchers)
    remaining = list(padded)

    for index, matcher in enumerate(matchers):
        match = next((line for line in remaining if matcher(line)), None)
        if match is not None:
            slots[index] = match
            remaining.remove(match)

    # Los candidatos que no encajaron en ningun patron completan los huecos
    # en orden de aparicion.
    for index in range(len(slots)):
        if slots[index] is None and remaining:
            slots[index] = remaining.pop(0)

    # Las lineas ausentes se rellenan para poder parsear el resto.
    return [slot if slot is not None else FILLER * line_length for slot in slots]


# ---------------------------------------------------------------------------
# Fechas
# ---------------------------------------------------------------------------
def _expand_yymmdd(value: str, *, kind: str) -> date | None:
    """Convierte ``YYMMDD`` al siglo correcto segun el tipo de fecha."""
    digits = re.sub(r"\D", "", value)
    if len(digits) != 6:
        return None

    year, month, day = int(digits[0:2]), int(digits[2:4]), int(digits[4:6])
    today = date.today()
    current_year = today.year

    if kind == "birth":
        # Una fecha de nacimiento no puede estar en el futuro.
        candidate = year + (2000 if year <= current_year % 100 else 1900)
        if candidate > current_year:
            candidate -= 100
    else:
        # La caducidad suele estar como maximo ~15 anos en el futuro.
        candidate = year + (2000 if year <= (current_year % 100) + 15 else 1900)

    try:
        return date(candidate, month, day)
    except ValueError:
        return None


def _split_names(field_value: str) -> tuple[str | None, str | None]:
    """Separa ``APELLIDOS<<NOMBRES`` en apellidos y nombres legibles."""
    cleaned = field_value.strip(FILLER)
    if not cleaned:
        return None, None

    parts = re.split(r"<{1,}", cleaned)
    surname = parts[0].replace(FILLER, " ").strip() or None
    given = " ".join(part for part in parts[1:] if part).strip() or None
    return surname, given


# ---------------------------------------------------------------------------
# Resultado
# ---------------------------------------------------------------------------
@dataclass(slots=True)
class MRZResult:
    """Resultado del analisis de la MRZ."""

    detected: bool = False
    mrz_format: str | None = None
    lines: list[str] = field(default_factory=list)
    normalized_lines: list[str] = field(default_factory=list)
    repairs: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    document_type: str | None = None
    issuing_country: str | None = None
    document_number: str | None = None
    nationality: str | None = None
    surname: str | None = None
    given_names: str | None = None
    birth_date: date | None = None
    expiry_date: date | None = None
    sex: str | None = None
    personal_number: str | None = None

    checks: dict[str, bool] = field(default_factory=dict)
    complete: bool = False

    @property
    def full_name(self) -> str | None:
        parts = [part for part in (self.given_names, self.surname) if part]
        return " ".join(parts) if parts else None

    @property
    def valid(self) -> bool:
        """La MRZ es consistente: detectada y con todos los checks correctos."""
        return self.detected and self.all_checks_passed

    @property
    def all_checks_passed(self) -> bool:
        return bool(self.checks) and all(self.checks.values())

    @property
    def passed_checks(self) -> int:
        return sum(1 for value in self.checks.values() if value)

    @property
    def total_checks(self) -> int:
        return len(self.checks)

    @property
    def inconsistent_fields(self) -> list[str]:
        """Campos cuyo digito de control no coincide con el valor impreso."""
        return [name for name, passed in self.checks.items() if not passed]

    @property
    def score(self) -> float:
        """Proporcion de digitos de control que coinciden (0..1)."""
        if not self.checks:
            return 0.0
        return self.passed_checks / len(self.checks)

    def as_dict(self) -> dict[str, Any]:
        return {
            "detected": self.detected,
            "format": self.mrz_format,
            "lines": self.lines,
            "normalized_lines": self.normalized_lines,
            "repairs": self.repairs,
            "errors": self.errors,
            "document_type": self.document_type,
            "issuing_country": self.issuing_country,
            "document_number": self.document_number,
            "nationality": self.nationality,
            "surname": self.surname,
            "given_names": self.given_names,
            "full_name": self.full_name,
            "birth_date": self.birth_date.isoformat() if self.birth_date else None,
            "expiry_date": self.expiry_date.isoformat() if self.expiry_date else None,
            "sex": self.sex,
            "personal_number": self.personal_number,
            "checks": self.checks,
            "passed_checks": self.passed_checks,
            "total_checks": self.total_checks,
            "inconsistent_fields": self.inconsistent_fields,
            "score": round(self.score, 4),
            "complete": self.complete,
            "valid": self.valid,
        }


# ---------------------------------------------------------------------------
# Parseo
# ---------------------------------------------------------------------------
def parse_mrz(lines: Iterable[str]) -> MRZResult:
    """Analiza y valida una o varias lineas MRZ.

    Args:
        lines: Lineas tal como las entrega el OCR (se normalizan/reparan).

    Returns:
        :class:`MRZResult` con los campos extraidos y el estado de cada
        digito de control.
    """
    raw_lines = [line for line in (str(item) for item in lines) if line.strip()]

    if not raw_lines:
        empty = MRZResult(lines=raw_lines)
        empty.errors.append("No se recibieron lineas MRZ para analizar.")
        return empty

    candidates = _dedupe_candidates([normalize_line(line) for line in raw_lines])
    mrz_format = detect_format(candidates)
    if mrz_format is None:
        unknown = MRZResult(lines=raw_lines)
        unknown.errors.append("No se pudo determinar el formato de la MRZ (TD1, TD2 o TD3).")
        return unknown

    line_length = expected_line_length(mrz_format)
    line_count = expected_line_count(mrz_format)
    pool = candidates[:MAX_MRZ_CANDIDATES]

    # Puede haber varias lecturas del mismo renglon (una por variante de
    # preprocesado). Se prueban las combinaciones posibles y se elige la que
    # supera mas digitos de control: es el mejor criterio de desempate.-
    combinations: list[tuple[str, ...]] = (
        [tuple(pool)]
        if len(pool) <= line_count
        else list(itertools.combinations(pool, line_count))
    )

    best: MRZResult | None = None
    for combination in combinations:
        analyser = _analyse(list(combination), mrz_format, line_length, line_count, raw_lines)
        if best is None or _rank(analyser) > _rank(best):
            best = analyser

    return best if best is not None else MRZResult(lines=raw_lines)


def _dedupe_candidates(candidates: list[str]) -> list[str]:
    """Elimina lineas vacias o repetidas conservando el orden de lectura."""
    unique: list[str] = []
    for candidate in candidates:
        if candidate and candidate not in unique:
            unique.append(candidate)
    return unique


def _rank(result: MRZResult) -> tuple[int, bool, int]:
    """Criterio de eleccion entre lecturas alternativas del mismo documento."""
    return (result.passed_checks, result.complete, len(result.checks))


def _analyse(
    candidates: list[str],
    mrz_format: str,
    line_length: int,
    line_count: int,
    raw_lines: list[str],
) -> MRZResult:
    """Ubica las lineas candidatas en sus posiciones y valida los controles."""
    result = MRZResult(lines=raw_lines)

    for line in candidates:
        _, repairs = repair_line(line, line_length)
        result.repairs.extend(repairs)

    if len(candidates) < line_count:
        result.repairs.append(
            f"Se esperaban {line_count} lineas MRZ y solo se leyeron {len(candidates)}; "
            "los campos ausentes se dejan vacios."
        )
        result.complete = False
    else:
        result.complete = True

    normalized = assign_slots(candidates, mrz_format, line_length)
    result.mrz_format = mrz_format
    result.normalized_lines = normalized
    result.detected = True

    try:
        if mrz_format == "TD1":
            _parse_td1(normalized, result)
        elif mrz_format == "TD2":
            _parse_td2(normalized, result)
        else:
            _parse_td3(normalized, result)
    except (IndexError, ValueError) as exc:  # defensivo: lineas deformadas
        result.errors.append(f"La MRZ esta mal formada: {exc}")
        result.detected = False

    return result


def _parse_td3(lines: list[str], result: MRZResult) -> None:
    """Pasaporte: 2 lineas de 44 caracteres."""
    first, second = lines[0], lines[1]

    result.document_type = first[0:2].strip(FILLER) or None
    result.issuing_country = first[2:5].strip(FILLER) or None
    result.surname, result.given_names = _split_names(first[5:44])

    document_number_field = second[0:9]
    birth_field = second[13:19]
    expiry_field = second[21:27]
    personal_field = second[28:42]

    _fill_common_fields(
        result,
        document_number_field=document_number_field,
        document_number_check=second[9],
        nationality_field=second[10:13],
        birth_field=birth_field,
        birth_check=second[19],
        sex_field=second[20],
        expiry_field=expiry_field,
        expiry_check=second[27],
        personal_field=personal_field,
        personal_check=second[42],
        composite_field=second[0:10] + second[13:20] + second[21:43],
        composite_check=second[43],
    )


def _parse_td2(lines: list[str], result: MRZResult) -> None:
    """Documento de identidad: 2 lineas de 36 caracteres."""
    first, second = lines[0], lines[1]

    result.document_type = first[0:2].strip(FILLER) or None
    result.issuing_country = first[2:5].strip(FILLER) or None
    result.surname, result.given_names = _split_names(first[5:36])

    _fill_common_fields(
        result,
        document_number_field=second[0:9],
        document_number_check=second[9],
        nationality_field=second[10:13],
        birth_field=second[13:19],
        birth_check=second[19],
        sex_field=second[20],
        expiry_field=second[21:27],
        expiry_check=second[27],
        personal_field=second[28:35],
        personal_check=FILLER,
        composite_field=second[0:10] + second[13:20] + second[21:35],
        composite_check=second[35],
    )


def _parse_td1(lines: list[str], result: MRZResult) -> None:
    """Cedula / DNI: 3 lineas de 30 caracteres."""
    first, second, third = lines[0], lines[1], lines[2]

    result.document_type = first[0:2].strip(FILLER) or None
    result.issuing_country = first[2:5].strip(FILLER) or None
    result.surname, result.given_names = _split_names(third[0:30])

    _fill_common_fields(
        result,
        document_number_field=first[5:14],
        document_number_check=first[14],
        nationality_field=second[15:18],
        birth_field=second[0:6],
        birth_check=second[6],
        sex_field=second[7],
        expiry_field=second[8:14],
        expiry_check=second[14],
        personal_field=second[18:29],
        personal_check=FILLER,
        composite_field=first[5:30] + second[0:7] + second[8:15] + second[18:29],
        composite_check=second[29],
    )


def _fill_common_fields(
    result: MRZResult,
    *,
    document_number_field: str,
    document_number_check: str,
    nationality_field: str,
    birth_field: str,
    birth_check: str,
    sex_field: str,
    expiry_field: str,
    expiry_check: str,
    personal_field: str,
    personal_check: str,
    composite_field: str,
    composite_check: str,
) -> None:
    """Vuelca los campos compartidos y valida todos los digitos de control."""
    result.document_number = document_number_field.replace(FILLER, "").strip() or None
    result.nationality = nationality_field.replace(FILLER, "").strip() or None
    result.sex = sex_field if sex_field in {"M", "F"} else None
    personal_clean = personal_field.replace(FILLER, "").strip()
    result.personal_number = personal_clean or None

    result.birth_date = _expand_yymmdd(birth_field, kind="birth")
    result.expiry_date = _expand_yymmdd(expiry_field, kind="expiry")

    result.checks = {
        "document_number": verify_check_digit(document_number_field, document_number_check),
        "birth_date": verify_check_digit(birth_field, birth_check),
        "expiry_date": verify_check_digit(expiry_field, expiry_check),
        "composite": verify_check_digit(composite_field, composite_check),
    }

    # El numero personal solo se valida si el documento lo utiliza.
    if personal_field.strip(FILLER):
        result.checks["personal_number"] = verify_check_digit(personal_field, personal_check)

    if result.birth_date is None:
        result.errors.append("La fecha de nacimiento de la MRZ no es una fecha valida.")
    if result.expiry_date is None:
        result.errors.append("La fecha de caducidad de la MRZ no es una fecha valida.")


def validate_mrz_text(lines: Iterable[str]) -> MRZResult:
    """Alias de :func:`parse_mrz` para el endpoint de validacion."""
    return parse_mrz(lines)


def extract_and_validate(texts: Iterable[str]) -> MRZResult:
    """Busca la MRZ dentro del texto del OCR y la valida."""
    candidates = find_mrz_lines(texts)
    if not candidates:
        return MRZResult(detected=False, errors=["No se encontro una zona MRZ legible."])
    return parse_mrz(candidates)


__all__ = [
    "CHECK_WEIGHTS",
    "MRZ_ALPHABET",
    "MRZResult",
    "assign_slots",
    "MAX_MRZ_CANDIDATES",
    "compute_check_digit",
    "detect_format",
    "extract_and_validate",
    "find_mrz_lines",
    "normalize_line",
    "parse_mrz",
    "repair_line",
    "validate_mrz_text",
    "verify_check_digit",
]
