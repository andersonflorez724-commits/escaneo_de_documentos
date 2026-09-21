"""Extraccion de los campos del documento a partir del texto del OCR.

El parser combina dos fuentes de informacion:

1. **MRZ** (cuando existe): es la fuente mas fiable porque admite validacion
   por digitos de control.
2. **Zonas visuales**: se buscan las etiquetas impresas en el documento
   ("NOMBRES", "CEDULA", "FECHA DE NACIMIENTO", ...) y se toma el valor que
   las acompania, ya sea en la misma linea o en la inmediatamente inferior.

Cada campo final queda etiquetado con su origen para poder auditar la
extraccion.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

from app.services.mrz import MRZResult

# ---------------------------------------------------------------------------
# Vocabularios
# ---------------------------------------------------------------------------
TYPE_HINTS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("CEDULA DE CIUDADANIA", ("CEDULA DE CIUDADANIA", "CEDULA CIUDADANIA")),
    ("PASAPORTE", ("PASAPORTE", "PASSPORT", "PASSEPORT")),
    ("DOCUMENTO NACIONAL DE IDENTIDAD", ("DOCUMENTO NACIONAL DE IDENTIDAD", "DNI")),
    ("CEDULA", ("CEDULA", "C.C.", "CC.", "CEDULA DE EXTRANJERIA")),
    ("TARJETA DE IDENTIDAD", ("TARJETA DE IDENTIDAD", "T.I.")),
    ("LICENCIA DE CONDUCCION", ("LICENCIA DE CONDUCCION", "DRIVER")),
    ("REGISTRO CIVIL", ("REGISTRO CIVIL",)),
)

DOCUMENT_TYPE_BY_MRZ_CODE = {
    "P": "PASAPORTE",
    "I": "DOCUMENTO DE IDENTIDAD",
    "A": "DOCUMENTO DE IDENTIDAD",
    "C": "DOCUMENTO DE IDENTIDAD",
    "V": "VISA",
}

NUMBER_LABEL_PATTERN = re.compile(
    r"\b(?:NO|N[RO]|NUM|NUMERO|No\.|DOC|DOCUMENTO|CEDULA|C\.C\.|CC|DNI|ID|IDENTIFICACION|PASAPORTE|PASSPORT)\b"
)
BIRTH_LABEL_PATTERN = re.compile(r"\b(?:NACIMIENTO|FECHA DE NAC|F\. NAC|BIRTH|BORN)\b")
EXPIRY_LABEL_PATTERN = re.compile(r"\b(?:VENCIMIENTO|CADUCIDAD|EXPEDICION|EXPEDITION|EXPIRY|VALID[OA]?)\b")
NAME_LABEL_PATTERN = re.compile(r"\b(?:NOMBRES|NOMBRE|APELLIDOS|APELLIDO|GIVEN NAMES|SURNAME)\b")

# Admite numeros puros (12345678), con prefijo de tipo (CC 12345678) y
# alfanumericos de pasaporte (L898902C3). Exige al menos 5 digitos para no
# confundirse con anos, folios cortos o numeros de via.
DOCUMENT_NUMBER_PATTERN = re.compile(r"\b(?P<value>[A-Z]{0,3}-?\d{5,12}[A-Z]{0,3}\d{0,3})\b")
DATE_PATTERN = re.compile(r"\b(\d{1,2})[/\-.](\d{1,2})[/\-.](\d{2,4})\b")
LONG_DATE_PATTERN = re.compile(r"\b(\d{1,2})\s+(?:DE\s+)?([A-Z]{3,10})\s+(?:DE\s+)?(\d{4})\b")

MONTHS_ES = {
    "ENE": 1, "FEB": 2, "MAR": 3, "ABR": 4, "MAY": 5, "JUN": 6,
    "JUL": 7, "AGO": 8, "SEP": 9, "SET": 9, "OCT": 10, "NOV": 11, "DIC": 12,
}

# Palabras que nunca forman parte de un nombre de persona.
NAME_STOPWORDS = frozenset(
    {
        "REPUBLICA", "DE", "DEL", "LA", "EL", "LOS", "LAS", "COLOMBIA", "PERU", "MEXICO",
        "CHILE", "ARGENTINA", "ECUADOR", "VENEZUELA", "BOLIVIA", "PANAMA", "ESPANA",
        "CEDULA", "CIUDADANIA", "EXTRANJERIA", "PASAPORTE", "PASSPORT", "TARJETA",
        "IDENTIDAD", "IDENTIFICACION", "DOCUMENTO", "NACIONAL", "REGISTRO", "CIVIL",
        "NOMBRES", "NOMBRE", "APELLIDOS", "APELLIDO", "GIVEN", "NAMES", "SURNAME",
        "FECHA", "NACIMIENTO", "EXPEDICION", "VENCIMIENTO", "CADUCIDAD", "SEXO",
        "NACIONALIDAD", "FIRMA", "REPUBLIC", "REGISTRADURIA", "MINISTERIO", "LICENCIA",
        "CONDUCCION", "CATEGORIA", "RESTRICCION", "ORGANISMO", "TRANSITO", "EMISION",
        "PRIMER", "SEGUNDO", "LUGAR", "ESTATURA", "GRUPO", "SANGUINEO", "RH", "TIPO",
        "MUESTRA", "SIN", "VALIDEZ", "TITULAR", "MENOR", "ADULTO", "MASCULINO", "FEMENINO",
        "NUMERO", "CARD", "NATIONAL", "UNITED", "STATES", "IDENTITY", "ID", "BEARER",
    }
)


def normalize_text(text: str) -> str:
    """Mayusculas sin acentos, util para comparar con las etiquetas."""
    decomposed = unicodedata.normalize("NFKD", str(text))
    without_accents = "".join(char for char in decomposed if not unicodedata.combining(char))
    return re.sub(r"\s+", " ", without_accents).upper().strip()


# ---------------------------------------------------------------------------
# Resultado
# ---------------------------------------------------------------------------
@dataclass(slots=True)
class DocumentFields:
    """Campos finales extraidos del documento."""

    document_number: str | None = None
    name: str | None = None
    document_type: str | None = None
    birth_date: str | None = None
    expiry_date: str | None = None
    sex: str | None = None
    nationality: str | None = None
    issuing_country: str | None = None

    confidence: float = 0.0
    sources: dict[str, str] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "document_number": self.document_number,
            "name": self.name,
            "document_type": self.document_type,
            "birth_date": self.birth_date,
            "expiry_date": self.expiry_date,
            "sex": self.sex,
            "nationality": self.nationality,
            "issuing_country": self.issuing_country,
            "confidence": round(self.confidence, 4),
            "sources": dict(self.sources),
            "warnings": list(self.warnings),
        }


# ---------------------------------------------------------------------------
# Utilidades de lineas
# ---------------------------------------------------------------------------
def build_lines(texts: Sequence[str]) -> list[str]:
    """Normaliza y limpia las lineas candidatas del OCR."""
    lines: list[str] = []
    for text in texts:
        for raw in str(text).splitlines():
            cleaned = re.sub(r"\s+", " ", raw).strip()
            if len(cleaned) >= 2:
                lines.append(cleaned)
    return lines


def _is_probable_date(value: str) -> bool:
    digits = re.sub(r"\D", "", value)
    if len(digits) != 8:
        return False
    month, day = int(digits[4:6]), int(digits[6:8])
    return 1 <= month <= 12 and 1 <= day <= 31


# ---------------------------------------------------------------------------
# Numero de documento
# ---------------------------------------------------------------------------
def extract_document_number(lines: Sequence[str]) -> tuple[str | None, float]:
    """Busca el numero de documento priorizando las lineas con etiqueta."""
    best_value: str | None = None
    best_score = -1.0

    for index, line in enumerate(lines):
        normalized = normalize_text(line)
        labelled = bool(NUMBER_LABEL_PATTERN.search(normalized))
        # Revisa tambien la linea siguiente: muchas veces el valor va debajo.
        nearby = labelled or (index > 0 and bool(NUMBER_LABEL_PATTERN.search(normalize_text(lines[index - 1]))))

        for match in DOCUMENT_NUMBER_PATTERN.finditer(normalized):
            value = match.group("value").replace("-", "").replace(" ", "")
            digits = re.sub(r"\D", "", value)
            if not (5 <= len(digits) <= 12):
                continue

            score = 0.0
            score += 2.0 if labelled else 0.0
            score += 1.0 if nearby else 0.0
            score += 1.0 if 7 <= len(digits) <= 10 else 0.0
            score += 0.4 if re.search(r"[A-Z]", value) else 0.0
            score -= 1.6 if _is_probable_date(value) else 0.0
            # Un numero de documento no es un ano suelto ni un telefono largo.
            score -= 1.2 if len(digits) == 4 else 0.0

            if score > best_score:
                best_score = score
                best_value = value

    return best_value, max(0.0, min(1.0, best_score / 4.0))


# ---------------------------------------------------------------------------
# Nombre
# ---------------------------------------------------------------------------
def extract_name(lines: Sequence[str], mr: MRZResult | None = None) -> tuple[str | None, float]:
    """Extrae el nombre del titular de las zonas visuales del documento."""
    if mr is not None and mr.full_name:
        return mr.full_name, 0.95

    best_name: str | None = None
    best_score = -1.0

    for index, line in enumerate(lines):
        normalized = normalize_text(line)
        if NAME_LABEL_PATTERN.search(normalized):
            # El valor puede estar en la misma linea o en la siguiente.
            remainder = NAME_LABEL_PATTERN.sub(" ", normalized).strip(" :.-")
            candidate_source = remainder if _looks_like_name(remainder) else None
            if candidate_source is None and index + 1 < len(lines):
                following = normalize_text(lines[index + 1])
                candidate_source = following if _looks_like_name(following) else None
            if candidate_source:
                return candidate_source, 0.9

        if _looks_like_name(normalized):
            word_count = len(normalized.split())
            score = word_count * 1.0 + (1.0 if index < len(lines) / 2 else 0.0)
            if score > best_score:
                best_score = score
                best_name = normalized

    return best_name, max(0.0, min(0.85, best_score / 6.0))


def _looks_like_name(value: str) -> bool:
    """Heuristica: 2-6 palabras alfabeticas, sin palabras de formulario."""
    words = [word for word in re.split(r"[\s,]+", value) if word]
    if not (2 <= len(words) <= 6):
        return False

    meaningful = 0
    for word in words:
        letters = re.sub(r"[^A-ZÑÜ]", "", word)
        if len(letters) < 2:
            return False
        if letters in NAME_STOPWORDS:
            continue
        meaningful += 1

    return meaningful >= 2 and all(re.fullmatch(r"[A-ZÑÜ]+", word) for word in words)


# ---------------------------------------------------------------------------
# Fechas
# ---------------------------------------------------------------------------
def _parse_date(value: str) -> str | None:
    """Convierte una fecha del OCR al formato ISO ``YYYY-MM-DD``."""
    match = DATE_PATTERN.search(value)
    if match:
        day, month, year = int(match.group(1)), int(match.group(2)), int(match.group(3))
        if year < 100:
            year += 2000 if year < 40 else 1900
        if 1 <= month <= 12 and 1 <= day <= 31 and 1900 <= year <= 2100:
            return f"{year:04d}-{month:02d}-{day:02d}"

    match = LONG_DATE_PATTERN.search(value)
    if match:
        day, month_name, year = int(match.group(1)), match.group(2)[:3], int(match.group(3))
        month = MONTHS_ES.get(month_name)
        if month and 1 <= day <= 31:
            return f"{year:04d}-{month:02d}-{day:02d}"

    return None


def extract_date(lines: Sequence[str], pattern: re.Pattern[str]) -> tuple[str | None, float]:
    """Busca la fecha asociada a una etiqueta (nacimiento / caducidad)."""
    for index, line in enumerate(lines):
        normalized = normalize_text(line)
        if not pattern.search(normalized):
            continue

        value = _parse_date(normalized)
        if value:
            return value, 0.85
        remainder = pattern.sub(" ", normalized).strip(" :.-")
        value = _parse_date(remainder)
        if value:
            return value, 0.8
        if index + 1 < len(lines):
            value = _parse_date(normalize_text(lines[index + 1]))
            if value:
                return value, 0.75

    return None, 0.0


def extract_sex(lines: Sequence[str]) -> str | None:
    for line in lines:
        normalized = normalize_text(line)
        if "MASCULINO" in normalized or re.search(r"\bSEXO\b\s*:?\s*M\b", normalized):
            return "M"
        if "FEMENINO" in normalized or re.search(r"\bSEXO\b\s*:?\s*F\b", normalized):
            return "F"
    return None


def detect_document_type(lines: Sequence[str], mr: MRZResult | None = None) -> str | None:
    """Clasifica el tipo de documento por pistas del texto o de la MRZ."""
    for type_name, hints in TYPE_HINTS:
        for line in lines:
            normalized = normalize_text(line)
            if any(hint in normalized for hint in hints):
                return type_name

    if mr is not None and mr.document_type:
        return DOCUMENT_TYPE_BY_MRZ_CODE.get(mr.document_type[0], mr.document_type)
    return None


# ---------------------------------------------------------------------------
# Composicion
# ---------------------------------------------------------------------------
def parse_document(texts: Iterable[str], mrz: MRZResult | None = None) -> DocumentFields:
    """Extrae todos los campos del documento combinando MRZ y texto visible.

    La MRZ tiene prioridad porque sus valores estan respaldados por digitos de
    control; el texto visible actua como respaldo.
    """
    lines = build_lines(list(texts))

    # Las lineas de la propia MRZ se excluyen del analisis visual: contaminan
    # la busqueda de numeros y nombres.
    visual_lines = [line for line in lines if mrz is None or not _is_mrz_like(line)]

    result = DocumentFields()
    confidence_terms: list[float] = []

    # ---------------------------- MRZ (prioridad) --------------------------
    if mrz is not None and mrz.detected:
        if mrz.document_number:
            result.document_number = mrz.document_number
            result.sources["document_number"] = "mrz"
            confidence_terms.append(0.95 if mrz.checks.get("document_number") else 0.7)
        if mrz.full_name:
            result.name = mrz.full_name
            result.sources["name"] = "mrz"
            confidence_terms.append(0.95)
        if mrz.birth_date:
            result.birth_date = mrz.birth_date.isoformat()
            result.sources["birth_date"] = "mrz"
            confidence_terms.append(0.95 if mrz.checks.get("birth_date") else 0.7)
        if mrz.expiry_date:
            result.expiry_date = mrz.expiry_date.isoformat()
            result.sources["expiry_date"] = "mrz"
            confidence_terms.append(0.95 if mrz.checks.get("expiry_date") else 0.7)
        if mrz.sex:
            result.sex = mrz.sex
            result.sources["sex"] = "mrz"
        if mrz.nationality:
            result.nationality = mrz.nationality
            result.sources["nationality"] = "mrz"
        if mrz.issuing_country:
            result.issuing_country = mrz.issuing_country
            result.sources["issuing_country"] = "mrz"

    # ------------------------ Respaldo: texto visible ----------------------
    if not result.document_number:
        value, score = extract_document_number(visual_lines)
        if value:
            result.document_number = value
            result.sources["document_number"] = "visual"
            confidence_terms.append(score)

    if not result.name:
        value, score = extract_name(visual_lines, mr=None)
        if value:
            result.name = value
            result.sources["name"] = "visual"
            confidence_terms.append(score)

    if not result.birth_date:
        value, score = extract_date(visual_lines, BIRTH_LABEL_PATTERN)
        if value:
            result.birth_date = value
            result.sources["birth_date"] = "visual"
            confidence_terms.append(score)

    if not result.expiry_date:
        value, score = extract_date(visual_lines, EXPIRY_LABEL_PATTERN)
        if value:
            result.expiry_date = value
            result.sources["expiry_date"] = "visual"
            confidence_terms.append(score)

    if not result.sex:
        result.sex = extract_sex(visual_lines)
        if result.sex:
            result.sources["sex"] = "visual"

    result.document_type = detect_document_type(visual_lines, mrz)
    if result.document_type:
        result.sources["document_type"] = "mrz" if (mrz and mrz.detected and mrz.document_type) else "visual"

    # ------------------------------- Confianza -----------------------------
    if confidence_terms:
        result.confidence = sum(confidence_terms) / len(confidence_terms)

    if not result.document_number:
        result.warnings.append("No se pudo leer el numero de documento.")
    if not result.name:
        result.warnings.append("No se pudo leer el nombre del titular.")
    if mrz is not None and mrz.detected and not mrz.all_checks_passed:
        result.warnings.append("La MRZ no supero todos los digitos de control.")

    return result


def _is_mrz_like(line: str) -> bool:
    """Detecta si una linea parece pertenecer a la zona MRZ."""
    compact = re.sub(r"\s+", "", line)
    if len(compact) < 28:
        return False
    if not re.fullmatch(r"[A-Z0-9<]+", compact):
        return False
    fillers = compact.count("<")
    return fillers >= 3 or (len(compact) >= 30 and fillers > 0)


__all__ = [
    "DocumentFields",
    "build_lines",
    "detect_document_type",
    "extract_date",
    "extract_document_number",
    "extract_name",
    "extract_sex",
    "normalize_text",
    "parse_document",
]
