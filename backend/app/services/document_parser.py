"""Extraccion de los campos del documento a partir del texto del OCR.

El parser combina tres fuentes de informacion:

1. **MRZ** (cuando existe): es la fuente mas fiable porque admite validacion
   por digitos de control.
2. **Zonas visuales**: se buscan las etiquetas impresas en el documento
   ("APELLIDOS", "NOMBRES", "NUIP", "FECHA DE NACIMIENTO", "LUGAR DE
   NACIMIENTO", "FECHA DE VENCIMIENTO", "FECHA Y LUGAR DE EXPEDICION",
   "G S RH", ...) y se toma el valor que las acompana, ya sea en la misma
   linea o en la inmediatamente inferior.
3. **Codigo de verificacion** del reverso de los documentos colombianos, que
   aporta el numero de documento, el sexo y la fecha de impresion como
   respaldo y permite contrastar lo leido en las zonas visuales.

El analisis **no depende de cual de las caras se haya fotografiado**: todas las
pistas son etiquetas impresas, de modo que el mismo parser sirve para una sola
cara o para la union de la frontal y el reverso.

Cada campo final queda etiquetado con su origen para poder auditar la
extraccion.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from datetime import date
from typing import Any, Iterable, Sequence

from app.services.mrz import MRZResult, is_verification_code_line, parse_verification_code

# ---------------------------------------------------------------------------
# Vocabularios
# ---------------------------------------------------------------------------
# El orden importa: los tipos mas especificos van primero. La Tarjeta de
# Identidad comparte el encabezado "IDENTIFICACION PERSONAL" con la Cedula, asi
# que debe resolverse antes de mirar las pistas genericas.
TYPE_HINTS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("TARJETA DE IDENTIDAD", ("TARJETA DE IDENTIDAD", "T.I.")),
    ("PASAPORTE", ("PASAPORTE", "PASSPORT", "PASSEPORT")),
    ("DOCUMENTO NACIONAL DE IDENTIDAD", ("DOCUMENTO NACIONAL DE IDENTIDAD", "DNI")),
    ("CEDULA DE CIUDADANIA", ("CEDULA DE CIUDADANIA", "CEDULA CIUDADANIA")),
    ("CEDULA", ("CEDULA", "C.C.", "CC.", "CEDULA DE EXTRANJERIA", "IDENTIFICACION PERSONAL")),
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
    r"\b(?:NUIP|NIP|NO|N[RO]|NUM|NUMERO|No\.|DOC|DOCUMENTO|CEDULA|C\.C\.|CC|DNI|ID|"
    r"IDENTIFICACION|PASAPORTE|PASSPORT)\b"
)
BIRTH_LABEL_PATTERN = re.compile(r"\b(?:FECHA DE NACIMIENTO|FECHA NACIMIENTO|F\.?\s?NAC|NACIMIENTO|BIRTH|BORN)\b")
BIRTH_PLACE_LABEL_PATTERN = re.compile(
    r"\b(?:LUGAR DE NACIMIENTO|LUGAR DE NAC|SITIO DE NACIMIENTO|NACIDO EN|PLACE OF BIRTH)\b"
)
EXPIRY_LABEL_PATTERN = re.compile(r"\b(?:VENCIMIENTO|CADUCIDAD|EXPIRY|EXPIRES|FECHA DE VENCIMIENTO|VALID[OA]?)\b")
ISSUE_LABEL_PATTERN = re.compile(r"\b(?:FECHA Y LUGAR DE EXPEDICION|FECHA DE EXPEDICION|EXPEDICION|EXPEDIDO|EXPEDIDA|ISSUE[D]?)\b")
NAME_LABEL_PATTERN = re.compile(r"\b(?:NOMBRES|NOMBRE|APELLIDOS|APELLIDO|GIVEN NAMES|SURNAME)\b")
SURNAME_LABEL_PATTERN = re.compile(r"\b(?:APELLIDOS|APELLIDO|SURNAMES?)\b")
GIVEN_NAME_LABEL_PATTERN = re.compile(r"\b(?:NOMBRES|NOMBRE|GIVEN NAMES?)\b")
NATIONALITY_LABEL_PATTERN = re.compile(r"\b(?:NACIONALIDAD|NATIONALITY|NAC\.?)\b")
# El OCR confunde la "S" de "G S RH" con el simbolo del dolar y el "O" del
# grupo con un cero, por eso el patron admite esas variantes. Tambien se lee
# "G.S." sin la palabra RH al final.
BLOOD_LABEL_PATTERN = re.compile(
    r"\b(?:G\s*[.$5S]?\s*[.$5S]?\s*RH|GSRH|G\s*[.$5S]\s*[.$5S]|G\.?\s*SANG\.?|"
    r"GRUPO SANGUINEO|TIPO DE SANGRE|BLOOD (?:GROUP|TYPE))\b"
)
FIRMA_PATTERN = re.compile(r"\bFIRMA\b")

# Admite numeros puros (12345678), con prefijo de tipo (CC 12345678) y
# alfanumericos de pasaporte (L898902C3). Exige al menos 5 digitos para no
# confundirse con anos, folios cortos o numeros de via.
DOCUMENT_NUMBER_PATTERN = re.compile(r"\b(?P<value>[A-Z]{0,3}-?\d{5,12}[A-Z]{0,3}\d{0,3})\b")
# Numeros agrupados con puntos de miles, como el NUIP colombiano
# (1.033.186.199). El patron exige grupos de tres digitos para no tragarse una
# fecha escrita con puntos ("12.08.1974" no encaja).
DOTTED_NUMBER_PATTERN = re.compile(r"(?<![\d.,/])\d{1,3}(?:\.\d{3}){1,3}(?![\d.,/])")

DATE_PATTERN = re.compile(r"\b(\d{1,2})\s*[/\-.]\s*(\d{1,2})\s*[/\-.]\s*(\d{2,4})\b")
# "15 ABR 2004", "20-SEP-2008" y "20 DE ABRIL DE 2022".
LONG_DATE_PATTERN = re.compile(
    r"\b(\d{1,2})\s*[-/=]?\s*(?:DE\s+)?([A-Z]{3,12})\s*[-/=]?\s*(?:DE\s+)?(\d{2,4})\b"
)
COMPACT_DATE_PATTERN = re.compile(r"(?<!\d)(\d{4})(\d{2})(\d{2})(?!\d)")

BLOOD_VALUE_PATTERN = re.compile(
    r"(?<![A-Z0-9])(?P<group>AB|A|B|O|0)\s*(?P<rh>\+|[-])(?!\d)"
    # El orden inverso ("+0") aparece cuando el OCR lee la casilla al reves.
    r"|(?<![A-Z0-9])(?P<rh_rev>\+|[-])\s*(?P<group_rev>AB|A|B|O|0)(?![A-Z0-9])"
)

MONTHS_ES = {
    "ENE": 1, "FEB": 2, "MAR": 3, "ABR": 4, "MAY": 5, "JUN": 6,
    "JUL": 7, "AGO": 8, "SEP": 9, "SET": 9, "OCT": 10, "NOV": 11, "DIC": 12,
}

# Codigos ISO de los paises que mas aparecen en la recepcion.
COUNTRY_CODES = {
    "COLOMBIA": "COL", "VENEZUELA": "VEN", "ECUADOR": "ECU", "PERU": "PER",
    "PANAMA": "PAN", "MEXICO": "MEX", "CHILE": "CHL", "ARGENTINA": "ARG",
    "BRASIL": "BRA", "BOLIVIA": "BOL", "ESPANA": "ESP",
    "ESTADOS UNIDOS": "USA", "REPUBLICA DOMINICANA": "DOM", "CUBA": "CUB",
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
        "NUIP", "REGISTRADOR", "NOTARIO", "ALCALDE", "GERENTE", "DELEGADO", "SECRETARIO",
        "PERSONAL", "REVERSO", "FRONTAL", "CARA", "INDICE", "DERECHO", "IZQUIERDO",
        "PULGAR", "ANULAR", "HUELLA", "CODIGO", "VERIFICACION",
    }
)

# Palabras que **nunca** forman parte de un valor (etiquetas, titulos del
# documento o paises). Se usan para validar los valores tomados tras una
# etiqueta sin prohibir los apellidos compuestos ("DE LA CRUZ", "DEL RIO").
HARD_LABEL_WORDS = frozenset(
    {
        "REPUBLICA", "CEDULA", "CIUDADANIA", "EXTRANJERIA", "PASAPORTE", "PASSPORT",
        "TARJETA", "IDENTIDAD", "IDENTIFICACION", "DOCUMENTO", "NACIONAL", "REGISTRO",
        "CIVIL", "NOMBRES", "NOMBRE", "APELLIDOS", "APELLIDO", "GIVEN", "NAMES",
        "SURNAME", "FECHA", "NACIMIENTO", "EXPEDICION", "VENCIMIENTO", "CADUCIDAD",
        "SEXO", "NACIONALIDAD", "FIRMA", "NUMERO", "NUIP", "LUGAR", "GRUPO",
        "SANGUINEO", "RH", "REGISTRADOR", "NOTARIO", "MINISTERIO", "REGISTRADURIA",
        "COLOMBIA", "PERU", "MEXICO", "CHILE", "ARGENTINA", "ECUADOR", "VENEZUELA",
        "BOLIVIA", "PANAMA", "ESPANA", "LICENCIA", "CONDUCCION", "TIPO", "ESTATURA",
        "REVERSO", "FRONTAL", "UNITED", "STATES", "REPUBLIC", "IDENTITY", "BEARER",
        "INDICE", "DERECHO", "IZQUIERDO", "PULGAR", "ANULAR", "HUELLA", "CODIGO",
        "VERIFICACION", "SANGRE", "PAIS",
    }
)

# Particulas que si pueden formar parte de un nombre o de un lugar.
NAME_PARTICLES = frozenset({"DE", "DEL", "LA", "EL", "LOS", "LAS", "Y", "DA", "DI", "VAN", "VON"})

# Etiquetas que preceden al nombre de un funcionario: la linea siguiente es su
# nombre, no el del titular.
OFFICER_LABEL_PATTERN = re.compile(
    r"\b(?:REGISTRADOR|NOTARIO|ALCALDE|GERENTE|DELEGADO|SECRETARIO|FUNCIONARIO|AUTORIDAD)\b"
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

    first_surname: str | None = None
    second_surname: str | None = None
    given_names: str | None = None

    birth_date: str | None = None
    birth_place: str | None = None
    issue_date: str | None = None
    issue_place: str | None = None
    expiry_date: str | None = None
    sex: str | None = None
    blood_type: str | None = None

    nationality: str | None = None
    issuing_country: str | None = None

    confidence: float = 0.0
    sources: dict[str, str] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    @property
    def surnames(self) -> str | None:
        """Apellidos completos, en el orden impreso en el documento."""
        parts = [part for part in (self.first_surname, self.second_surname) if part]
        return " ".join(parts) if parts else None

    def as_dict(self) -> dict[str, Any]:
        return {
            "document_number": self.document_number,
            "name": self.name,
            "document_type": self.document_type,
            "first_surname": self.first_surname,
            "second_surname": self.second_surname,
            "given_names": self.given_names,
            "birth_date": self.birth_date,
            "birth_place": self.birth_place,
            "issue_date": self.issue_date,
            "issue_place": self.issue_place,
            "expiry_date": self.expiry_date,
            "sex": self.sex,
            "blood_type": self.blood_type,
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
    """Indica si un numero de 8 digitos encaja con una fecha.

    Se aceptan las dos maquetaciones habituales (``DDMMAAAA`` y
    ``AAAAMMDD``) para que un numero de documento no se confunda con una
    fecha escrita sin separadores.
    """
    digits = re.sub(r"\D", "", value)
    if len(digits) != 8:
        return False

    day, month = int(digits[0:2]), int(digits[2:4])
    year, month_alt, day_alt = int(digits[4:8]), int(digits[4:6]), int(digits[6:8])
    day_first = 1 <= month <= 12 and 1 <= day <= 31
    year_first = 1900 <= year <= 2100 and 1 <= month_alt <= 12 and 1 <= day_alt <= 31
    return day_first or year_first


def _clean_value(value: str) -> str:
    """Quita separadores y etiquetas que hayan quedado pegadas al valor."""
    cleaned = value.strip(" :;.-")
    cleaned = re.sub(r"^(?:DE|DEL|LA|EL)\s+", "", cleaned)
    return re.sub(r"\s+", " ", cleaned).strip(" :;.-")


def _cut_at_labels(value: str, patterns: Sequence[re.Pattern[str]]) -> str:
    """Recorta el valor en cuanto aparece otra etiqueta del documento."""
    cut = len(value)
    for pattern in patterns:
        match = pattern.search(value)
        if match:
            cut = min(cut, match.start())
    return value[:cut]


def _words(value: str) -> list[str]:
    return [word for word in re.split(r"[\s,]+", value) if word]


# ---------------------------------------------------------------------------
# Numero de documento
# ---------------------------------------------------------------------------
def _number_candidates(normalized: str) -> list[tuple[str, bool]]:
    """Pares ``(valor, agrupado_con_puntos)`` presentes en una linea."""
    candidates: list[tuple[str, bool]] = [
        (match.group("value").replace("-", ""), False)
        for match in DOCUMENT_NUMBER_PATTERN.finditer(normalized)
    ]
    candidates.extend((match.group(0).replace(".", ""), True) for match in DOTTED_NUMBER_PATTERN.finditer(normalized))
    return candidates


def extract_document_number(lines: Sequence[str]) -> tuple[str | None, float]:
    """Busca el numero de documento priorizando las lineas con etiqueta."""
    best_value: str | None = None
    best_score = -1.0

    for index, line in enumerate(lines):
        normalized = normalize_text(line)
        labelled = bool(NUMBER_LABEL_PATTERN.search(normalized))
        # Revisa tambien la linea siguiente: muchas veces el valor va debajo.
        nearby = labelled or (index > 0 and bool(NUMBER_LABEL_PATTERN.search(normalize_text(lines[index - 1]))))

        for value, dotted in _number_candidates(normalized):
            value = value.replace(" ", "")
            digits = re.sub(r"\D", "", value)
            if not (5 <= len(digits) <= 12):
                continue

            score = 0.0
            score += 2.0 if labelled else 0.0
            score += 1.0 if nearby else 0.0
            score += 1.0 if 7 <= len(digits) <= 10 else 0.0
            score += 0.4 if re.search(r"[A-Z]", value) else 0.0
            # El NUIP va agrupado con puntos de miles: es una pista fuerte.
            score += 1.2 if dotted else 0.0
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
# Tokens que el OCR inventa al transcribir la rubrica del titular y que se
# cuelan en el valor de NOMBRES ("SARCA.M.", "Fdo.". ...).
SIGNATURE_TOKEN_PATTERN = re.compile(r"^[A-Z]{1,6}\.(?:[A-Z]\.)+$|^[A-Z]{1,6}\.$")


def _clean_name_value(value: str) -> str:
    """Quita fragmentos de firma del valor de un campo de nombre."""
    kept = [word for word in _words(value) if not SIGNATURE_TOKEN_PATTERN.match(word)]
    return " ".join(kept)


def _looks_like_person_names(value: str) -> bool:
    """Heuristica: 1-5 palabras alfabeticas que no son etiquetas del documento."""
    value = _clean_name_value(value)
    words = _words(_clean_value(value))
    if not (1 <= len(words) <= 5):
        return False
    if any(word in HARD_LABEL_WORDS for word in words):
        return False

    return any(
        len(re.sub(r"[^A-ZÑÜ]", "", word)) >= 2 and word not in NAME_PARTICLES
        for word in words
    )


@dataclass(frozen=True, slots=True)
class _ValueOption:
    """Valor candidato de un campo y donde se encontro respecto a su rotulo."""

    value: str
    source: str  # "same" | "next" | "previous"
    line_index: int


# Los documentos rotulan sus valores de dos maneras: el rotulo encima del dato
# (Cedula nueva) o el rotulo debajo (Tarjeta de Identidad). El valor pegado al
# rotulo en la misma linea es la pista mas directa; las lineas vecinas se
# puntuan aparte.
_VALUE_SIDE_BASE = {"same": 1.2, "next": 1.0, "previous": 1.0}
_VALUE_LINE_OFFSET = {"same": 0, "next": 1, "previous": -1}


def _label_options(
    lines: Sequence[str],
    pattern: re.Pattern[str],
    *,
    stop_patterns: Sequence[re.Pattern[str]],
) -> list[_ValueOption]:
    """Valores candidatos que acompanan a una etiqueta del documento."""
    options: list[_ValueOption] = []

    for index, line in enumerate(lines):
        normalized = normalize_text(line)
        match = pattern.search(normalized)
        if not match:
            continue

        remainder = _clean_value(_cut_at_labels(normalized[match.end():], stop_patterns))
        cleaned = _clean_name_value(remainder)
        if cleaned and _looks_like_person_names(cleaned):
            options.append(_ValueOption(cleaned, "same", index))

        for source, position in (("next", index + 1), ("previous", index - 1)):
            if not 0 <= position < len(lines):
                continue
            neighbour = _clean_value(_cut_at_labels(normalize_text(lines[position]), stop_patterns))
            cleaned = _clean_name_value(neighbour)
            if cleaned and _looks_like_person_names(cleaned):
                options.append(_ValueOption(cleaned, source, index))

    return options


def _option_score(lines: Sequence[str], option: _ValueOption) -> float:
    """Puntua un candidato: posicion, contenido y cercania a la firma."""
    score = _VALUE_SIDE_BASE.get(option.source, 1.0)
    if _words(option.value):
        score += 0.5

    # La rubrica del titular se lee como texto suelto ("IRMA", "Iose"): si el
    # candidato esta pegado a la palabra FIRMA, casi seguro es basura.
    window = range(max(0, option.line_index - 1), min(len(lines), option.line_index + 2))
    if any(FIRMA_PATTERN.search(normalize_text(lines[position])) for position in window):
        score -= 2.0

    # La imprenta del documento va en mayusculas; el reconocedor transcribe la
    # firma manuscrita con minusculas mezcladas ("Iose"). Es una pista barata
    # para descartar el nombre del firmante.
    value_index = option.line_index + _VALUE_LINE_OFFSET[option.source]
    if 0 <= value_index < len(lines) and not lines[value_index].isupper():
        score -= 1.5

    return score


def extract_names(lines: Sequence[str]) -> tuple[str | None, str | None]:
    """Separa los apellidos y los nombres usando las etiquetas impresas.

    Cada rotulo aporta varios candidatos (el valor de su linea, el de la linea
    de arriba y el de la de abajo) y se elige la pareja mas coherente: los dos
    campos de un mismo documento comparten lado respecto a sus rotulos, lo que
    resuelve tanto la maquetacion "rotulo encima" como "rotulo debajo" sin
    confundir el ruido de la firma con el nombre del titular.

    Returns:
        ``(apellidos, nombres)`` tal como aparecen en el documento (sin
        dividir todavia en primer/segundo apellido).
    """
    surname_options = _label_options(
        lines,
        SURNAME_LABEL_PATTERN,
        stop_patterns=(GIVEN_NAME_LABEL_PATTERN, NAME_LABEL_PATTERN, BIRTH_LABEL_PATTERN),
    )
    given_options = _label_options(
        lines,
        GIVEN_NAME_LABEL_PATTERN,
        stop_patterns=(SURNAME_LABEL_PATTERN, NAME_LABEL_PATTERN, BIRTH_LABEL_PATTERN),
    )
    if not surname_options and not given_options:
        return None, None

    best_score = float("-inf")
    best_surnames: str | None = None
    best_given: str | None = None

    for surnames in surname_options or [None]:
        for given in given_options or [None]:
            score = 0.0
            if surnames is not None:
                score += _option_score(lines, surnames)
            if given is not None:
                score += _option_score(lines, given)
            if surnames is not None and given is not None:
                # Los dos campos del documento se rotulan del mismo lado.
                score += 1.5 if surnames.source == given.source else 0.0
                if surnames.value == given.value:
                    score -= 2.0

            if score > best_score:
                best_score = score
                best_surnames = surnames.value if surnames else None
                best_given = given.value if given else None

    return best_surnames, best_given


def extract_name(lines: Sequence[str], mr: MRZResult | None = None) -> tuple[str | None, float]:
    """Extrae el nombre del titular de las zonas visuales del documento.

    Se antepone el orden que usan los documentos colombianos (nombres y luego
    apellidos) para que el campo ``name`` sea legible de un vistazo.
    """
    if mr is not None and mr.full_name:
        return mr.full_name, 0.95

    surnames, given = extract_names(lines)
    if surnames or given:
        parts = [part for part in (given, surnames) if part]
        return " ".join(parts), 0.9 if (surnames and given) else 0.75

    best_name: str | None = None
    best_score = -1.0

    for index, line in enumerate(lines):
        normalized = normalize_text(line)
        # El nombre del funcionario que firma el reverso no es el titular.
        if index > 0 and OFFICER_LABEL_PATTERN.search(normalize_text(lines[index - 1])):
            continue

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
        if letters in NAME_STOPWORDS or letters in HARD_LABEL_WORDS:
            return False
        meaningful += 1

    return meaningful >= 2 and all(re.fullmatch(r"[A-ZÑÜ]+", word) for word in words)


def split_surnames(surnames: str | None) -> tuple[str | None, str | None]:
    """Divide los apellidos en primer y segundo apellido."""
    if not surnames:
        return None, None
    parts = _words(surnames)
    if not parts:
        return None, None
    if len(parts) == 1:
        return parts[0], None
    if len(parts) == 2:
        return parts[0], parts[1]
    # Con mas de dos palabras el resto se agrupa en el segundo apellido.
    return parts[0], " ".join(parts[1:])


# ---------------------------------------------------------------------------
# Fechas
# ---------------------------------------------------------------------------
def _parse_date(value: str) -> str | None:
    """Convierte una fecha del OCR al formato ISO ``YYYY-MM-DD``."""
    matches = _parse_date_candidates(value)
    return matches[0] if matches else None


def _parse_date_candidates(value: str) -> list[str]:
    """Todas las fechas presentes en el texto, en orden de aparicion."""
    found: list[str] = []

    def add(candidate: str | None) -> None:
        if candidate and candidate not in found:
            found.append(candidate)

    for match in DATE_PATTERN.finditer(value):
        day, month, year = int(match.group(1)), int(match.group(2)), int(match.group(3))
        if year < 100:
            year += 2000 if year < 40 else 1900
        if 1 <= month <= 12 and 1 <= day <= 31 and 1900 <= year <= 2100:
            add(f"{year:04d}-{month:02d}-{day:02d}")

    for match in LONG_DATE_PATTERN.finditer(value):
        day, month_name, year = int(match.group(1)), match.group(2)[:3], int(match.group(3))
        month = MONTHS_ES.get(month_name)
        if year < 100:
            year += 2000 if year < 40 else 1900
        if month and 1 <= day <= 31 and 1900 <= year <= 2100:
            add(f"{year:04d}-{month:02d}-{day:02d}")

    for match in COMPACT_DATE_PATTERN.finditer(value):
        year, month, day = int(match.group(1)), int(match.group(2)), int(match.group(3))
        if 1900 <= year <= 2100 and 1 <= month <= 12 and 1 <= day <= 31:
            add(f"{year:04d}-{month:02d}-{day:02d}")

    return found


def extract_date(
    lines: Sequence[str],
    pattern: re.Pattern[str],
    *,
    exclude: re.Pattern[str] | None = None,
    avoid: Sequence[re.Pattern[str]] = (),
) -> tuple[str | None, float]:
    """Busca la fecha asociada a una etiqueta (nacimiento / caducidad).

    Los documentos imprimen el valor de cada campo junto a su etiqueta de tres
    formas distintas, y las tres hay que resolverlas:

    * en la misma linea y detras (``VENCIMIENTO 15/04/2022``),
    * en la linea siguiente (``FECHA DE NACIMIENTO`` / ``12/08/1974``),
    * en la linea anterior, cuando el documento rotula *debajo* del dato
      (``20-SEP-2026`` sobre ``FECHA DE VENCIMIENTO``).

    Nunca se acepta una fecha que aparezca **antes** de la etiqueta dentro de
    la misma linea: el detector de OCR une a veces dos zonas del documento y
    deja la fecha de otra zona delante de la etiqueta (``15/04/2022 FECHA DE
    NACIMIENTO``); esa fecha no es su valor y tomarla producia fechas de
    nacimiento equivocadas.

    Args:
        exclude: Etiqueta que descarta la linea. Se usa para que ``LUGAR DE
            NACIMIENTO`` no se confunda con ``FECHA DE NACIMIENTO``.
        avoid: Otras etiquetas de fecha que descartan la linea como fuente. Una
            linea que ya lleva una etiqueta de otra fecha pertenece a esa otra
            fecha (por eso la caducidad no debe tomar la fecha de expedicion de
            la linea siguiente).
    """
    def rejected(text: str) -> bool:
        if exclude is not None and exclude.search(text):
            return True
        return any(other.search(text) for other in avoid)

    for index, line in enumerate(lines):
        normalized = normalize_text(line)
        if rejected(normalized):
            continue

        match = pattern.search(normalized)
        if not match:
            continue

        # 1) Fechas que siguen a la etiqueta en la misma linea. Si el OCR
        #    pego varias zonas ("... 30 ENE 2031 21 AGO 1988"), la de
        #    nacimiento es la mas antigua y la de vencimiento la mas futura.
        candidates = _parse_date_candidates(normalized[match.end():])
        if candidates:
            if pattern is BIRTH_LABEL_PATTERN:
                return min(candidates), 0.85
            if pattern is EXPIRY_LABEL_PATTERN:
                return max(candidates), 0.85
            return candidates[0], 0.85

        # Fechas vecinas que no llevan otra etiqueta de fecha en su linea.
        previous_date: str | None = None
        if index > 0:
            previous = normalize_text(lines[index - 1])
            if not rejected(previous):
                previous_date = _parse_date(previous)

        following_date: str | None = None
        if index + 1 < len(lines):
            following = normalize_text(lines[index + 1])
            if not rejected(following):
                following_date = _parse_date(following)

        # Layout colombiano "valor / rotulo / valor / rotulo": si hay fecha
        # arriba y abajo del rotulo, la de abajo pertenece al rotulo siguiente
        # (por ejemplo el vencimiento no debe tomar la fecha de expedicion
        # cuando el OCR separo "02-FEB-2016 MEDELLIN" de "FECHA Y LUGAR DE
        # EXPEDICION" en dos renglones).
        if previous_date and following_date and index + 2 < len(lines):
            after_following = normalize_text(lines[index + 2])
            if rejected(after_following) or pattern.search(after_following):
                following_date = None

        # 2) Etiqueta y valor en lineas distintas: el valor va debajo.
        if following_date:
            return following_date, 0.75

        # 3) El rotulo esta debajo del valor.
        if previous_date:
            return previous_date, 0.7

        # 4) Ultimo recurso: cualquier fecha de la linea. Cubre maquetaciones
        #    invertidas (``12/08/1974 NACIMIENTO``) con menor confianza.
        value = _parse_date(normalized)
        if value:
            return value, 0.6

    return None, 0.0


def extract_sex(lines: Sequence[str]) -> str | None:
    for index, line in enumerate(lines):
        normalized = normalize_text(line)
        if "MASCULINO" in normalized:
            return "M"
        if "FEMENINO" in normalized:
            return "F"

        if not re.search(r"\bSEXO\b", normalized):
            continue

        # 1) El valor acompana a la etiqueta en la misma linea ("SEXO M").
        if re.search(r"\bSEXO\b\s*:?[ ]*M\b", normalized):
            return "M"
        if re.search(r"\bSEXO\b\s*:?[ ]*F\b", normalized):
            return "F"

        # 2) La etiqueta va en un renglon de rotulos y el valor en el de
        #    arriba o en el de abajo ("FECHA DE VENCIMIENTO G S RH SEXO" sobre
        #    "20-SEP-2026 0+ M").
        for offset in (1, -1):
            position = index + offset
            if 0 <= position < len(lines):
                value = _standalone_sex(normalize_text(lines[position]))
                if value:
                    return value

    # 3) Sin etiqueta visible (captura con camara): el valor viaja en la misma
    #    linea que una fecha o el grupo sanguineo ("13-NOV-2026 0+ M").
    for line in lines:
        normalized = normalize_text(line)
        if not (_parse_date(normalized) or BLOOD_VALUE_PATTERN.search(normalized)):
            continue
        value = _standalone_sex(normalized)
        if value:
            return value

    return None


def _standalone_sex(text: str) -> str | None:
    """Busca un token suelto ``M`` o ``F`` (o su forma larga) en un renglon."""
    for token in re.split(r"[\s,.\-|]+", text):
        if token in {"M", "MASCULINO"}:
            return "M"
        if token in {"F", "FEMENINO"}:
            return "F"
    return None


# ---------------------------------------------------------------------------
# Lugares, expedicion y grupo sanguineo
# ---------------------------------------------------------------------------
def _looks_like_place(value: str) -> bool:
    """Heuristica: al menos tres letras y sin pinta de fecha ni de etiqueta."""
    cleaned = _clean_value(value)
    letters = re.sub(r"[^A-Z]", "", cleaned)
    if len(letters) < 3:
        return False
    # Ruido tipico del OCR ("C22TA", "3OGOTA"): digitos entremezclados con
    # letras no son un lugar legible.
    if any(char.isdigit() for char in cleaned):
        return False
    if _parse_date(cleaned):
        return False
    words = _words(cleaned)
    if not words or len(words) > 5:
        return False
    # Ninguna palabra puede ser una etiqueta del documento.
    return all(word.strip("()") not in HARD_LABEL_WORDS for word in words)


def _strip_date_and_labels(value: str) -> str:
    """Deja solo el lugar de un texto que mezcla fecha y lugar."""
    text = value
    match = DATE_PATTERN.search(text)
    if match:
        text = text[: match.start()] + " " + text[match.end():]
    match = LONG_DATE_PATTERN.search(text)
    if match:
        text = text[: match.start()] + " " + text[match.end():]
    text = re.sub(
        r"\b(?:FECHA|Y|LUGAR|DE|DEL|EXPEDICION|EXPEDIDO|NACIMIENTO|NACIDO|EN)\b",
        " ",
        text,
    )
    text = re.sub(r"\s+", " ", text)
    text = text.replace(",", " ").replace(".", " ")
    return _clean_value(re.sub(r"\s+", " ", text))


def extract_issue(lines: Sequence[str]) -> tuple[str | None, str | None, float]:
    """Extrae la fecha y el lugar de expedicion del documento.

    En los documentos colombianos ambos campos comparten etiqueta
    (``FECHA Y LUGAR DE EXPEDICION``). El valor puede ir detras de la etiqueta
    (``... EXPEDICION 20 ABR 2022, CARTAGENA``), en la linea siguiente
    (``FECHA Y LUGAR DE EXPEDICION`` / ``02-FEB-2016 MEDELLIN``) o delante
    (``02-FEB-2016 MEDELLIN FECHA Y LUGAR DE EXPEDICION``).
    """
    for index, line in enumerate(lines):
        normalized = normalize_text(line)
        match = ISSUE_LABEL_PATTERN.search(normalized)
        if not match:
            continue

        sources = [
            _clean_value(normalized[match.end():]),
            _clean_value(normalized[: match.start()]),
        ]
        if index + 1 < len(lines):
            sources.append(normalize_text(lines[index + 1]))
        if index > 0:
            sources.append(normalize_text(lines[index - 1]))

        issued: str | None = None
        place: str | None = None
        for source in sources:
            if not source:
                continue
            if issued is None:
                issued = _parse_date(source)
            if place is None:
                candidate = _strip_date_and_labels(source)
                if _looks_like_place(candidate):
                    place = candidate
            if issued and place:
                break

        if issued or place:
            return issued, place, 0.8 if (issued and place) else 0.7

    return None, None, 0.0


def extract_place(
    lines: Sequence[str],
    pattern: re.Pattern[str],
    *,
    stop_patterns: Sequence[re.Pattern[str]] = (),
) -> tuple[str | None, float]:
    """Extrae un lugar (nacimiento) a partir de su etiqueta."""
    for index, line in enumerate(lines):
        normalized = normalize_text(line)
        match = pattern.search(normalized)
        if not match:
            continue

        remainder = _clean_value(_cut_at_labels(normalized[match.end():], stop_patterns))
        if _looks_like_place(remainder):
            return remainder, 0.85

        if index + 1 < len(lines):
            following = _clean_value(_cut_at_labels(normalize_text(lines[index + 1]), stop_patterns))
            if _looks_like_place(following):
                return following, 0.7

        # El rotulo puede estar *debajo* del valor. Se sube mientras las lineas
        # sigan siendo parte del lugar y se unen en orden de lectura, lo que
        # tambien recompone un lugar partido en dos renglones
        # ("MEDELLIN" + "(ANTIOQUIA)").
        parts: list[str] = []
        cursor = index - 1
        while cursor >= 0 and len(parts) < 3:
            candidate = _clean_value(_cut_at_labels(normalize_text(lines[cursor]), stop_patterns))
            if not _looks_like_place(candidate):
                break
            parts.insert(0, candidate)
            cursor -= 1
        if parts:
            return " ".join(parts), 0.7

    return None, 0.0


def extract_blood_type(lines: Sequence[str]) -> tuple[str | None, float]:
    """Extrae el grupo sanguineo y el RH (``O+``, ``A-``, ...).

    El valor suele compartir renglon con los rotulos de vencimiento y sexo
    (``20-SEP-2026 0+ M`` sobre ``FECHA DE VENCIMIENTO G S RH SEXO``), pero
    tambien puede ir detras o debajo del rotulo: se prueban las tres
    posiciones.
    """
    for index, line in enumerate(lines):
        normalized = normalize_text(line)
        match = BLOOD_LABEL_PATTERN.search(normalized)
        if not match:
            continue

        sources = [normalized[match.end():]]
        for offset in (1, -1, 2, -2):
            position = index + offset
            if 0 <= position < len(lines):
                sources.append(normalize_text(lines[position]))

        for position, source in enumerate(sources):
            value = _find_blood_value(source)
            if value:
                return value, 0.85 if position == 0 else 0.7

    # Sin etiqueta visible: se acepta un token suelto del tipo "O+".
    for line in lines:
        value = _find_blood_value(normalize_text(line))
        if value:
            return value, 0.5

    return None, 0.0


def _find_blood_value(text: str) -> str | None:
    """Normaliza el valor del grupo sanguineo encontrado en un texto."""
    match = BLOOD_VALUE_PATTERN.search(text)
    if not match:
        return None

    # El reconocedor lee a menudo la letra O como un cero, y a veces invierte
    # el par ("+0" en lugar de "0+").
    if match.group("group") is not None:
        group = match.group("group")
        rh = match.group("rh")
    else:
        group = match.group("group_rev")
        rh = match.group("rh_rev")
    group = "O" if group == "0" else group
    return f"{group}{rh}"


def _normalize_country_code(value: str | None) -> str | None:
    """Corrige la confusion OCR 0/O en codigos ISO de tres letras (``C0L``)."""
    if not value:
        return value
    fixed = re.sub(r"0", "O", value.upper())
    if re.fullmatch(r"[A-Z]{3}", fixed):
        return fixed
    return value.upper()


def extract_nationality(lines: Sequence[str]) -> tuple[str | None, str | None]:
    """Devuelve ``(nacionalidad, pais emisor)`` en codigo ISO de tres letras."""
    nationality: str | None = None
    issuant: str | None = None

    for line in lines:
        normalized = normalize_text(line)
        for country, code in COUNTRY_CODES.items():
            if country in normalized:
                issuant = issuant or code

        match = NATIONALITY_LABEL_PATTERN.search(normalized)
        if match and nationality is None:
            value = _clean_value(normalized[match.end():])
            nationality = COUNTRY_CODES.get(value) or (
                _normalize_country_code(value) if re.fullmatch(r"[A-Z0O]{3}", value) else None
            )

    if nationality is None and issuant is not None:
        # En un documento nacional la nacionalidad del titular es la del pais
        # que lo emite cuando el propio documento no la imprime.
        nationality = issuant

    return nationality, issuant


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
# Fallbacks sin rotulos (captura con camara)
# ---------------------------------------------------------------------------
# Con la camara el OCR pierde los rotulos pequenos del documento y solo deja
# los valores grandes legibles. Estas funciones recuperan los campos usando
# la estructura tipica de la Tarjeta de Identidad / Cedula colombianas sin
# inventar datos: solo se aceptan candidatos con forma inequvoca.
def _collect_dates(lines: Sequence[str]) -> list[str]:
    """Fechas ISO presentes en las lineas, en orden de aparicion y sin repetir."""
    found: list[str] = []
    for line in lines:
        value = _parse_date(normalize_text(line))
        if value and value not in found:
            found.append(value)
    return found


def _is_likely_birth(value: str) -> bool:
    """Una fecha claramente antigua es de nacimiento, no de vencimiento."""
    return value <= f"{date.today().year - 10}-12-31"


def _is_likely_expiry(value: str) -> bool:
    """Una fecha reciente o futura es de vencimiento."""
    return value >= f"{date.today().year - 3}-01-01"


def _number_line_index(lines: Sequence[str], document_number: str | None) -> int | None:
    """Indice de la linea que imprime el numero de documento (valor visual)."""
    if not document_number:
        return None
    compact = re.sub(r"[\s.]", "", document_number)
    for index, line in enumerate(lines):
        line_compact = re.sub(r"[\s.]", "", normalize_text(line))
        if compact and compact in line_compact:
            return index
    for index, line in enumerate(lines):
        if DOTTED_NUMBER_PATTERN.search(normalize_text(line)):
            return index
    return None


def _name_words_from_line(line: str) -> list[str]:
    """Tokens de una linea que parecen letras impresas (no ruido del OCR)."""
    normalized = normalize_text(line)
    tokens: list[str] = []
    for token in re.split(r"[\s,]+", normalized):
        if not token:
            continue
        if not re.fullmatch(r"[A-ZÑÜ]{2,}", token):
            continue
        if token in HARD_LABEL_WORDS:
            continue
        tokens.append(token)
    return tokens


def _name_lines_without_labels(
    lines: Sequence[str],
    number_index: int | None,
    reserved: set[str],
) -> list[str]:
    """Lineas con forma de nombre entre el numero y la primera fecha.

    Solo se mira la ventana ``numero .. primera fecha``: es donde la Tarjeta
    de Identidad imprime apellidos y nombres. Se descartan lineas tocadas por
    ``FIRMA`` o por el rotulo de un funcionario, y candidatos ya tomados para
    otros campos.
    """
    if number_index is None:
        return []

    end = len(lines)
    for index in range(number_index + 1, len(lines)):
        if _parse_date(normalize_text(lines[index])):
            end = index
            break

    candidates: list[str] = []
    for index in range(number_index + 1, end):
        window = range(max(0, index - 1), min(len(lines), index + 2))
        if any(FIRMA_PATTERN.search(normalize_text(lines[position])) for position in window):
            continue
        if index > 0 and OFFICER_LABEL_PATTERN.search(normalize_text(lines[index - 1])):
            continue

        tokens = _name_words_from_line(lines[index])
        if not tokens:
            continue
        if all(token in NAME_PARTICLES for token in tokens):
            continue

        candidate = " ".join(tokens)
        if candidate.upper() in reserved:
            continue
        candidates.append(candidate)
    return candidates


def _reserved_upper(result: DocumentFields) -> set[str]:
    """Valores ya tomados, en mayusculas, para no reutilizarlos como nombre."""
    values = [
        result.document_number,
        result.name,
        result.first_surname,
        result.second_surname,
        result.given_names,
        result.birth_date,
        result.expiry_date,
        result.issue_date,
        result.birth_place,
        result.issue_place,
        result.blood_type,
        result.sex,
        result.nationality,
    ]
    return {value.upper() for value in values if value}


def _apply_date_fallback(
    result: DocumentFields,
    visual_lines: Sequence[str],
    confidence_terms: list[float],
) -> None:
    """Asigna fechas por posicion cuando los rotulos no se leyeron.

    Reglas (solo sobre fechas aun no usadas):

    * nacimiento = la mas antigua si hay dos o mas, o si parece antigua;
    * vencimiento = la mas reciente si parece de vencimiento o va despues del
      nacimiento y no parece de nacimiento;
    * el resto (o la del medio con tres fechas) = expedicion.
    """
    assigned = {result.birth_date, result.expiry_date, result.issue_date}
    pool = sorted(set(_collect_dates(visual_lines)) - assigned - {None})
    if not pool:
        return

    birth = result.birth_date
    expiry = result.expiry_date
    issue = result.issue_date

    if not birth and (len(pool) >= 2 or _is_likely_birth(pool[0])):
        birth = pool[0]

    if not expiry:
        candidate = max(pool)
        if candidate != birth and (
            _is_likely_expiry(candidate)
            or (birth is not None and candidate > birth and not _is_likely_birth(candidate))
        ):
            expiry = candidate

    if not issue:
        leftovers = [value for value in pool if value not in (birth, expiry)]
        if leftovers:
            issue = leftovers[0]

    if not birth and not expiry and not issue:
        # Ultimo recurso: una unica fecha y sin pistas de caducidad ni de
        # expedicion: se asume de nacimiento.
        birth = pool[0]

    for field_name, value in (
        ("birth_date", birth),
        ("expiry_date", expiry),
        ("issue_date", issue),
    ):
        if value and not getattr(result, field_name):
            setattr(result, field_name, value)
            result.sources[field_name] = "visual"
            confidence_terms.append(0.55)


def _apply_name_fallback(
    result: DocumentFields,
    visual_lines: Sequence[str],
    confidence_terms: list[float],
) -> None:
    """Apellidos y nombres por ventanilla cuando no hay rotulos legibles."""
    if result.first_surname and result.given_names:
        return

    number_index = _number_line_index(visual_lines, result.document_number)
    if number_index is None:
        return

    candidates = _name_lines_without_labels(visual_lines, number_index, _reserved_upper(result))
    if not candidates:
        return

    if len(candidates) >= 2:
        if not result.first_surname:
            first, second = split_surnames(candidates[0])
            if first:
                result.first_surname = first
                result.sources["first_surname"] = "visual"
                confidence_terms.append(0.55)
            if second and not result.second_surname:
                result.second_surname = second
                result.sources["second_surname"] = "visual"
        if not result.given_names:
            result.given_names = candidates[1]
            result.sources["given_names"] = "visual"
            confidence_terms.append(0.55)
    elif not result.first_surname and not result.given_names and not result.name:
        # Una sola linea inequiva: se toma como nombre completo del titular.
        result.name = candidates[0]
        result.sources["name"] = "visual"
        confidence_terms.append(0.55)
        return

    if not result.name and (result.given_names or result.surnames):
        result.name = " ".join(
            part for part in (result.given_names, result.surnames) if part
        )
        result.sources["name"] = result.sources.get("given_names", "visual")


def _apply_birth_place_fallback(
    result: DocumentFields,
    visual_lines: Sequence[str],
    confidence_terms: list[float],
) -> None:
    """Lugar de nacimiento: la linea de al lado de su fecha."""
    if result.birth_place or not result.birth_date:
        return

    for index, line in enumerate(visual_lines):
        if _parse_date(normalize_text(line)) != result.birth_date:
            continue
        for offset in (1, -1):
            position = index + offset
            if not 0 <= position < len(visual_lines):
                continue
            candidate = _clean_value(normalize_text(visual_lines[position]))
            if not _looks_like_place(candidate):
                continue
            if offset == 1 and position + 1 < len(visual_lines):
                following = _clean_value(normalize_text(visual_lines[position + 1]))
                if following.startswith("("):
                    candidate = f"{candidate} {following}"
            result.birth_place = candidate
            result.sources["birth_place"] = "visual"
            confidence_terms.append(0.55)
            return


def _apply_issue_place_fallback(
    result: DocumentFields,
    visual_lines: Sequence[str],
    confidence_terms: list[float],
) -> None:
    """Lugar de expedicion: el resto de la linea que lleva su fecha."""
    if result.issue_place or not result.issue_date:
        return

    for line in visual_lines:
        normalized = normalize_text(line)
        if _parse_date(normalized) != result.issue_date:
            continue
        candidate = _strip_date_and_labels(normalized)
        if candidate and _looks_like_place(candidate):
            result.issue_place = candidate
            result.sources["issue_place"] = "visual"
            confidence_terms.append(0.55)
            return


def _apply_nationality_fallback(
    result: DocumentFields,
    visual_lines: Sequence[str],
    confidence_terms: list[float],
) -> None:
    """NUIP con puntos = documento colombiano: nacionalidad COL."""
    if result.nationality:
        return
    if not any(DOTTED_NUMBER_PATTERN.search(normalize_text(line)) for line in visual_lines):
        return

    result.nationality = "COL"
    result.sources["nationality"] = "visual"
    confidence_terms.append(0.55)
    if not result.issuing_country:
        result.issuing_country = "COL"
        result.sources["issuing_country"] = "visual"


def _apply_label_free_fallbacks(
    result: DocumentFields,
    visual_lines: Sequence[str],
    confidence_terms: list[float],
) -> None:
    """Rellena los campos que los rotulos perdidos dejaron vacios."""
    if not visual_lines:
        return
    _apply_date_fallback(result, visual_lines, confidence_terms)
    _apply_name_fallback(result, visual_lines, confidence_terms)
    _apply_birth_place_fallback(result, visual_lines, confidence_terms)
    _apply_issue_place_fallback(result, visual_lines, confidence_terms)
    _apply_nationality_fallback(result, visual_lines, confidence_terms)


# ---------------------------------------------------------------------------
# Composicion
# ---------------------------------------------------------------------------
def parse_document(
    texts: Iterable[str],
    mrz: MRZResult | None = None,
) -> DocumentFields:
    """Extrae todos los campos del documento combinando MRZ y texto visible.

    La MRZ tiene prioridad porque sus valores estan respaldados por digitos de
    control; el texto visible actua como respaldo. Como todas las pistas son
    etiquetas impresas, la funcion acepta indistintamente el texto de una cara
    o la union del de la frontal y el reverso.
    """
    lines = build_lines(list(texts))

    # Las lineas de la MRZ y el codigo de verificacion de los documentos
    # colombianos se excluyen del analisis visual: contaminan la busqueda de
    # numeros y nombres.
    visual_lines = [
        line
        for line in lines
        if not _is_mrz_like(line) and not is_verification_code_line(line)
    ]

    result = DocumentFields()
    confidence_terms: list[float] = []

    # ---------------------------- MRZ (prioridad) --------------------------
    # Solo se confia en la MRZ cuando es consistente (todos los digitos de
    # control pasan) o cuando el campo concreto supero su propio control. Una
    # MRZ basura (texto visual detectado como TD3) no debe pisar los campos
    # que el texto visible ya leyo bien.
    if mrz is not None and mrz.detected:
        mrz_ok = mrz.valid or mrz.all_checks_passed
        # Sexo, nacionalidad y pais no tienen digito propio: si las fechas de
        # la linea 2 superan su control, la franja es real y bastan como
        # prueba estructural aunque el composite falle por un OCR malo del
        # numero de documento.
        structural_ok = mrz_ok or (
            bool(mrz.checks.get("birth_date")) and bool(mrz.checks.get("expiry_date"))
        )

        def _mrz_field_ok(check_name: str) -> bool:
            if mrz_ok:
                return True
            return bool(mrz.checks.get(check_name))

        if mrz.document_number and _mrz_field_ok("document_number"):
            result.document_number = mrz.document_number
            result.sources["document_number"] = "mrz"
            confidence_terms.append(0.95 if mrz.checks.get("document_number") else 0.75)
        if mrz.full_name and structural_ok:
            result.name = mrz.full_name
            result.sources["name"] = "mrz"
            confidence_terms.append(0.95)
        if mrz.surname and structural_ok:
            result.first_surname, result.second_surname = split_surnames(mrz.surname)
            result.sources["first_surname"] = "mrz"
            if result.second_surname:
                result.sources["second_surname"] = "mrz"
        if mrz.given_names and structural_ok:
            result.given_names = mrz.given_names
            result.sources["given_names"] = "mrz"
        if mrz.birth_date and _mrz_field_ok("birth_date"):
            result.birth_date = mrz.birth_date.isoformat()
            result.sources["birth_date"] = "mrz"
            confidence_terms.append(0.95 if mrz.checks.get("birth_date") else 0.75)
        if mrz.expiry_date and _mrz_field_ok("expiry_date"):
            result.expiry_date = mrz.expiry_date.isoformat()
            result.sources["expiry_date"] = "mrz"
            confidence_terms.append(0.95 if mrz.checks.get("expiry_date") else 0.75)
        if mrz.sex and structural_ok:
            result.sex = mrz.sex
            result.sources["sex"] = "mrz"
        if mrz.nationality and structural_ok:
            result.nationality = _normalize_country_code(mrz.nationality)
            result.sources["nationality"] = "mrz"
        if mrz.issuing_country and structural_ok:
            result.issuing_country = _normalize_country_code(mrz.issuing_country)
            result.sources["issuing_country"] = "mrz"

    # ------------------- Codigo de verificacion del reverso -----------------
    # No es una MRZ: es el codigo que imprime el reverso de la Tarjeta de
    # Identidad y de la Cedula. Se usa como contraste y como ultimo respaldo,
    # por eso se lee ahora y se aplica despues del texto visible.
    code = next(
        (parsed for line in lines if (parsed := parse_verification_code(line)) is not None),
        None,
    )

    # ------------------------ Texto visible --------------------------------
    if not result.document_number:
        value, score = extract_document_number(visual_lines)
        if value:
            result.document_number = value
            result.sources["document_number"] = "visual"
            confidence_terms.append(score)

    surnames, given = extract_names(visual_lines)
    if surnames:
        first, second = split_surnames(surnames)
        if not result.first_surname:
            result.first_surname = first
            result.sources["first_surname"] = "visual"
        if not result.second_surname and second:
            result.second_surname = second
            result.sources["second_surname"] = "visual"
    if given and not result.given_names:
        result.given_names = given
        result.sources["given_names"] = "visual"

    if not result.name:
        value, score = extract_name(visual_lines, mr=None)
        if value:
            result.name = value
            result.sources["name"] = "visual"
            confidence_terms.append(score)

    if not result.name and (result.given_names or result.surnames):
        result.name = " ".join(part for part in (result.given_names, result.surnames) if part)
        result.sources["name"] = result.sources.get("given_names", "visual")

    if not result.birth_date:
        value, score = extract_date(
            visual_lines,
            BIRTH_LABEL_PATTERN,
            exclude=BIRTH_PLACE_LABEL_PATTERN,
            avoid=(EXPIRY_LABEL_PATTERN, ISSUE_LABEL_PATTERN),
        )
        if value:
            result.birth_date = value
            result.sources["birth_date"] = "visual"
            confidence_terms.append(score)

    if not result.birth_place:
        value, score = extract_place(
            visual_lines,
            BIRTH_PLACE_LABEL_PATTERN,
            stop_patterns=(BIRTH_LABEL_PATTERN, ISSUE_LABEL_PATTERN, EXPIRY_LABEL_PATTERN),
        )
        if value:
            result.birth_place = value
            result.sources["birth_place"] = "visual"
            confidence_terms.append(score)

    if not result.expiry_date:
        value, score = extract_date(
            visual_lines,
            EXPIRY_LABEL_PATTERN,
            avoid=(BIRTH_LABEL_PATTERN, BIRTH_PLACE_LABEL_PATTERN, ISSUE_LABEL_PATTERN),
        )
        if value:
            result.expiry_date = value
            result.sources["expiry_date"] = "visual"
            confidence_terms.append(score)

    if not result.issue_date or not result.issue_place:
        value, place, score = extract_issue(visual_lines)
        if value and not result.issue_date:
            result.issue_date = value
            result.sources["issue_date"] = "visual"
            confidence_terms.append(score)
        if place and not result.issue_place:
            result.issue_place = place
            result.sources["issue_place"] = "visual"

    if not result.sex:
        result.sex = extract_sex(visual_lines)
        if result.sex:
            result.sources["sex"] = "visual"

    if not result.blood_type:
        value, score = extract_blood_type(visual_lines)
        if value:
            result.blood_type = value
            result.sources["blood_type"] = "visual"
            confidence_terms.append(score)

    nationality, issuant = extract_nationality(visual_lines)
    if nationality and not result.nationality:
        result.nationality = nationality
        result.sources["nationality"] = "visual"
    if issuant and not result.issuing_country:
        result.issuing_country = issuant
        result.sources["issuing_country"] = "visual"

    result.document_type = detect_document_type(visual_lines, mrz)
    if result.document_type:
        result.sources["document_type"] = "mrz" if (mrz and mrz.detected and mrz.document_type) else "visual"

    # ------------- Ultimo respaldo: codigo de verificacion del reverso ------
    # Solo rellena lo que no se haya leido en la MRZ ni en el texto visible: lo
    # impreso en el documento siempre es preferible.
    if code:
        if not result.document_number:
            digits = re.sub(r"\D", "", code["number"])
            if 5 <= len(digits) <= 12:
                result.document_number = digits
                result.sources["document_number"] = "barcode"
                confidence_terms.append(0.7)
        if not result.sex and code["sex"] in {"M", "F"}:
            result.sex = code["sex"]
            result.sources["sex"] = "barcode"
        if not result.issue_date:
            value = _parse_date(code["date"])
            if value:
                result.issue_date = value
                result.sources["issue_date"] = "barcode"

    # ------------- Fallbacks cuando el OCR no dejo rotulos legibles --------
    # La captura con camara suele perder las etiquetas pequenas del documento;
    # sin ellas los extractores anteriores solo rellenan el numero y la
    # sangre. Se completan fechas, nombres y lugares por su posicion tipica.
    _apply_label_free_fallbacks(result, visual_lines, confidence_terms)

    # ------------------------------- Coherencia ----------------------------
    if code and result.document_number:
        expected = re.sub(r"\D", "", code["number"])
        found = re.sub(r"\D", "", result.document_number)
        if expected and found and expected != found:
            result.warnings.append(
                "El numero de documento leido no coincide con el codigo de "
                "verificacion del reverso; revisa la fotografia."
            )

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
    "BIRTH_LABEL_PATTERN",
    "BIRTH_PLACE_LABEL_PATTERN",
    "BLOOD_LABEL_PATTERN",
    "DocumentFields",
    "EXPIRY_LABEL_PATTERN",
    "ISSUE_LABEL_PATTERN",
    "build_lines",
    "detect_document_type",
    "extract_blood_type",
    "extract_date",
    "extract_document_number",
    "extract_issue",
    "extract_name",
    "extract_names",
    "extract_nationality",
    "extract_place",
    "extract_sex",
    "normalize_text",
    "parse_document",
    "split_surnames",
]
