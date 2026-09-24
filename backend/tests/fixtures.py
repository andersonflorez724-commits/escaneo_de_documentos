"""Generadores de imagenes sinteticas para las pruebas.

Se dibuja un documento de identificacion ficticio con OpenCV para poder
ejercitar el pipeline completo (deteccion del contorno, OCR real con
EasyOCR y validacion de la MRZ) sin depender de documentos ni de datos
personales reales.

La MRZ se genera con los digitos de control que exige el estandar ICAO 9303,
de modo que el documento de prueba sea **coherente consigo mismo**.
"""

from __future__ import annotations

import cv2
import numpy as np

from app.services.mrz import FILLER, compute_check_digit

# Pasaporte de ejemplo del estandar ICAO 9303 (vectores oficiales).
ICAO_TD3_LINES = (
    "P<UTOERIKSSON<<ANNA<MARIA<<<<<<<<<<<<<<<<<<<",
    "L898902C36UTO7408122F1204159ZE184226B<<<<<10",
)

# Texto que el OCR entrega sobre una Tarjeta de Identidad colombiana real.
#
# La cara frontal lleva el numero y el nombre con los rotulos **debajo** del
# valor, y la rubrica del titular se cuela entre las lineas; el reverso lleva
# las fechas, el lugar de nacimiento, el grupo sanguineo y el codigo de
# verificacion. Sirve para comprobar que el parser no confunde la firma con el
# nombre ni la fecha de expedicion con la de vencimiento.
TARJETA_IDENTIDAD_FRONT_TEXT = [
    "REPÚBLICA DE COLOMBIA",
    "5 6 IDENTIFICACIÓN PERSONAL",
    "FIRMA TARJETA DE IDENTIDAD",
    "NÚMERO 5",
    "1.033.186.199",
    "OCAMPO ARTEAGA",
    "APELLIDOS IRMA",
    "JUAN JOSE",
    "NOMBRES",
    "Iose",
    "FIRMA",
]

TARJETA_IDENTIDAD_BACK_TEXT = [
    "FECHA DE NACIMIENTO 20-SEP-2008",
    "MEDELLIN",
    "(ANTIOQUIA)",
    "LUGAR DE NACIMIENTO",
    "20-SEP-2026 0+ M",
    "FECHA DE VENCIMIENTO G $ RH SEXO",
    "02-FEB-2016 MEDELLIN FECHA Y LUGAR DE EXPEDICIÓN |",
    "INDiCE Derecho P-0100150-00799122-M-1033186199-20160309 0048868905A "
    "REGISTRADOAINACIONAL 45514468",
    "P-0100150-00799122-M-1033186199-20160309 0048866905A 45514466",
]

# Proporcion cercana a una tarjeta de identificacion real (85,6 x 54 mm).
DOCUMENT_WIDTH = 1400
DOCUMENT_HEIGHT = 880

FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_MONO = cv2.FONT_HERSHEY_DUPLEX
TD3_LINE_LENGTH = 44


# ---------------------------------------------------------------------------
# MRZ valida
# ---------------------------------------------------------------------------
def build_td3_mrz(
    *,
    document_number: str = "12345678",
    surname: str = "PEREZ",
    given_names: str = "JUAN",
    issuing_country: str = "COL",
    nationality: str = "COL",
    birth: str = "740812",
    expiry: str = "220415",
    sex: str = "M",
    personal_number: str = "",
) -> tuple[str, str]:
    """Construye una MRZ TD3 valida para los datos indicados.

    Los digitos de control se calculan con el algoritmo ICAO 9293, el mismo
    que usa el validador del backend.
    """
    name_field = f"{surname}<<{given_names}".replace(" ", FILLER)
    line1 = f"P<{issuing_country}{name_field}"[:TD3_LINE_LENGTH].ljust(TD3_LINE_LENGTH, FILLER)

    doc_field = document_number.ljust(9, FILLER)
    personal_field = personal_number.ljust(14, FILLER)

    line2 = (
        doc_field
        + compute_check_digit(doc_field)
        + nationality.ljust(3, FILLER)
        + birth
        + compute_check_digit(birth)
        + sex
        + expiry
        + compute_check_digit(expiry)
        + personal_field
        + compute_check_digit(personal_field)
    )
    composite_field = line2[0:10] + line2[13:20] + line2[21:43]
    line2 += compute_check_digit(composite_field)

    assert len(line1) == TD3_LINE_LENGTH, len(line1)
    assert len(line2) == TD3_LINE_LENGTH, len(line2)
    return line1, line2


# ---------------------------------------------------------------------------
# Dibujo
# ---------------------------------------------------------------------------
def _put(
    image: np.ndarray,
    text: str,
    x: int,
    y: int,
    *,
    scale: float = 1.0,
    color: tuple[int, int, int] = (25, 25, 25),
    thickness: int = 2,
    mono: bool = False,
) -> None:
    cv2.putText(
        image,
        text,
        (x, y),
        FONT_MONO if mono else FONT,
        scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


def render_face(image: np.ndarray, x: int, y: int, width: int = 280, height: int = 350) -> None:
    """Dibuja un rostro sintetico (oval + ojos + nariz + boca)."""
    center = (x + width // 2, y + height // 2)
    axes = (width // 2, height // 2)
    cv2.ellipse(image, center, axes, 0, 0, 360, (168, 190, 214), -1)
    cv2.ellipse(image, center, axes, 0, 0, 360, (120, 140, 165), 3)

    eye_y = y + int(height * 0.38)
    eye_dx = int(width * 0.20)
    for offset in (-eye_dx, eye_dx):
        cv2.ellipse(image, (center[0] + offset, eye_y), (30, 16), 0, 0, 360, (250, 250, 250), -1)
        cv2.circle(image, (center[0] + offset, eye_y), 12, (55, 45, 40), -1)
        cv2.ellipse(image, (center[0] + offset, eye_y - 22), (30, 8), 0, 0, 360, (70, 55, 50), 3)

    cv2.line(image, (center[0], eye_y + 18), (center[0] - 9, eye_y + 70), (130, 145, 165), 4)
    cv2.ellipse(image, (center[0], y + int(height * 0.72)), (38, 19), 0, 0, 180, (95, 70, 75), 6)
    cv2.ellipse(image, (center[0], y + int(height * 0.74)), (32, 15), 0, 0, 180, (60, 40, 45), 4)


def render_synthetic_document(
    *,
    document_number: str = "12345678",
    name: str = "JUAN PEREZ",
    tax_id: str = "CC 12345678",
    birth_text: str = "12/08/1974",
    expiry_text: str = "15/04/2022",
    with_mrz: bool = True,
    with_face: bool = True,
    mrz_lines: tuple[str, str] | None = None,
    background: tuple[int, int, int] = (238, 240, 245),
    noise: float = 0.0,
) -> np.ndarray:
    """Genera la imagen (BGR) de un documento de identificacion ficticio.

    Si no se indican ``mrz_lines`` se construye una MRZ coherente con los
    datos impresos en el propio documento.
    """
    if mrz_lines is None:
        mrz_lines = build_td3_mrz(document_number=document_number)

    image = np.full((DOCUMENT_HEIGHT, DOCUMENT_WIDTH, 3), background, dtype=np.uint8)
    cv2.rectangle(image, (10, 10), (DOCUMENT_WIDTH - 11, DOCUMENT_HEIGHT - 11), (60, 70, 95), 4)

    _put(image, "REPUBLICA DE COLOMBIA", 55, 88, scale=1.35, color=(35, 45, 95), thickness=3)
    _put(image, "CEDULA DE CIUDADANIA", 55, 142, scale=1.05, color=(60, 60, 60))
    cv2.line(image, (55, 162), (1010, 162), (90, 100, 130), 2)

    _put(image, "NUMERO", 55, 225, scale=0.85, color=(90, 95, 110))
    _put(image, tax_id, 55, 275, scale=1.25, thickness=3)

    _put(image, "NOMBRES", 55, 360, scale=0.85, color=(90, 95, 110))
    _put(image, name, 55, 410, scale=1.30, thickness=3)

    # La etiqueta y su valor van en la misma linea, como en la mayoria de
    # documentos reales ("FECHA NAC. 12/08/1974"). Asi la extraccion sigue
    # siendo correcta aunque el detector de OCR una varias zonas del
    # documento en un solo bloque: el valor siempre queda detras de su
    # etiqueta. La variante con el valor en la linea de abajo se cubre con
    # los textos deterministas de las pruebas del parser.
    _put(image, "FECHA DE NACIMIENTO", 55, 500, scale=0.85, color=(90, 95, 110), thickness=1)
    _put(image, birth_text, 430, 502, scale=1.25, thickness=3)

    _put(image, "SEXO", 55, 615, scale=0.85, color=(90, 95, 110), thickness=1)
    _put(image, "M", 210, 615, scale=1.25, thickness=3)

    # Etiqueta real de caducidad: en los documentos colombianos "EXPEDICION"
    # es la fecha de expedicion, no la de vencimiento.
    _put(image, "FECHA DE VENCIMIENTO", 55, 700, scale=0.85, color=(90, 95, 110), thickness=1)
    _put(image, expiry_text, 480, 702, scale=1.25, thickness=3)

    if with_face:
        render_face(image, x=1060, y=210)

    if with_mrz:
        # Banda blanca de alto contraste, como en los documentos reales.
        # La escala y el grosor estan ajustados para que el OCR lea la MRZ de
        # forma fiable: con trazos mas gruesos los caracteres se pegan y el
        # reconocedor confunde digitos. La banda deja 35 px libres bajo la
        # ultima linea: si el renglon toca el borde, el recorte del contorno le
        # come la mitad inferior y el OCR solo lee una de las dos lineas.
        band_top = DOCUMENT_HEIGHT - 125
        cv2.rectangle(
            image, (14, band_top), (DOCUMENT_WIDTH - 15, DOCUMENT_HEIGHT - 14), (255, 255, 255), -1
        )
        _put(image, mrz_lines[0], 40, band_top + 42, scale=1.05, mono=True, thickness=1)
        _put(image, mrz_lines[1], 40, band_top + 80, scale=1.05, mono=True, thickness=1)

    if noise > 0:
        noise_layer = np.random.default_rng(seed=7).normal(0, noise, image.shape)
        image = np.clip(image.astype(np.float32) + noise_layer, 0, 255).astype(np.uint8)

    return image


def encode(image: np.ndarray, extension: str = ".jpg", quality: int = 95) -> bytes:
    """Codifica la imagen a bytes JPEG o PNG."""
    params: list[int] = []
    if extension.lower() in {".jpg", ".jpeg"}:
        params = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    success, buffer = cv2.imencode(extension, image, params)
    if not success:  # pragma: no cover - defensivo
        raise RuntimeError("No se pudo codificar la imagen de prueba.")
    return buffer.tobytes()


def document_bytes(**kwargs: object) -> bytes:
    """Atajo: documento sintetico codificado como JPEG."""
    return encode(render_synthetic_document(**kwargs))  # type: ignore[arg-type]


def skin_tone_face_image(width: int = 400, height: int = 400) -> np.ndarray:
    """Rectangulo con tono de piel: dispara el detector heuristico."""
    image = np.full((height, width, 3), 250, dtype=np.uint8)
    cv2.rectangle(image, (120, 90), (280, 310), (140, 175, 208), -1)
    return image


__all__ = [
    "ICAO_TD3_LINES",
    "TARJETA_IDENTIDAD_BACK_TEXT",
    "TARJETA_IDENTIDAD_FRONT_TEXT",
    "build_td3_mrz",
    "document_bytes",
    "encode",
    "render_face",
    "render_synthetic_document",
    "skin_tone_face_image",
]
