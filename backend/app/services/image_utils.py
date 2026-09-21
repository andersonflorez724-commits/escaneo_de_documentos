"""Utilidades de imagen (OpenCV + Pillow) usadas por el pipeline de ML.

Centraliza la decodificacion, el realce y el analisis de calidad de la foto
del documento. Cuando OpenCV no esta disponible las funciones degradan a
implementaciones con Pillow/numpy en lugar de fallar.
"""

from __future__ import annotations

import io
import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

from app.core.errors import AppError

logger = logging.getLogger(__name__)

try:  # pragma: no cover - depende del entorno
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None  # type: ignore[assignment]

CV2_AVAILABLE = cv2 is not None

ALLOWED_MIME_TYPES: frozenset[str] = frozenset(
    {"image/jpeg", "image/jpg", "image/png", "image/webp", "image/bmp", "image/tiff"}
)

# Umbrales de calidad (calibrados sobre fotos de documentos con movil)
BLUR_THRESHOLD = 90.0
DARK_THRESHOLD = 55.0
BRIGHT_THRESHOLD = 215.0
MIN_MEGAPIXELS = 0.12


class InvalidImageError(AppError):
    """La imagen no se pudo decodificar o no tiene un formato soportado.

    Hereda de :class:`AppError`, de modo que el manejador global la traduce a
    un **422** con el formato de error uniforme de la API.
    """

    status_code = 422
    code = "invalid_image"
    default_message = "No se pudo leer la imagen. Envia una foto en formato JPEG o PNG."


# ---------------------------------------------------------------------------
# Modelo de calidad
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class ImageQuality:
    """Metricas objetivas de la fotografia recibida."""

    width: int
    height: int
    megapixels: float
    blur_score: float
    brightness: float
    contrast: float
    is_blurry: bool
    is_too_dark: bool
    is_overexposed: bool
    is_low_resolution: bool
    score: float
    warnings: tuple[str, ...] = field(default=())

    def as_dict(self) -> dict[str, Any]:
        return {
            "width": self.width,
            "height": self.height,
            "megapixels": round(self.megapixels, 2),
            "blur_score": round(self.blur_score, 2),
            "brightness": round(self.brightness, 2),
            "contrast": round(self.contrast, 2),
            "is_blurry": self.is_blurry,
            "is_too_dark": self.is_too_dark,
            "is_overexposed": self.is_overexposed,
            "is_low_resolution": self.is_low_resolution,
            "score": round(self.score, 3),
            "warnings": list(self.warnings),
        }


# ---------------------------------------------------------------------------
# Decodificacion
# ---------------------------------------------------------------------------
def decode_image(data: bytes) -> np.ndarray:
    """Convierte los bytes de una imagen en un array BGR de OpenCV.

    Raises:
        InvalidImageError: si los bytes no corresponden a una imagen valida.
    """
    if not data:
        raise InvalidImageError("El archivo de imagen esta vacio.")

    try:
        with Image.open(io.BytesIO(data)) as image:
            # Corrige la orientacion declarada en el EXIF (fotos de movil).
            image = ImageOps.exif_transpose(image)
            rgb = image.convert("RGB")
            array = np.asarray(rgb, dtype=np.uint8)
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise InvalidImageError(
            "No se pudo leer la imagen. Envia una foto en formato JPEG o PNG."
        ) from exc

    if array.ndim != 3 or array.shape[2] != 3:
        raise InvalidImageError("La imagen no tiene el formato de color esperado.")

    # Pillow entrega RGB; OpenCV trabaja en BGR.
    return np.ascontiguousarray(array[:, :, ::-1])


def encode_jpeg(image: np.ndarray, quality: int = 82) -> bytes:
    """Codifica un array BGR a JPEG (para la vista previa del recorte)."""
    rgb = np.ascontiguousarray(image[:, :, ::-1])
    buffer = io.BytesIO()
    Image.fromarray(rgb).save(buffer, format="JPEG", quality=quality, optimize=True)
    return buffer.getvalue()


def to_grayscale(image: np.ndarray) -> np.ndarray:
    """Version en escala de grises de una imagen BGR."""
    if image.ndim == 2:
        return image
    if cv2 is not None:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    weights = np.array([0.114, 0.587, 0.299], dtype=np.float32)
    return (image.astype(np.float32) @ weights).astype(np.uint8)


# ---------------------------------------------------------------------------
# Escalado y realce
# ---------------------------------------------------------------------------
def resize_max(image: np.ndarray, max_dimension: int) -> tuple[np.ndarray, float]:
    """Reduce la imagen para que su lado mayor no supere ``max_dimension``.

    Returns:
        ``(imagen, factor_de_escala)`` donde el factor es ``<= 1.0``.
    """
    if max_dimension <= 0:
        return image, 1.0

    height, width = image.shape[:2]
    longest = max(height, width)
    if longest <= max_dimension:
        return image, 1.0

    scale = max_dimension / float(longest)
    new_size = (max(1, int(round(width * scale))), max(1, int(round(height * scale))))
    if cv2 is not None:
        resized = cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)
    else:  # pragma: no cover
        resized = np.asarray(Image.fromarray(image[:, :, ::-1]).resize(new_size)).astype(np.uint8)[:, :, ::-1]
    return np.ascontiguousarray(resized), scale


def upscale_small(image: np.ndarray, min_width: int = 1000) -> np.ndarray:
    """Amplia imagenes pequenas: el OCR pierde precision en fotos diminutas."""
    height, width = image.shape[:2]
    if width >= min_width:
        return image
    scale = min_width / float(width)
    new_size = (int(width * scale), int(height * scale))
    if cv2 is not None:
        return cv2.resize(image, new_size, interpolation=cv2.INTER_CUBIC)
    return np.asarray(  # pragma: no cover
        Image.fromarray(image[:, :, ::-1]).resize(new_size, Image.BICUBIC)
    ).astype(np.uint8)[:, :, ::-1]


def enhance_for_ocr(image: np.ndarray) -> np.ndarray:
    """Realce previo al OCR: CLAHE + reduccion de ruido + enfoque.

    Mejora el contraste local de documentos con reflejos o iluminacion desigual,
    que es el escenario tipico en una recepcion.
    """
    gray = to_grayscale(image)
    if cv2 is None:  # pragma: no cover
        return gray

    clahe = cv2.createCLAHE(clipLimit=2.6, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    enhanced = cv2.bilateralFilter(enhanced, 7, 55, 55)

    # Enfoque con unsharp mask.
    blurred = cv2.GaussianBlur(enhanced, (0, 0), sigmaX=1.6)
    sharpened = cv2.addWeighted(enhanced, 1.5, blurred, -0.5, 0)
    return sharpened


def adaptive_binarize(image: np.ndarray) -> np.ndarray:
    """Binarizacion adaptativa (util para documentos con sombras)."""
    gray = to_grayscale(image)
    if cv2 is None:  # pragma: no cover
        return gray
    return cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 25, 12
    )


# ---------------------------------------------------------------------------
# Enderezado y recorte del documento
# ---------------------------------------------------------------------------
def estimate_skew_angle(image: np.ndarray) -> float:
    """Estima el angulo de inclinacion (grados) del contenido del documento."""
    if cv2 is None:  # pragma: no cover
        return 0.0

    gray = to_grayscale(image)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(
        edges, 1, np.pi / 180.0, threshold=120, minLineLength=max(60, image.shape[1] // 4), maxLineGap=12
    )
    if lines is None:
        return 0.0

    # OpenCV 5 devuelve (N, 4); las versiones anteriores (N, 1, 4).
    segments = np.asarray(lines).reshape(-1, 4)

    angles: list[float] = []
    for x1, y1, x2, y2 in segments:
        angle = float(np.degrees(np.arctan2(float(y2 - y1), float(x2 - x1))))
        # Solo interesan las lineas casi horizontales (renglones de texto).
        if -20.0 <= angle <= 20.0:
            angles.append(angle)

    if not angles:
        return 0.0

    angle = float(np.median(angles))
    return 0.0 if abs(angle) < 0.6 else angle


def deskew(image: np.ndarray) -> tuple[np.ndarray, float]:
    """Rota la imagen para compensar la inclinacion detectada."""
    angle = estimate_skew_angle(image)
    if abs(angle) < 0.6 or cv2 is None:
        return image, 0.0

    height, width = image.shape[:2]
    matrix = cv2.getRotationMatrix2D((width / 2.0, height / 2.0), angle, 1.0)
    rotated = cv2.warpAffine(
        image,
        matrix,
        (width, height),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )
    return rotated, angle


def find_document_quad(image: np.ndarray) -> np.ndarray | None:
    """Localiza el cuadrilatero del documento dentro de la foto.

    Returns:
        Array ``(4, 2)`` con las esquinas ordenadas, o ``None`` si no se
        detecta con suficiente confianza.
    """
    if cv2 is None:  # pragma: no cover
        return None

    gray = to_grayscale(image)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(gray, 60, 180)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)

    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    min_area = 0.25 * image.shape[0] * image.shape[1]

    best: np.ndarray | None = None
    best_area = 0.0
    for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:8]:
        area = cv2.contourArea(contour)
        if area < min_area or area <= best_area:
            continue
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
        if len(approx) == 4 and cv2.isContourConvex(approx):
            best = approx.reshape(4, 2).astype(np.float32)
            best_area = area

    return best


def warp_document(image: np.ndarray, quad: np.ndarray, margin_ratio: float = 0.02) -> np.ndarray:
    """Aplica la transformacion de perspectiva para enderezar el documento."""
    if cv2 is None:  # pragma: no cover
        return image

    points = _order_quad_corners(quad)
    top_left, top_right, bottom_right, bottom_left = points

    width = int(max(np.linalg.norm(bottom_right - bottom_left), np.linalg.norm(top_right - top_left)))
    height = int(max(np.linalg.norm(top_right - bottom_right), np.linalg.norm(top_left - bottom_left)))
    if width < 32 or height < 32:
        return image

    destination = np.array(
        [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype=np.float32
    )
    matrix = cv2.getPerspectiveTransform(points.astype(np.float32), destination)
    warped = cv2.warpPerspective(image, matrix, (width, height))

    # Recorta un margen interno para eliminar el fondo que rodea al documento.
    margin_x = int(width * margin_ratio)
    margin_y = int(height * margin_ratio)
    inner = warped[margin_y : height - margin_y, margin_x : width - margin_x]
    return inner if inner.size else warped


def _order_quad_corners(quad: np.ndarray) -> np.ndarray:
    """Ordena 4 puntos como (sup-izq, sup-der, inf-der, inf-izq)."""
    points = quad.reshape(4, 2).astype(np.float32)
    ordered = np.zeros((4, 2), dtype=np.float32)

    sums = points.sum(axis=1)
    ordered[0] = points[np.argmin(sums)]
    ordered[2] = points[np.argmax(sums)]

    diffs = np.diff(points, axis=1).reshape(-1)
    ordered[1] = points[np.argmin(diffs)]
    ordered[3] = points[np.argmax(diffs)]
    return ordered


# ---------------------------------------------------------------------------
# Calidad
# ---------------------------------------------------------------------------
def analyze_quality(image: np.ndarray) -> ImageQuality:
    """Calcula metricas de nitidez, exposicion y resolucion de la foto."""
    gray = to_grayscale(image)
    height, width = gray.shape[:2]

    if cv2 is not None:
        blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    else:  # pragma: no cover
        blur_score = float(np.var(np.diff(gray.astype(np.float32), axis=1)))

    brightness = float(np.mean(gray))
    contrast = float(np.std(gray))
    megapixels = (width * height) / 1_000_000.0

    is_blurry = blur_score < BLUR_THRESHOLD
    is_too_dark = brightness < DARK_THRESHOLD
    is_overexposed = brightness > BRIGHT_THRESHOLD
    is_low_resolution = megapixels < MIN_MEGAPIXELS

    warnings: list[str] = []
    if is_blurry:
        warnings.append("La imagen parece movida o desenfocada.")
    if is_too_dark:
        warnings.append("La imagen esta demasiado oscura.")
    if is_overexposed:
        warnings.append("La imagen tiene demasiada luz (posible reflejo).")
    if is_low_resolution:
        warnings.append("La resolucion es baja para un OCR fiable.")

    # Puntaje compuesto 0..1 usado como confianza de calidad.
    sharpness_term = min(1.0, blur_score / (BLUR_THRESHOLD * 2.5))
    exposure_term = 1.0 - min(1.0, abs(brightness - 150.0) / 150.0)
    contrast_term = min(1.0, contrast / 70.0)
    resolution_term = min(1.0, megapixels / 1.5)
    score = (
        0.40 * sharpness_term
        + 0.25 * exposure_term
        + 0.20 * contrast_term
        + 0.15 * resolution_term
    )

    return ImageQuality(
        width=width,
        height=height,
        megapixels=megapixels,
        blur_score=blur_score,
        brightness=brightness,
        contrast=contrast,
        is_blurry=is_blurry,
        is_too_dark=is_too_dark,
        is_overexposed=is_overexposed,
        is_low_resolution=is_low_resolution,
        score=max(0.0, min(1.0, score)),
        warnings=tuple(warnings),
    )


# ---------------------------------------------------------------------------
# Recorte de la zona de la MRZ
# ---------------------------------------------------------------------------
def crop_mrz_band(image: np.ndarray, ratio: float = 0.22) -> np.ndarray:
    """Recorta la franja inferior donde se ubica la MRZ (ICAO 9293)."""
    height = image.shape[0]
    top = int(height * (1.0 - ratio))
    band = image[top:height, :]
    return band if band.size else image


__all__ = [
    "ALLOWED_MIME_TYPES",
    "CV2_AVAILABLE",
    "ImageQuality",
    "InvalidImageError",
    "adaptive_binarize",
    "analyze_quality",
    "crop_mrz_band",
    "decode_image",
    "deskew",
    "encode_jpeg",
    "enhance_for_ocr",
    "estimate_skew_angle",
    "find_document_quad",
    "resize_max",
    "to_grayscale",
    "upscale_small",
    "warp_document",
]
