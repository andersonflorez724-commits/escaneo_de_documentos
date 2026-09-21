"""Deteccion del rostro del documento con OpenCV.

La fotografia del titular es la zona menos tolerante a errores: si esta
ausente, borrosa o recortada el documento no sirve para identificar al
cliente. Este modulo localiza el rostro con clasificadores en cascada de
Haar (incluidos en OpenCV, sin pesos externos) y mide la nitidez del recorte.

Cuando OpenCV no esta disponible se aplica un detector heuristico por tono
de piel sobre el espacio de color YCrCb para no devolver siempre ``False``.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from app.core.config import Settings, get_settings
from app.services.image_utils import to_grayscale

logger = logging.getLogger(__name__)

try:  # pragma: no cover - depende del entorno
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None  # type: ignore[assignment]

CV2_AVAILABLE = cv2 is not None

# OpenCV 5 elimino `CascadeClassifier` y dejo de empaquetar las cascadas de
# Haar. Se comprueba en tiempo de ejecucion para poder diagnosticarlo bien.
HAAR_AVAILABLE = CV2_AVAILABLE and hasattr(cv2, "CascadeClassifier")

# Nitidez minima (varianza del laplaciano) exigida al recorte del rostro.
FACE_SHARPNESS_THRESHOLD = 55.0

# Proporcion del area del documento que puede ocupar el rostro. En una cedula
# o pasaporte la fotografia ocupa entre el 8 % y el 30 % de la superficie, por
# eso el minimo se fija en 2 %: asi se descartan las pequenas regiones con
# textura (sellos, codigos de barras o texto) que disparan falsos positivos.
MIN_FACE_AREA_RATIO = 0.02
MAX_FACE_AREA_RATIO = 0.45

# Ningun rostro de documento puede medir menos de esto (lado menor, en px).
MIN_FACE_SIDE_PX = 48


@dataclass(frozen=True, slots=True)
class FaceDetection:
    """Resultado de la inspeccion del rostro del documento."""

    detected: bool
    count: int
    confidence: float
    method: str
    box: tuple[int, int, int, int] | None = None
    sharpness: float = 0.0
    area_ratio: float = 0.0
    valid_photo: bool = False
    warnings: tuple[str, ...] = field(default=())

    def as_dict(self) -> dict[str, Any]:
        return {
            "detected": self.detected,
            "count": self.count,
            "confidence": round(self.confidence, 4),
            "method": self.method,
            "box": list(self.box) if self.box else None,
            "sharpness": round(self.sharpness, 2),
            "area_ratio": round(self.area_ratio, 5),
            "valid_photo": self.valid_photo,
            "warnings": list(self.warnings),
        }


def _measure_sharpness(image: np.ndarray, box: tuple[int, int, int, int]) -> float:
    """Varianza del laplaciano del recorte (nitidez del rostro)."""
    x, y, width, height = box
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(image.shape[1], x + width), min(image.shape[0], y + height)
    crop = image[y0:y1, x0:x1]
    if crop.size == 0:
        return 0.0

    gray = to_grayscale(crop)
    if cv2 is not None:
        return float(cv2.Laplacian(gray, cv2.CV_64F).var())
    return float(np.var(np.diff(gray.astype(np.float32), axis=1)))  # pragma: no cover


def _evaluate(
    image: np.ndarray,
    box: tuple[int, int, int, int],
    *,
    confidence: float,
    method: str,
    count: int = 1,
) -> FaceDetection:
    """Construye el resultado y decide si la foto del documento es valida."""
    height, width = image.shape[:2]
    area_ratio = (box[2] * box[3]) / float(width * height)
    sharpness = _measure_sharpness(image, box)

    warnings: list[str] = []
    if sharpness < FACE_SHARPNESS_THRESHOLD:
        warnings.append("El rostro del documento aparece desenfocado.")
    if area_ratio < MIN_FACE_AREA_RATIO or min(box[2], box[3]) < MIN_FACE_SIDE_PX:
        warnings.append("El rostro detectado es demasiado pequeno.")
    if area_ratio > MAX_FACE_AREA_RATIO:
        warnings.append("El rostro detectado ocupa casi toda la imagen; revisa el encuadre.")

    valid = (
        MIN_FACE_AREA_RATIO <= area_ratio <= MAX_FACE_AREA_RATIO
        and min(box[2], box[3]) >= MIN_FACE_SIDE_PX
        and sharpness >= FACE_SHARPNESS_THRESHOLD
    )

    return FaceDetection(
        detected=True,
        count=count,
        confidence=confidence,
        method=method,
        box=box,
        sharpness=sharpness,
        area_ratio=area_ratio,
        valid_photo=valid,
        warnings=tuple(warnings),
    )


def _not_detected(method: str, warning: str) -> FaceDetection:
    return FaceDetection(
        detected=False,
        count=0,
        confidence=0.0,
        method=method,
        valid_photo=False,
        warnings=(warning,),
    )


# ---------------------------------------------------------------------------
# Contrato
# ---------------------------------------------------------------------------
class BaseFaceDetector(ABC):
    """Interfaz comun de los detectores de rostro."""

    name: str = "base"

    @abstractmethod
    def detect(self, image: np.ndarray) -> FaceDetection:
        """Localiza el rostro de la fotografia del documento."""


# ---------------------------------------------------------------------------
# OpenCV: clasificadores en cascada de Haar
# ---------------------------------------------------------------------------
class HaarFaceDetector(BaseFaceDetector):
    """Detector de rostro frontal con las cascadas incluidas en OpenCV."""

    name = "opencv-haar"

    CASCADES = (
        "haarcascade_frontalface_default.xml",
        "haarcascade_frontalface_alt2.xml",
    )

    def __init__(self, min_size: int = 40) -> None:
        self.min_size = min_size
        self._classifiers: list[Any] = []

    def _load(self) -> list[Any]:
        if self._classifiers:
            return self._classifiers

        if cv2 is None:  # pragma: no cover
            return []

        base = Path(cv2.data.haarcascades)  # type: ignore[attr-defined]
        for name in self.CASCADES:
            path = base / name
            if not path.is_file():
                logger.warning("No se encontro la cascada %s", path)
                continue
            classifier = cv2.CascadeClassifier(str(path))
            if not classifier.empty():
                self._classifiers.append(classifier)

        if not self._classifiers:  # pragma: no cover
            logger.warning("No se pudo cargar ninguna cascada de Haar.")

        return self._classifiers

    def detect(self, image: np.ndarray) -> FaceDetection:
        if not HAAR_AVAILABLE:  # pragma: no cover - segun la version de OpenCV
            return _not_detected(
                self.name,
                "Esta version de OpenCV no incluye las cascadas de Haar "
                "(OpenCV 5 las elimino). Instala opencv-python-headless<5.",
            )

        classifiers = self._load()
        if not classifiers:  # pragma: no cover
            return _not_detected(self.name, "No hay clasificadores de Haar disponibles.")

        gray = to_grayscale(image)
        gray = cv2.equalizeHist(gray)

        # Tamano minimo real: ademas del minimo configurado se exige un
        # porcentaje del lado de la imagen para evitar falsos positivos sobre
        # el texto del documento.
        min_side = max(self.min_size, int(min(image.shape[:2]) * 0.06))

        candidates: list[tuple[int, int, int, int]] = []
        for classifier in classifiers:
            found = classifier.detectMultiScale(
                gray,
                scaleFactor=1.08,
                minNeighbors=6,
                minSize=(min_side, min_side),
                flags=cv2.CASCADE_SCALE_IMAGE,
            )
            candidates.extend(tuple(int(value) for value in rect) for rect in found)

        if not candidates:
            # Segunda pasada mas permisiva: documentos con foto pequena o con
            # iluminacion desigual suelen necesitarla.
            for classifier in classifiers:
                relaxed_side = max(40, int(min_side * 0.7))
                found = classifier.detectMultiScale(
                    gray,
                    scaleFactor=1.05,
                    minNeighbors=4,
                    minSize=(relaxed_side, relaxed_side),
                )
                candidates.extend(tuple(int(value) for value in rect) for rect in found)

        if not candidates:
            return _not_detected(
                self.name,
                "No se detecto un rostro en el documento (la foto puede faltar o estar tapada).",
            )

        # Se prefiere el rostro mas nitido, no solamente el mas grande.
        best_box: tuple[int, int, int, int] | None = None
        best_score = -1.0
        for box in _deduplicate(candidates):
            sharpness = _measure_sharpness(image, box)
            area_ratio = (box[2] * box[3]) / float(image.shape[0] * image.shape[1])
            if not (MIN_FACE_AREA_RATIO * 0.5 <= area_ratio <= MAX_FACE_AREA_RATIO):
                continue
            score = sharpness * (1.0 + min(area_ratio, 0.2) * 4.0)
            if score > best_score:
                best_score = score
                best_box = box

        if best_box is None:
            return _not_detected(self.name, "Las regiones detectadas no tienen tamano de rostro valido.")

        return _evaluate(
            image,
            best_box,
            confidence=min(1.0, 0.45 + best_score / 900.0),
            method=self.name,
            count=len(candidates),
        )


def _deduplicate(boxes: list[tuple[int, int, int, int]], iou_threshold: float = 0.35) -> list[tuple[int, int, int, int]]:
    """Fusiona cuadros muy solapados manteniendo el de mayor area."""
    def iou(first: tuple[int, int, int, int], second: tuple[int, int, int, int]) -> float:
        ax, ay, aw, ah = first
        bx, by, bw, bh = second
        x_left, y_top = max(ax, bx), max(ay, by)
        x_right, y_bottom = min(ax + aw, bx + bw), min(ay + ah, by + bh)
        if x_right <= x_left or y_bottom <= y_top:
            return 0.0
        intersection = (x_right - x_left) * (y_bottom - y_top)
        union = aw * ah + bw * bh - intersection
        return intersection / union if union else 0.0

    kept: list[tuple[int, int, int, int]] = []
    for box in sorted(boxes, key=lambda item: item[2] * item[3], reverse=True):
        if all(iou(box, existing) < iou_threshold for existing in kept):
            kept.append(box)
    return kept


# ---------------------------------------------------------------------------
# Sin OpenCV: heuristica por tono de piel
# ---------------------------------------------------------------------------
class HeuristicFaceDetector(BaseFaceDetector):
    """Detector aproximado por segmentacion de tono de piel (YCrCb)."""

    name = "heuristic-skin"

    def detect(self, image: np.ndarray) -> FaceDetection:
        if cv2 is None:  # pragma: no cover
            return _not_detected(self.name, "OpenCV no esta disponible para inspeccionar el rostro.")

        ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
        mask = cv2.inRange(ycrcb, np.array([0, 133, 77], dtype=np.uint8), np.array([255, 173, 127], dtype=np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8), iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8), iterations=2)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return _not_detected(self.name, "No se detectaron zonas con tono de piel.")

        height, width = image.shape[:2]
        best: tuple[int, int, int, int] | None = None
        best_area = 0.0
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            area = float(w * h)
            if h == 0 or w == 0:
                continue
            aspect = w / float(h)
            if not (0.55 <= aspect <= 1.5):
                continue
            area_ratio = area / float(width * height)
            if MIN_FACE_AREA_RATIO <= area_ratio <= MAX_FACE_AREA_RATIO and area > best_area:
                best, best_area = (x, y, w, h), area

        if best is None:
            return _not_detected(self.name, "No se encontro una region con proporciones de rostro.")

        return _evaluate(
            image,
            best,
            confidence=0.45,
            method=self.name,
            count=len(contours),
        )


# ---------------------------------------------------------------------------
# Fabrica
# ---------------------------------------------------------------------------
def _build_detector(settings: Settings) -> BaseFaceDetector:
    choice = (settings.face_detector or "auto").lower()

    if choice == "heuristic":
        return HeuristicFaceDetector()
    if choice == "opencv":
        if HAAR_AVAILABLE:
            return HaarFaceDetector()
        logger.warning(
            "FACE_DETECTOR=opencv pero las cascadas de Haar no estan disponibles "
            "(OpenCV %s); se usa la heuristica por tono de piel.",
            getattr(cv2, "__version__", "ausente"),
        )
        return HeuristicFaceDetector()

    if HAAR_AVAILABLE:
        return HaarFaceDetector()

    logger.warning(
        "OpenCV %s no incluye las cascadas de Haar; se usa el detector heuristico. "
        "Instala opencv-python-headless<5 para la deteccion con clasificadores.",
        getattr(cv2, "__version__", "ausente"),
    )
    return HeuristicFaceDetector()


@lru_cache(maxsize=1)
def get_face_detector() -> BaseFaceDetector:
    """Devuelve (y cachea) el detector de rostro configurado."""
    return _build_detector(get_settings())


def reset_face_detector() -> None:
    """Limpia la cache del detector (util en pruebas)."""
    get_face_detector.cache_clear()


__all__ = [
    "BaseFaceDetector",
    "CV2_AVAILABLE",
    "HAAR_AVAILABLE",
    "FaceDetection",
    "HaarFaceDetector",
    "HeuristicFaceDetector",
    "get_face_detector",
    "reset_face_detector",
]
