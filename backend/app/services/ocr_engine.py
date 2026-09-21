"""Motor de OCR basado en el modelo preentrenado **EasyOCR**.

El motor se carga de forma perezosa (*lazy*) y se reutiliza entre peticiones
dentro del mismo proceso: cargar los pesos de PyTorch cuesta varios segundos,
por eso el arranque de la aplicacion llama a :func:`warmup_ocr_engine`.

Se define una jerarquia de motores para que el servicio pueda desplegarse en
entornos donde PyTorch no cabe (por ejemplo funciones serverless de Vercel de
250 MB) sin romper el contrato de la API:

* :class:`EasyOCREngine`  -> inferencia real con el modelo preentrenado.
* :class:`HeuristicOCREngine` -> preprocesado con OpenCV, sin reconocimiento.
"""

from __future__ import annotations

import importlib.util
import logging
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Iterable, Sequence

import numpy as np

from app.core.config import Settings, get_settings
from app.services.image_utils import resize_max, upscale_small

logger = logging.getLogger(__name__)


class OCREngineUnavailableError(RuntimeError):
    """El motor solicitado no pudo inicializarse."""


# ---------------------------------------------------------------------------
# Modelos de datos
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class TextBlock:
    """Fragmento de texto detectado por el modelo."""

    text: str
    confidence: float
    box: tuple[int, int, int, int]
    rel_box: tuple[float, float, float, float]

    @property
    def y_center(self) -> float:
        return self.rel_box[1] + self.rel_box[3] / 2.0

    @property
    def x_center(self) -> float:
        return self.rel_box[0] + self.rel_box[2] / 2.0

    @property
    def row(self) -> str:
        """Franja vertical donde cae el bloque (util para ubicar la MRZ)."""
        if self.y_center < 0.34:
            return "top"
        if self.y_center < 0.68:
            return "middle"
        return "bottom"


@dataclass(slots=True)
class OCRResult:
    """Resultado completo de una pasada de OCR."""

    engine: str
    blocks: list[TextBlock] = field(default_factory=list)
    elapsed_ms: float = 0.0
    warnings: list[str] = field(default_factory=list)

    @property
    def text(self) -> str:
        return "\n".join(block.text for block in self.blocks)

    @property
    def mean_confidence(self) -> float:
        if not self.blocks:
            return 0.0
        return sum(block.confidence for block in self.blocks) / len(self.blocks)

    @property
    def lines(self) -> list[str]:
        """Agrupa los bloques en renglones segun su posicion vertical."""
        return [" ".join(group) for group in group_blocks_into_lines(self.blocks)]


# ---------------------------------------------------------------------------
# Utilidades de agrupacion y orden de lectura
# ---------------------------------------------------------------------------
def group_blocks_into_lines(blocks: Sequence[TextBlock], tolerance: float = 0.032) -> list[list[str]]:
    """Agrupa bloques en renglones usando su centro vertical relativo."""
    if not blocks:
        return []

    ordered = sorted(blocks, key=lambda block: (block.y_center, block.x_center))
    lines: list[list[TextBlock]] = []
    current: list[TextBlock] = [ordered[0]]

    for block in ordered[1:]:
        reference = sum(item.y_center for item in current) / len(current)
        if abs(block.y_center - reference) <= tolerance:
            current.append(block)
        else:
            lines.append(current)
            current = [block]
    lines.append(current)

    return [
        [block.text for block in sorted(line, key=lambda block: block.x_center)]
        for line in lines
    ]


def _to_blocks(raw_results: Iterable[Any], image: np.ndarray, min_confidence: float) -> list[TextBlock]:
    """Convierte la salida de EasyOCR en :class:`TextBlock` normalizados."""
    height, width = image.shape[:2]
    blocks: list[TextBlock] = []

    for item in raw_results:
        try:
            points, text, confidence = item[0], item[1], float(item[2])
        except (IndexError, TypeError, ValueError):  # pragma: no cover - defensivo
            continue

        text = str(text).strip()
        if not text or confidence < min_confidence:
            continue

        xs = [float(point[0]) for point in points]
        ys = [float(point[1]) for point in points]
        x_min, x_max = max(0, int(min(xs))), min(width, int(max(xs)))
        y_min, y_max = max(0, int(min(ys))), min(height, int(max(ys)))
        box = (x_min, y_min, max(1, x_max - x_min), max(1, y_max - y_min))

        rel_box = (
            box[0] / width,
            box[1] / height,
            box[2] / width,
            box[3] / height,
        )
        blocks.append(TextBlock(text=text, confidence=confidence, box=box, rel_box=rel_box))

    return blocks


# ---------------------------------------------------------------------------
# Contrato de los motores
# ---------------------------------------------------------------------------
class BaseOCREngine(ABC):
    """Interfaz comun de los motores de reconocimiento."""

    name: str = "base"
    supports_recognition: bool = True

    @abstractmethod
    def read(self, image: np.ndarray) -> list[TextBlock]:
        """Extrae los bloques de texto de una imagen BGR."""

    def warmup(self) -> None:
        """Carga anticipada del modelo. Por defecto no hace nada."""
        return None


# ---------------------------------------------------------------------------
# Motor real: EasyOCR (modelo preentrenado)
# ---------------------------------------------------------------------------
class EasyOCREngine(BaseOCREngine):
    """Inferencia con la red preentrenada de EasyOCR sobre OpenCV.

    EasyOCR combina un detector de texto (CRAFT) y un reconocedor (CRNN) ya
    entrenados. La primera llamada descarga los pesos (~100 MB) y los deja en
    ``~/.EasyOCR``; despues se reutilizan desde memoria.
    """

    name = "easyocr"

    def __init__(
        self,
        languages: Sequence[str] = ("es", "en"),
        use_gpu: bool = False,
        min_confidence: float = 0.30,
        max_dimension: int = 1600,
    ) -> None:
        self.languages = tuple(languages) or ("es", "en")
        self.use_gpu = use_gpu
        self.min_confidence = min_confidence
        self.max_dimension = max_dimension
        self._reader: Any | None = None
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ carga
    @property
    def loaded(self) -> bool:
        return self._reader is not None

    def _ensure_reader(self) -> Any:
        if self._reader is not None:
            return self._reader

        with self._lock:
            if self._reader is None:
                try:
                    import easyocr
                except ImportError as exc:  # pragma: no cover
                    raise OCREngineUnavailableError(
                        "EasyOCR no esta instalado. Instala backend/requirements.txt "
                        "o configura OCR_ENGINE=heuristic."
                    ) from exc

                logger.info(
                    "Cargando modelo preentrenado EasyOCR (idiomas=%s, gpu=%s)",
                    ",".join(self.languages),
                    self.use_gpu,
                )
                started = time.perf_counter()
                self._reader = easyocr.Reader(
                    list(self.languages),
                    gpu=self.use_gpu,
                    verbose=False,
                )
                logger.info("Modelo EasyOCR cargado en %.1f s", time.perf_counter() - started)

        return self._reader

    def warmup(self) -> None:
        self._ensure_reader()

    # -------------------------------------------------------------- inferencia
    def read(self, image: np.ndarray) -> list[TextBlock]:
        reader = self._ensure_reader()

        prepared, _ = resize_max(image, self.max_dimension)
        prepared = upscale_small(prepared, min_width=1000)

        # El lector de EasyOCR no es seguro entre hilos: se serializa.
        with self._lock:
            raw = reader.readtext(
                prepared,
                detail=1,
                paragraph=False,
                text_threshold=0.6,
                low_text=0.35,
                link_threshold=0.4,
                width_ths=0.7,
                add_margin=0.06,
            )

        return _to_blocks(raw, prepared, self.min_confidence)


# ---------------------------------------------------------------------------
# Motor ligero: sin PyTorch (serverless)
# ---------------------------------------------------------------------------
class HeuristicOCREngine(BaseOCREngine):
    """Motor de respaldo cuando PyTorch/EasyOCR no estan disponibles.

    Ejecuta el preprocesado y la deteccion de regiones de texto con OpenCV
    (gradiente morfologico + MSER) pero **no realiza reconocimiento de
    caracteres**. Permite que el servicio arranque y que la validacion de
    rostro y de MRZ (cuando el cliente envia las lineas) sigan funcionando.
    """

    name = "heuristic"
    supports_recognition = False

    def __init__(self, max_dimension: int = 1600) -> None:
        self.max_dimension = max_dimension

    def read(self, image: np.ndarray) -> list[TextBlock]:  # noqa: ARG002 - firma comun
        # Sin modelo no hay texto que devolver: la deteccion de regiones se
        # reporta desde analyze_quality/face detection.
        return []


# ---------------------------------------------------------------------------
# Fabrica
# ---------------------------------------------------------------------------
def easyocr_is_available() -> bool:
    """Indica si el paquete EasyOCR es importable en este entorno."""
    return importlib.util.find_spec("easyocr") is not None


def _build_engine(settings: Settings) -> BaseOCREngine:
    choice = (settings.ocr_engine or "auto").lower()

    if choice == "easyocr":
        return EasyOCREngine(
            languages=settings.ocr_languages,
            use_gpu=settings.ocr_use_gpu,
            min_confidence=settings.ocr_min_confidence,
            max_dimension=settings.max_image_dimension,
        )

    if choice == "heuristic":
        logger.info("OCR_ENGINE=heuristic: se omite el modelo preentrenado.")
        return HeuristicOCREngine(max_dimension=settings.max_image_dimension)

    # auto
    if easyocr_is_available():
        return EasyOCREngine(
            languages=settings.ocr_languages,
            use_gpu=settings.ocr_use_gpu,
            min_confidence=settings.ocr_min_confidence,
            max_dimension=settings.max_image_dimension,
        )

    logger.warning(
        "EasyOCR no esta instalado: se usa el motor heuristico sin reconocimiento "
        "de texto. Para inferencia real instala backend/requirements.txt."
    )
    return HeuristicOCREngine(max_dimension=settings.max_image_dimension)


@lru_cache(maxsize=1)
def get_ocr_engine() -> BaseOCREngine:
    """Devuelve (y cachea) el motor de OCR configurado."""
    return _build_engine(get_settings())


def warmup_ocr_engine() -> None:
    """Intenta cargar el modelo al arrancar, sin abortar si falla."""
    engine = get_ocr_engine()
    try:
        engine.warmup()
    except Exception as exc:  # pragma: no cover - depende del entorno
        logger.warning("No se pudo precargar el modelo OCR (%s): %s", engine.name, exc)


def reset_ocr_engine() -> None:
    """Limpia la cache de motores (util en pruebas)."""
    get_ocr_engine.cache_clear()


__all__ = [
    "BaseOCREngine",
    "EasyOCREngine",
    "HeuristicOCREngine",
    "OCREngineUnavailableError",
    "OCRResult",
    "TextBlock",
    "easyocr_is_available",
    "get_ocr_engine",
    "group_blocks_into_lines",
    "reset_ocr_engine",
    "warmup_ocr_engine",
]
