"""Orquestador del pipeline de lectura de documentos.

Encadena las etapas del proceso:

```
bytes ─► decodificacion ─► escalado ─► calidad
      ─► recorte del documento ─► enderezado
      ─► deteccion de rostro ─► OCR (pagina completa + franja MRZ)
      ─► validacion MRZ ─► extraccion de campos ─► respuesta
```

El servicio es agnostico al motor de OCR y al detector de rostro: recibe las
dependencias por constructor, lo que permite probarlo con dobles de prueba
sin cargar PyTorch.
"""

from __future__ import annotations

import base64
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from app.core.config import Settings, get_settings
from app.services.document_parser import DocumentFields, parse_document
from app.services.face_detector import BaseFaceDetector, FaceDetection, get_face_detector
from app.services.image_utils import (
    ImageQuality,
    InvalidImageError,
    adaptive_binarize,
    analyze_quality,
    crop_mrz_band,
    decode_image,
    deskew,
    encode_jpeg,
    enhance_for_ocr,
    find_document_quad,
    resize_max,
    upscale_small,
    warp_document,
)
from app.services.mrz import MRZResult, extract_and_validate
from app.services.ocr_engine import BaseOCREngine, OCRResult, TextBlock, get_ocr_engine

logger = logging.getLogger(__name__)

# Peso relativo de cada senal en la confianza final.
CONFIDENCE_WEIGHTS = {
    "fields": 0.35,
    "ocr": 0.20,
    "mrz": 0.25,
    "quality": 0.20,
}

PREVIEW_MAX_DIMENSION = 900
PREVIEW_JPEG_QUALITY = 70


class DocumentScanError(RuntimeError):
    """Fallo controlado durante el procesamiento del documento."""


@dataclass(slots=True)
class ScanResult:
    """Resultado completo del escaneo de un documento."""

    fields: DocumentFields
    mrz: MRZResult
    face: FaceDetection
    quality: ImageQuality
    engine: str
    confidence: float = 0.0
    processing_ms: float = 0.0
    document_detected: bool = False
    deskew_angle: float = 0.0
    text_lines: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    preview_jpeg: bytes | None = None

    @property
    def document_number(self) -> str | None:
        return self.fields.document_number

    @property
    def name(self) -> str | None:
        return self.fields.name

    @property
    def valid_photo(self) -> bool:
        return self.face.valid_photo

    @property
    def preview_base64(self) -> str | None:
        if not self.preview_jpeg:
            return None
        return base64.b64encode(self.preview_jpeg).decode("ascii")


def mask_document_number(value: str | None, visible: int = 4) -> str | None:
    """Enmascara el numero de documento para los registros (privacidad)."""
    if not value:
        return None
    digits = re.sub(r"\D", "", value)
    if len(digits) <= visible:
        return "*" * len(digits)
    return "*" * (len(digits) - visible) + digits[-visible:]


def _dedupe_blocks(blocks: list[TextBlock]) -> list[TextBlock]:
    """Elimina lecturas repetidas conservando la de mayor confianza."""
    best: dict[str, TextBlock] = {}
    for block in blocks:
        key = re.sub(r"\s+", "", block.text).upper()
        if not key:
            continue
        current = best.get(key)
        if current is None or block.confidence > current.confidence:
            best[key] = block
    return sorted(best.values(), key=lambda item: (item.y_center, item.x_center))


def _combined_confidence(
    fields: DocumentFields,
    ocr: OCRResult,
    mrz: MRZResult,
    quality: ImageQuality,
) -> float:
    """Combina las senales del pipeline en una confianza global 0..1."""
    components: dict[str, float] = {
        "fields": fields.confidence,
        "ocr": ocr.mean_confidence,
        "quality": quality.score,
    }

    if mrz.detected:
        # Si la MRZ existe, su consistencia pesa tanto como los campos.
        components["mrz"] = mrz.score
    else:
        # Sin MRZ se reparte el peso entre las senales disponibles.
        remaining = CONFIDENCE_WEIGHTS["mrz"]
        components["fields"] = min(1.0, fields.confidence + remaining * 0.6)
        components["quality"] = min(1.0, quality.score + remaining * 0.4)

    total = 0.0
    for name, value in components.items():
        total += CONFIDENCE_WEIGHTS.get(name, 0.0) * max(0.0, min(1.0, value))

    return max(0.0, min(1.0, total))


class DocumentScanner:
    """Servicio de escaneo de documentos de identificacion."""

    def __init__(
        self,
        engine: BaseOCREngine | None = None,
        face_detector: BaseFaceDetector | None = None,
        settings: Settings | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self._engine = engine
        self._face_detector = face_detector

    # ------------------------------------------------------------ dependencias
    @property
    def engine(self) -> BaseOCREngine:
        if self._engine is None:
            self._engine = get_ocr_engine()
        return self._engine

    @property
    def face_detector(self) -> BaseFaceDetector:
        if self._face_detector is None:
            self._face_detector = get_face_detector()
        return self._face_detector

    # ------------------------------------------------------------- pipeline
    def scan(self, data: bytes, *, include_preview: bool = False) -> ScanResult:
        """Procesa la fotografia del documento y extrae sus campos.

        Args:
            data: Bytes de la imagen (JPEG/PNG/...).
            include_preview: Si es ``True`` agrega un JPEG reducido del
                documento ya recortado y enderezado.

        Raises:
            InvalidImageError: si la imagen no se puede decodificar.
            DocumentScanError: si el motor de OCR falla inesperadamente.
        """
        if len(data) > self.settings.max_upload_bytes:
            raise InvalidImageError(
                f"La imagen supera el limite de {self.settings.max_upload_mb} MB."
            )

        started = time.perf_counter()

        # 1. Decodificacion y normalizacion de tamano.
        image = decode_image(data)
        image, _ = resize_max(image, self.settings.max_image_dimension)

        warnings: list[str] = []

        # 2. Calidad de la foto original (antes de recortar).
        quality = analyze_quality(image)
        warnings.extend(quality.warnings)

        # 3. Localizacion y rectificacion del documento.
        quad = find_document_quad(image)
        document_detected = quad is not None
        if document_detected:
            image = warp_document(image, quad)
        else:
            warnings.append(
                "No se detecto el contorno del documento; se proceso la imagen completa."
            )

        # 4. Enderezado fino.
        image, deskew_angle = deskew(image)

        # 5. Rostro del documento.
        face = self.face_detector.detect(image)
        warnings.extend(face.warnings)

        # 6. OCR de la pagina completa.
        enhanced = enhance_for_ocr(image)
        blocks = self.engine.read(enhanced)

        # 7. Pasadas dedicadas a la MRZ (franja inferior ampliada).
        blocks.extend(self._read_mrz_passes(image))

        blocks = _dedupe_blocks(blocks)
        elapsed_ms = (time.perf_counter() - started) * 1000.0

        ocr = OCRResult(engine=self.engine.name, blocks=blocks, elapsed_ms=elapsed_ms)

        if not self.engine.supports_recognition:
            warnings.append(
                "El motor de OCR activo no realiza reconocimiento de caracteres "
                "(PyTorch/EasyOCR no esta instalado en este entorno)."
            )
        elif not blocks:
            warnings.append("El OCR no detecto texto legible en la fotografia.")

        # 8. Validacion de la MRZ.
        mrz = extract_and_validate([block.text for block in blocks])
        warnings.extend(mrz.errors)

        # 9. Extraccion de campos.
        fields = parse_document(ocr.lines, mrz)
        warnings.extend(fields.warnings)

        # 10. Vista previa opcional.
        preview = self._build_preview(image) if include_preview else None

        confidence = _combined_confidence(fields, ocr, mrz, quality)

        logger.info(
            "Documento procesado en %.0f ms (motor=%s, confianza=%.2f, documento=%s)",
            elapsed_ms,
            self.engine.name,
            confidence,
            mask_document_number(fields.document_number),
        )

        return ScanResult(
            fields=fields,
            mrz=mrz,
            face=face,
            quality=quality,
            engine=self.engine.name,
            confidence=confidence,
            processing_ms=elapsed_ms,
            document_detected=document_detected,
            deskew_angle=deskew_angle,
            text_lines=ocr.lines,
            warnings=list(dict.fromkeys(warnings)),
            preview_jpeg=preview,
        )

    def _read_mrz_passes(self, image: np.ndarray) -> list[TextBlock]:
        """Aplica variantes de preprocesado a la franja de la MRZ.

        La tipografia OCR-B del documento es muy pequena: ampliarla y probar
        dos realces distintos aumenta notablemente la tasa de acierto sin
        penalizar el resto del pipeline.
        """
        blocks: list[TextBlock] = []
        variants: tuple[tuple[float, int, Any], ...] = (
            (0.22, 1600, enhance_for_ocr),
            (0.32, 2000, adaptive_binarize),
        )

        for ratio, min_width, transform in variants:
            band = upscale_small(crop_mrz_band(image, ratio=ratio), min_width=min_width)
            blocks.extend(self.engine.read(transform(band)))

        return blocks

    @staticmethod
    def _build_preview(image: np.ndarray) -> bytes | None:
        """Genera un JPEG reducido del documento procesado."""
        try:
            preview, _ = resize_max(image, PREVIEW_MAX_DIMENSION)
            return encode_jpeg(preview, quality=PREVIEW_JPEG_QUALITY)
        except Exception as exc:  # pragma: no cover - la vista previa es opcional
            logger.debug("No se pudo generar la vista previa: %s", exc)
            return None


_scanner = DocumentScanner()


def get_document_scanner() -> DocumentScanner:
    """Dependencia de FastAPI que entrega el escaner configurado."""
    return _scanner


__all__ = [
    "DocumentScanError",
    "DocumentScanner",
    "ScanResult",
    "get_document_scanner",
    "mask_document_number",
]
