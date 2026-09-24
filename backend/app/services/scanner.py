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

Un documento de identificacion tiene **dos caras** y cada una aporta datos
distintos (la frontal el numero, el nombre y la foto; el reverso las fechas, el
lugar de nacimiento y el grupo sanguineo). :meth:`DocumentScanner.scan_sides`
procesa todas las caras recibidas y **fusiona** su texto antes de extraer los
campos, de modo que la validacion de la MRZ del reverso sirva tambien para
comprobar el numero leido en la frontal.
"""

from __future__ import annotations

import base64
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

from app.core.config import Settings, get_settings
from app.core.errors import DocumentScanError, PayloadTooLargeError
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

# Etiquetas legibles de las caras del documento.
SIDE_LABELS = {
    "front": "cara frontal",
    "back": "reverso",
    "extra": "cara adicional",
}

SINGLE_SIDE_WARNING = (
    "Se analizo una sola cara del documento: el reverso aporta la fecha y el "
    "lugar de nacimiento, la expedicion y el grupo sanguineo."
)


__all__ = [
    "DocumentScanError",
    "DocumentScanner",
    "SIDE_LABELS",
    "ScanResult",
    "SideInfo",
    "get_document_scanner",
    "mask_document_number",
]


@dataclass(slots=True)
class SideInfo:
    """Resumen del analisis de una de las caras del documento."""

    side: str
    ok: bool = True
    width: int = 0
    height: int = 0
    quality_score: float = 0.0
    document_detected: bool = False
    deskew_angle: float = 0.0
    face_detected: bool = False
    valid_photo: bool = False
    used_mrz_passes: bool = False
    text_lines: list[str] = field(default_factory=list)
    processing_ms: float = 0.0
    error: str | None = None
    preview_jpeg: bytes | None = None

    @property
    def label(self) -> str:
        return SIDE_LABELS.get(self.side, self.side)

    def as_dict(self) -> dict[str, Any]:
        return {
            "side": self.side,
            "label": self.label,
            "ok": self.ok,
            "width": self.width,
            "height": self.height,
            "quality_score": round(self.quality_score, 4),
            "document_detected": self.document_detected,
            "deskew_angle": round(self.deskew_angle, 2),
            "face_detected": self.face_detected,
            "valid_photo": self.valid_photo,
            "used_mrz_passes": self.used_mrz_passes,
            "text_lines": list(self.text_lines),
            "processing_ms": round(self.processing_ms, 2),
            "error": self.error,
            "preview_base64": (
                base64.b64encode(self.preview_jpeg).decode("ascii") if self.preview_jpeg else None
            ),
        }


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
    used_mrz_passes: bool = False
    sides: list[SideInfo] = field(default_factory=list)

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

    @property
    def both_sides(self) -> bool:
        """Se analizaron al menos dos caras del documento."""
        return len([side for side in self.sides if side.ok]) >= 2


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


@dataclass(slots=True)
class _SideAnalysis:
    """Resultado intermedio del analisis de una cara."""

    info: SideInfo
    blocks: list[TextBlock] = field(default_factory=list)
    quality: ImageQuality | None = None
    face: FaceDetection | None = None
    warnings: list[str] = field(default_factory=list)


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
        self.used_mrz_passes = False

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
        """Procesa la fotografia de una cara del documento y extrae sus campos.

        Args:
            data: Bytes de la imagen (JPEG/PNG/...).
            include_preview: Si es ``True`` agrega un JPEG reducido del
                documento ya recortado y enderezado.

        Raises:
            InvalidImageError: si la imagen no se puede decodificar.
            DocumentScanError: si el motor de OCR falla inesperadamente.

        Es el equivalente a ``scan_sides([("front", data)])``: se mantiene para
        quien captura una sola imagen (por ejemplo el reverso de un pasaporte).
        """
        return self.scan_sides([("front", data)], include_preview=include_preview)

    def scan_sides(
        self,
        sides: Sequence[tuple[str, bytes]],
        *,
        include_preview: bool = False,
    ) -> ScanResult:
        """Procesa una o varias caras del documento y fusiona sus resultados.

        Args:
            sides: Pares ``(cara, bytes)`` en el orden de captura. La cara es
                una etiqueta libre (``front``, ``back``, ...) que solo se usa
                para informar y para etiquetar las advertencias.
            include_preview: Si es ``True`` agrega una vista previa JPEG en
                base64 del documento recortado de cada cara.

        Raises:
            InvalidImageError: si **ninguna** de las imagenes se puede decodificar.
            PayloadTooLargeError: si alguna imagen supera el limite de tamano.
            DocumentScanError: si no se recibe ninguna imagen.
        """
        if not sides:
            raise DocumentScanError("No se recibio ninguna imagen para analizar.")

        started = time.perf_counter()

        analyses: list[_SideAnalysis] = []
        for side, data in sides:
            if len(data) > self.settings.max_upload_bytes:
                raise PayloadTooLargeError(
                    f"La imagen supera el limite de {self.settings.max_upload_mb} MB."
                )
            analyses.append(self._analyse_side(data, side, include_preview=include_preview))

        usable = [analysis for analysis in analyses if analysis.info.ok]
        if not usable:
            # Ninguna cara se pudo decodificar: se propaga el error como si se
            # hubiera enviado una sola imagen (422 invalid_image).
            raise InvalidImageError(
                analyses[0].info.error or "No se pudo decodificar la imagen enviada."
            )

        warnings: list[str] = []
        multiple = len(analyses) > 1
        for analysis in analyses:
            prefix = f"[{analysis.info.label}] " if multiple else ""
            warnings.extend(f"{prefix}{warning}" for warning in analysis.warnings)

        # El texto de todas las caras se fusiona: el parser trabaja con
        # etiquetas impresas, asi que una sola pasada encuentra los campos de
        # la frontal y del reverso, y la MRZ del reverso valida el numero leido
        # en la frontal. Los renglones se agrupan **por cara**: cada imagen
        # tiene su propio sistema de coordenadas y mezclarlas produciria
        # renglones sin sentido.
        blocks: list[TextBlock] = []
        lines: list[str] = []
        for analysis in usable:
            side_blocks = _dedupe_blocks(analysis.blocks)
            blocks.extend(side_blocks)
            lines.extend(OCRResult(engine=self.engine.name, blocks=side_blocks).lines)

        elapsed_ms = (time.perf_counter() - started) * 1000.0

        ocr = OCRResult(engine=self.engine.name, blocks=blocks, elapsed_ms=elapsed_ms)

        if not self.engine.supports_recognition:
            warnings.append(
                "El motor de OCR activo no realiza reconocimiento de caracteres "
                "(RapidOCR/EasyOCR no esta instalado en este entorno)."
            )
        elif not blocks:
            warnings.append("El OCR no detecto texto legible en la fotografia.")

        mrz = extract_and_validate([*(block.text for block in blocks), *lines])
        warnings.extend(mrz.errors)

        fields = parse_document(lines, mrz)
        warnings.extend(fields.warnings)

        face = self._best_face(usable)
        quality = self._best_quality(usable)
        self.used_mrz_passes = any(analysis.info.used_mrz_passes for analysis in usable)

        if len(usable) == 1:
            warnings.append(SINGLE_SIDE_WARNING)

        confidence = _combined_confidence(fields, ocr, mrz, quality)

        logger.info(
            "Documento procesado en %.0f ms (caras=%d, motor=%s, confianza=%.2f, documento=%s, mrz_extra=%s)",
            elapsed_ms,
            len(usable),
            self.engine.name,
            confidence,
            mask_document_number(fields.document_number),
            self.used_mrz_passes,
        )

        return ScanResult(
            fields=fields,
            mrz=mrz,
            face=face,
            quality=quality,
            engine=self.engine.name,
            confidence=confidence,
            processing_ms=elapsed_ms,
            document_detected=any(analysis.info.document_detected for analysis in usable),
            deskew_angle=usable[0].info.deskew_angle,
            text_lines=lines[: self.settings.ocr_max_text_lines],
            warnings=list(dict.fromkeys(warnings)),
            preview_jpeg=next(
                (analysis.info.preview_jpeg for analysis in usable if analysis.info.preview_jpeg),
                None,
            ),
            used_mrz_passes=self.used_mrz_passes,
            sides=[analysis.info for analysis in analyses],
        )

    # ------------------------------------------------------------ una cara
    def _analyse_side(
        self,
        data: bytes,
        side: str,
        *,
        include_preview: bool,
    ) -> _SideAnalysis:
        """Ejecuta el pipeline sobre una de las caras del documento."""
        info = SideInfo(side=side)
        analysis = _SideAnalysis(info=info)
        side_started = time.perf_counter()

        try:
            # 1. Decodificacion y normalizacion de tamano.
            try:
                image = decode_image(data)
            except InvalidImageError as exc:
                info.ok = False
                info.error = str(exc)
                return analysis

            image, _ = resize_max(image, self.settings.max_image_dimension)
            info.width, info.height = int(image.shape[1]), int(image.shape[0])

            # 2. Calidad de la foto original (antes de recortar).
            quality = analyze_quality(image)
            analysis.quality = quality
            info.quality_score = quality.score
            analysis.warnings.extend(quality.warnings)

            # 3. Localizacion y rectificacion del documento.
            quad = find_document_quad(image)
            info.document_detected = quad is not None
            if quad is not None:
                image = warp_document(image, quad)
            else:
                analysis.warnings.append(
                    "No se detecto el contorno del documento; se proceso la imagen completa."
                )

            # 4. Enderezado fino.
            image, deskew_angle = deskew(image)
            info.deskew_angle = deskew_angle

            # 5. Rostro del documento.
            face = self.face_detector.detect(image)
            analysis.face = face
            info.face_detected = face.detected
            info.valid_photo = face.valid_photo
            analysis.warnings.extend(face.warnings)

            # 6. OCR de la pagina completa.
            blocks = list(self.engine.read(enhance_for_ocr(image)))

            # 7. Pasadas extra dedicadas a la MRZ. Solo se ejecutan si la
            #    lectura de la pagina completa no basto: en documentos bien
            #    encuadrados esto ahorra dos tercios del tiempo de inferencia.
            # Los renglones agrupados se suman a los bloques crudos: cuando el
            # detector parte una linea de la MRZ en dos, la union de ambos
            # trozos es lo unico que la reconstruye.
            preliminary_lines = OCRResult(engine=self.engine.name, blocks=blocks).lines
            preliminary = extract_and_validate(
                [*(block.text for block in blocks), *preliminary_lines]
            )
            info.used_mrz_passes = not (
                preliminary.detected and preliminary.valid and preliminary.complete
            )
            if info.used_mrz_passes:
                blocks.extend(self._read_mrz_passes(image, already_read=blocks))

            analysis.blocks = blocks
            info.text_lines = [block.text for block in blocks]

            # 8. Vista previa opcional de la cara procesada.
            if include_preview:
                info.preview_jpeg = self._build_preview(image)
        finally:
            # Los fallos del motor suben tal cual: solo la decodificacion de la
            # imagen se tolera cara a cara (ver `scan_sides`).
            info.processing_ms = (time.perf_counter() - side_started) * 1000.0

        return analysis

    def _read_mrz_passes(self, image: np.ndarray, already_read: list[TextBlock]) -> list[TextBlock]:  # noqa: D401
        """Aplica variantes de preprocesado a la franja de la MRZ.

        La tipografia OCR-B del documento es muy pequena: ampliar la franja y
        probar realces distintos sube mucho la tasa de acierto. Las variantes
        se prueban **en orden y se cortan en cuanto una produce una MRZ
        completa y consistente**, porque cada pasada cuesta segundos de
        inferencia.
        """
        collected: list[TextBlock] = []
        accumulated = list(already_read)

        variants: tuple[tuple[float, int, Any], ...] = (
            (0.22, 1600, enhance_for_ocr),
            (0.32, 1600, adaptive_binarize),
        )

        for ratio, min_width, transform in variants:
            band = upscale_small(crop_mrz_band(image, ratio=ratio), min_width=min_width)
            found = self.engine.read(transform(band))
            collected.extend(found)
            accumulated.extend(found)

            if not self.engine.supports_recognition:
                break

            check = extract_and_validate([block.text for block in accumulated])
            if check.detected and check.valid and check.complete:
                break

        return collected

    @staticmethod
    def _best_face(analyses: Sequence[_SideAnalysis]) -> FaceDetection:
        """Elige la inspeccion de rostro mas favorable entre las caras.

        La foto del titular vive en la cara frontal; si solo se fotografio el
        reverso no habra rostro y el resultado sera negativo.
        """
        detected = [analysis for analysis in analyses if analysis.face and analysis.face.detected]
        if not detected:
            for analysis in analyses:
                if analysis.face is not None:
                    return analysis.face
            return FaceDetection(
                detected=False,
                count=0,
                confidence=0.0,
                method="none",
                valid_photo=False,
                warnings=("No se analizo ninguna cara con fotografia del titular.",),
            )

        return max(
            detected,
            key=lambda analysis: (
                analysis.face.valid_photo,
                analysis.face.sharpness or 0.0,
            ),
        ).face  # type: ignore[return-value]

    @staticmethod
    def _best_quality(analyses: Sequence[_SideAnalysis]) -> ImageQuality:
        """Devuelve la calidad de la cara mejor fotografiada."""
        qualities = [analysis.quality for analysis in analyses if analysis.quality is not None]
        if not qualities:  # pragma: no cover - defensivo
            raise DocumentScanError("No se pudo analizar la calidad de ninguna imagen.")
        return max(qualities, key=lambda quality: quality.score)

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
