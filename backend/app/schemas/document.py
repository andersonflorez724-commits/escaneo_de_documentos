"""Esquemas Pydantic para el escaneo y la validacion de documentos."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

# ---------------------------------------------------------------------------
# Piezas reutilizables
# ---------------------------------------------------------------------------
class QualityInfo(BaseModel):
    """Metricas de calidad de la fotografia recibida."""

    model_config = ConfigDict(extra="ignore")

    width: int = Field(description="Ancho en pixeles tras el escalado.", examples=[1600])
    height: int = Field(description="Alto en pixeles tras el escalado.", examples=[1000])
    megapixels: float = Field(description="Megapixeles de la imagen procesada.", examples=[1.6])
    blur_score: float = Field(description="Varianza del laplaciano; valores bajos indican desenfoque.", examples=[320.5])
    brightness: float = Field(description="Brillo medio (0-255).", examples=[142.3])
    contrast: float = Field(description="Desviacion estandar del gris.", examples=[58.1])
    is_blurry: bool = Field(description="La imagen parece movida.", examples=[False])
    is_too_dark: bool = Field(description="La imagen esta subexpuesta.", examples=[False])
    is_overexposed: bool = Field(description="La imagen tiene reflejos o exceso de luz.", examples=[False])
    is_low_resolution: bool = Field(description="Resolucion insuficiente para un OCR fiable.", examples=[False])
    score: float = Field(ge=0.0, le=1.0, description="Puntaje compuesto de calidad (0-1).", examples=[0.82])
    warnings: list[str] = Field(default_factory=list, description="Advertencias sobre la foto.")


class FaceInfo(BaseModel):
    """Resultado de la inspeccion del rostro del documento."""

    detected: bool = Field(description="Se encontro un rostro en el documento.", examples=[True])
    count: int = Field(default=0, description="Numero de candidatos localizados.", examples=[1])
    confidence: float = Field(default=0.0, ge=0.0, le=1.0, description="Confianza de la deteccion.", examples=[0.78])
    method: str = Field(description="Algoritmo usado (`opencv-haar` o `heuristic-skin`).", examples=["opencv-haar"])
    box: list[int] | None = Field(default=None, description="Cuadro del rostro `[x, y, ancho, alto]` en pixeles.")
    sharpness: float = Field(default=0.0, description="Nitidez del recorte del rostro.", examples=[180.4])
    area_ratio: float = Field(default=0.0, description="Proporcion del area del documento que ocupa el rostro.", examples=[0.04])
    valid_photo: bool = Field(description="La foto del titular es legible y utilizable.", examples=[True])
    warnings: list[str] = Field(default_factory=list)


class MRZInfo(BaseModel):
    """Resultado del analisis de la zona de lectura mecanica."""

    detected: bool = Field(description="Se encontro una MRZ legible.", examples=[True])
    format: str | None = Field(default=None, description="Formato ICAO detectado: `TD1`, `TD2` o `TD3`.", examples=["TD3"])
    lines: list[str] = Field(default_factory=list, description="Lineas tal como llegaron del OCR.")
    normalized_lines: list[str] = Field(default_factory=list, description="Lineas normalizadas a la longitud oficial.")
    repairs: list[str] = Field(default_factory=list, description="Correcciones aplicadas al texto del OCR.")
    errors: list[str] = Field(default_factory=list, description="Problemas encontrados durante el analisis.")
    document_type: str | None = Field(default=None, description="Codigo de tipo de documento de la MRZ.", examples=["P"])
    issuing_country: str | None = Field(default=None, description="Codigo ISO del pais emisor.", examples=["UTO"])
    document_number: str | None = Field(default=None, examples=["L898902C3"])
    nationality: str | None = Field(default=None, examples=["UTO"])
    surname: str | None = Field(default=None, examples=["ERIKSSON"])
    given_names: str | None = Field(default=None, examples=["ANNA MARIA"])
    full_name: str | None = Field(default=None, examples=["ANNA MARIA ERIKSSON"])
    birth_date: str | None = Field(default=None, description="Fecha ISO `YYYY-MM-DD`.", examples=["1974-08-12"])
    expiry_date: str | None = Field(default=None, description="Fecha ISO `YYYY-MM-DD`.", examples=["2012-04-15"])
    sex: str | None = Field(default=None, description="`M`, `F` o `None`.", examples=["F"])
    personal_number: str | None = Field(default=None)
    checks: dict[str, bool] = Field(default_factory=dict, description="Resultado de cada digito de control.")
    passed_checks: int = Field(default=0, description="Digitos de control correctos.", examples=[5])
    total_checks: int = Field(default=0, description="Digitos de control evaluados.", examples=[5])
    inconsistent_fields: list[str] = Field(
        default_factory=list,
        description="Campos cuyo digito de control no coincide (documento ilegible o alterado).",
        examples=[[]],
    )
    complete: bool = Field(default=False, description="Se leyeron todas las lineas de la MRZ.", examples=[True])
    score: float = Field(default=0.0, ge=0.0, le=1.0, description="Proporcion de checks correctos.", examples=[1.0])
    valid: bool = Field(default=False, description="La MRZ es consistente (todos los checks correctos).", examples=[True])


class MRZRequest(BaseModel):
    """Entrada del endpoint `/validate-mrz`.

    Acepta las lineas MRZ ya separadas o el texto crudo del OCR del que se
    extraen automaticamente.
    """

    mrz_lines: list[str] | None = Field(
        default=None,
        max_length=3,
        description="Lineas de la MRZ (2 para TD2/TD3, 3 para TD1).",
        examples=[["P<UTOERIKSSON<<ANNA<MARIA<<<<<<<<<<<<<<<<<<<", "L898902C36UTO7408122F1204159ZE184226B<<<<<10"]],
    )
    text: str | None = Field(
        default=None,
        description="Alternativa: texto completo del OCR del que se localizan las lineas MRZ.",
        examples=["REPUBLICA DE COLOMBIA\nP<UTOERIKSSON<<ANNA<MARIA<<<<<<<<<<<<<<<<<<<\nL898902C36UTO7408122F1204159ZE184226B<<<<<10"],
    )

    @model_validator(mode="after")
    def _require_input(self) -> "MRZRequest":
        has_lines = bool(self.mrz_lines and any(str(line).strip() for line in self.mrz_lines))
        has_text = bool(self.text and self.text.strip())
        if not has_lines and not has_text:
            raise ValueError("Debes enviar `mrz_lines` o bien `text` con el contenido de la MRZ.")
        return self


class MRZValidationResponse(BaseModel):
    """Resultado de la validacion de consistencia de la MRZ."""

    valid: bool = Field(description="Todas las comprobaciones pasaron.", examples=[True])
    message: str = Field(description="Resumen legible del resultado.")
    inconsistent_fields: list[str] = Field(default_factory=list, description="Campos cuyo digito de control no coincide.")
    mrz: MRZInfo = Field(description="Detalle completo del analisis.")


class SideInfoOut(BaseModel):
    """Resumen del analisis de una de las caras del documento."""

    side: str = Field(description="Etiqueta de la cara: `front`, `back`, ...", examples=["front"])
    label: str = Field(description="Nombre legible de la cara.", examples=["cara frontal"])
    ok: bool = Field(description="La imagen de esta cara se pudo procesar.", examples=[True])
    width: int = Field(default=0, description="Ancho en pixeles tras el escalado.", examples=[1600])
    height: int = Field(default=0, description="Alto en pixeles tras el escalado.", examples=[1000])
    quality_score: float = Field(default=0.0, ge=0.0, le=1.0, description="Calidad de esta cara.")
    document_detected: bool = Field(default=False, description="Se detecto el contorno del documento.")
    deskew_angle: float = Field(default=0.0, description="Angulo de inclinacion corregido, en grados.")
    face_detected: bool = Field(default=False, description="Se encontro un rostro en esta cara.")
    valid_photo: bool = Field(default=False, description="La foto del titular de esta cara es valida.")
    used_mrz_passes: bool = Field(default=False, description="Necesito pasadas extra para la MRZ.")
    processing_ms: float = Field(default=0.0, description="Tiempo de procesamiento de la cara, en milisegundos.")
    error: str | None = Field(default=None, description="Motivo por el que la cara no se pudo procesar.")
    preview_base64: str | None = Field(default=None, description="Vista previa JPEG en base64 de esta cara.")


class DocumentFieldsOut(BaseModel):
    """Campos extraidos del documento y su origen."""

    document_number: str | None = Field(default=None, examples=["1.033.186.199"])
    name: str | None = Field(default=None, description="Nombres y apellidos del titular.", examples=["JUAN JOSE OCAMPO ARTEAGA"])
    document_type: str | None = Field(default=None, examples=["TARJETA DE IDENTIDAD"])
    first_surname: str | None = Field(default=None, description="Primer apellido.", examples=["OCAMPO"])
    second_surname: str | None = Field(default=None, description="Segundo apellido.", examples=["ARTEAGA"])
    given_names: str | None = Field(default=None, description="Nombres de pila.", examples=["JUAN JOSE"])
    birth_date: str | None = Field(default=None, examples=["2004-04-15"])
    birth_place: str | None = Field(default=None, description="Lugar de nacimiento.", examples=["MEDELLIN (ANTIOQUIA)"])
    issue_date: str | None = Field(default=None, description="Fecha de expedicion.", examples=["2022-04-20"])
    issue_place: str | None = Field(default=None, description="Lugar de expedicion.", examples=["CARTAGENA"])
    expiry_date: str | None = Field(default=None, examples=["2032-04-19"])
    sex: str | None = Field(default=None, examples=["F"])
    blood_type: str | None = Field(default=None, description="Grupo sanguineo con el factor RH.", examples=["O+"])
    nationality: str | None = Field(default=None, examples=["COL"])
    issuing_country: str | None = Field(default=None, examples=["COL"])
    confidence: float = Field(default=0.0, ge=0.0, le=1.0, description="Confianza de la extraccion.", examples=[0.91])
    sources: dict[str, str] = Field(
        default_factory=dict,
        description="Origen de cada campo (`mrz` o `visual`).",
    )
    warnings: list[str] = Field(default_factory=list)


class DocumentScanResponse(BaseModel):
    """Respuesta del endpoint `/scan-document`."""

    success: bool = Field(description="El documento se proceso correctamente.", examples=[True])
    message: str = Field(description="Resumen legible del resultado.")

    # Campos destacados que pide el enunciado del taller.
    document_number: str | None = Field(default=None, description="Numero de documento extraido.", examples=["12345678"])
    name: str | None = Field(default=None, description="Nombre del titular.", examples=["JUAN PEREZ"])
    valid_photo: bool = Field(default=False, description="El documento tiene una foto de rostro valida.", examples=[True])

    document_type: str | None = Field(default=None, examples=["CEDULA DE CIUDADANIA"])
    confidence: float = Field(default=0.0, ge=0.0, le=1.0, description="Confianza global del escaneo.", examples=[0.88])

    fields: DocumentFieldsOut = Field(description="Todos los campos extraidos.")
    mrz: MRZInfo = Field(description="Analisis de la zona de lectura mecanica.")
    face: FaceInfo = Field(description="Inspeccion del rostro.")
    quality: QualityInfo = Field(description="Calidad de la fotografia.")

    # Analisis cara a cara: la frontal aporta el numero, el nombre y la foto; el
    # reverso las fechas, el lugar de nacimiento y el grupo sanguineo.
    sides: list[SideInfoOut] = Field(
        default_factory=list,
        description="Resumen del analisis de cada cara enviada (frontal, reverso, ...).",
    )

    engine: str = Field(description="Motor de OCR utilizado.", examples=["easyocr"])
    processing_ms: float = Field(description="Tiempo total de procesamiento en milisegundos.", examples=[2480.5])
    sides_processed: int = Field(default=0, description="Numero de caras analizadas correctamente.")
    both_sides: bool = Field(
        default=False,
        description="Se analizaron la cara frontal y el reverso: la lectura es completa.",
    )
    document_detected: bool = Field(default=False, description="Se detecto y recorto el contorno del documento.")
    deskew_angle: float = Field(default=0.0, description="Angulo de inclinacion corregido, en grados.")
    text_lines: list[str] = Field(default_factory=list, description="Texto crudo reconocido por el OCR.")
    warnings: list[str] = Field(default_factory=list, description="Advertencias del procesamiento.")
    preview_base64: str | None = Field(
        default=None,
        description="Vista previa JPEG en base64 del documento recortado (si se solicito).",
    )


class ErrorResponse(BaseModel):
    """Formato uniforme de los errores de la API.

    Todas las respuestas con codigo >= 400 comparten esta forma, de modo que
    el cliente pueda tratarlas igual sin inspeccionar el cuerpo.
    """

    detail: str | list[dict[str, Any]] = Field(
        description="Descripcion del error o lista de errores de validacion.",
        examples=["La imagen supera el limite de 8 MB."],
    )
    code: str = Field(description="Codigo interno del error.", examples=["invalid_image"])
    request_id: str | None = Field(
        default=None,
        description="Identificador de la peticion, para correlacionar con los registros.",
        examples=["9f2a1c3d4e5b6a7c"],
    )
    context: dict[str, Any] | None = Field(default=None, description="Informacion adicional para depuracion.")


__all__ = [
    "DocumentFieldsOut",
    "DocumentScanResponse",
    "ErrorResponse",
    "FaceInfo",
    "MRZInfo",
    "MRZRequest",
    "MRZValidationResponse",
    "QualityInfo",
    "SideInfoOut",
]
