"""Endpoints de lectura e inspeccion de documentos de identificacion."""

from __future__ import annotations

import logging
from typing import Annotated

from fastapi import APIRouter, Body, Depends, File, HTTPException, Query, UploadFile, status

from app.api.deps import CurrentUser
from app.core.config import get_settings
from app.schemas.document import (
    DocumentFieldsOut,
    DocumentScanResponse,
    FaceInfo,
    MRZInfo,
    MRZRequest,
    MRZValidationResponse,
    QualityInfo,
)
from app.services.image_utils import ALLOWED_MIME_TYPES, InvalidImageError
from app.services.mrz import extract_and_validate, parse_mrz
from app.services.scanner import (
    DocumentScanner,
    DocumentScanError,
    ScanResult,
    get_document_scanner,
    mask_document_number,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Documentos"])

settings = get_settings()

FILE_DESCRIPTION = (
    "Fotografia del documento de identificacion (JPEG, PNG o WEBP). "
    "Puede ser la captura directa de la camara del dispositivo."
)

MRZ_EXAMPLES = {
    "pasaporte_td3": {
        "summary": "Pasaporte (TD3) valido: ejemplo oficial de ICAO 9303",
        "value": {
            "mrz_lines": [
                "P<UTOERIKSSON<<ANNA<MARIA<<<<<<<<<<<<<<<<<<<",
                "L898902C36UTO7408122F1204159ZE184226B<<<<<10",
            ]
        },
    },
    "documento_alterado": {
        "summary": "Documento con el numero manipulado (falla el digito de control)",
        "value": {
            "mrz_lines": [
                "P<UTOERIKSSON<<ANNA<MARIA<<<<<<<<<<<<<<<<<<<",
                "L898902C46UTO7408122F1204159ZE184226B<<<<<10",
            ]
        },
    },
    "texto_ocr": {
        "summary": "Texto crudo del OCR: las lineas se localizan automaticamente",
        "value": {
            "text": (
                "REPUBLICA DE COLOMBIA\n"
                "P<UTOERIKSSON<<ANNA<MARIA<<<<<<<<<<<<<<<<<<<\n"
                "L898902C36UTO7408122F1204159ZE184226B<<<<<10"
            )
        },
    },
}


async def _read_upload(upload: UploadFile, max_bytes: int) -> bytes:
    """Lee el archivo subido respetando el limite de tamano.

    Se lee por bloques para no cargar en memoria un archivo enorme antes de
    validarlo.
    """
    if upload.content_type and upload.content_type.lower() not in ALLOWED_MIME_TYPES:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=(
                f"El tipo de archivo '{upload.content_type}' no esta soportado. "
                "Envia una imagen JPEG, PNG o WEBP."
            ),
        )

    chunks: list[bytes] = []
    total = 0
    while chunk := await upload.read(64 * 1024):
        total += len(chunk)
        if total > max_bytes:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"La imagen supera el limite de {settings.max_upload_mb} MB.",
            )
        chunks.append(chunk)
    await upload.close()

    if total == 0:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="El archivo recibido esta vacio.",
        )

    return b"".join(chunks)


def _build_response(result: ScanResult) -> DocumentScanResponse:
    """Traduce el resultado del escaneo al contrato publico de la API."""
    fields = result.fields

    if not fields.document_number and not fields.name:
        message = "No se pudo extraer informacion legible del documento. Intenta con mejor iluminacion."
    elif result.mrz.detected and not result.mrz.valid:
        message = "Datos extraidos, pero la MRZ del documento no es consistente."
    elif not result.valid_photo:
        message = "Datos extraidos, pero la foto del titular no supera la inspeccion."
    else:
        message = "Documento leido y validado correctamente."

    return DocumentScanResponse(
        success=bool(fields.document_number or fields.name),
        message=message,
        document_number=fields.document_number,
        name=fields.name,
        valid_photo=result.valid_photo,
        document_type=fields.document_type,
        confidence=round(result.confidence, 4),
        fields=DocumentFieldsOut.model_validate(fields.as_dict()),
        mrz=MRZInfo.model_validate(result.mrz.as_dict()),
        face=FaceInfo.model_validate(result.face.as_dict()),
        quality=QualityInfo.model_validate(result.quality.as_dict()),
        engine=result.engine,
        processing_ms=round(result.processing_ms, 2),
        document_detected=result.document_detected,
        deskew_angle=round(result.deskew_angle, 2),
        text_lines=result.text_lines,
        warnings=result.warnings,
        preview_base64=result.preview_base64,
    )


@router.post(
    "/scan-document",
    response_model=DocumentScanResponse,
    status_code=status.HTTP_200_OK,
    summary="Leer los datos de un documento de identificacion",
    description=(
        "Recibe la **foto del documento** tomada por la camara y devuelve los "
        "datos del titular.\n\n"
        "El pipeline ejecuta: decodificacion e.g. EXIF, analisis de calidad, "
        "deteccion y rectificacion del contorno del documento, enderezado, "
        "deteccion del rostro, OCR con el modelo preentrenado (pagina completa "
        "y franja MRZ) y validacion ICAO 9293.\n\n"
        "Campos minimos garantizados en la respuesta: `document_number`, "
        "`name` y `valid_photo`."
    ),
    responses={
        401: {"description": "Token JWT ausente, invalido o expirado."},
        413: {"description": "La imagen supera el limite de tamano."},
        415: {"description": "Tipo de archivo no soportado."},
        422: {"description": "La imagen no se pudo decodificar."},
    },
)
async def scan_document(
    file: Annotated[
        UploadFile,
        File(
            description=FILE_DESCRIPTION,
            examples=["documento.jpg"],
        ),
    ],
    current_user: CurrentUser,
    scanner: Annotated[DocumentScanner, Depends(get_document_scanner)],
    include_preview: Annotated[
        bool,
        Query(
            description=(
                "Incluye una vista previa JPEG en base64 del documento ya recortado "
                "y enderezado. Aumenta el tamano de la respuesta."
            )
        ),
    ] = False,
) -> DocumentScanResponse:
    data = await _read_upload(file, settings.max_upload_bytes)

    try:
        result = scanner.scan(data, include_preview=include_preview)
    except InvalidImageError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc
    except DocumentScanError as exc:
        logger.warning("Fallo el procesamiento del documento: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="No se pudo procesar el documento.",
        ) from exc
    except Exception as exc:  # pragma: no cover - salvaguarda
        logger.exception("Error inesperado procesando el documento")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error interno al procesar la imagen.",
        ) from exc

    logger.info(
        "Usuario %s escaneo un documento (numero=%s, confianza=%.2f)",
        current_user.email,
        mask_document_number(result.document_number),
        result.confidence,
    )

    return _build_response(result)


@router.post(
    "/validate-mrz",
    response_model=MRZValidationResponse,
    status_code=status.HTTP_200_OK,
    summary="Validar la consistencia de caracteres de una MRZ",
    description=(
        "Recalcula los **digitos de control** de la zona de lectura mecanica y "
        "los compara con los impresos en el documento (estandar ICAO 9293: "
        "modulo 10 con pesos 7-3-1).\n\n"
        "Puede recibir las lineas ya separadas en `mrz_lines` o el texto crudo "
        "del OCR en `text`, del cual se localizan automaticamente las lineas.\n\n"
        "Se soportan los formatos **TD1** (3x30, cedulas y DNI), **TD2** (2x36) "
        "y **TD3** (2x44, pasaportes). Ante varias lecturas del mismo renglon "
        "se elige la que supera mas comprobaciones.\n\n"
        "Un documento autentico devuelve `valid: true`; una fotocopia mal leida "
        "o un documento manipulado falla en al menos un digito de control."
    ),
    responses={
        401: {"description": "Token JWT ausente, invalido o expirado."},
        422: {"description": "No se envio ni `mrz_lines` ni `text`."},
    },
)
async def validate_mrz(
    payload: Annotated[MRZRequest, Body(openapi_examples=MRZ_EXAMPLES)],
    current_user: CurrentUser,
) -> MRZValidationResponse:
    if payload.mrz_lines:
        result = parse_mrz(payload.mrz_lines)
    else:
        result = extract_and_validate((payload.text or "").splitlines())

    if not result.detected:
        message = "No se encontro una MRZ legible en el contenido enviado."
    elif result.valid:
        message = (
            f"MRZ {result.mrz_format} consistente: "
            f"{result.total_checks} de {result.total_checks} digitos de control coinciden."
        )
    else:
        inconsistent = ", ".join(result.inconsistent_fields)
        message = (
            f"MRZ inconsistente: {result.passed_checks} de {result.total_checks} "
            f"digitos de control coinciden. Campos con error: {inconsistent}."
        )

    logger.info(
        "Usuario %s valido una MRZ (formato=%s, valida=%s)",
        current_user.email,
        result.mrz_format,
        result.valid,
    )

    return MRZValidationResponse(
        valid=result.valid,
        message=message,
        inconsistent_fields=result.inconsistent_fields,
        mrz=MRZInfo.model_validate(result.as_dict()),
    )
