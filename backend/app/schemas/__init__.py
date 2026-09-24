"""Esquemas Pydantic (contratos de entrada/salida de la API)."""

from app.schemas.document import (
    DocumentFieldsOut,
    DocumentScanResponse,
    FaceInfo,
    MRZInfo,
    MRZRequest,
    MRZValidationResponse,
    QualityInfo,
    SideInfoOut,
)

__all__ = [
    "DocumentFieldsOut",
    "DocumentScanResponse",
    "FaceInfo",
    "MRZInfo",
    "MRZRequest",
    "MRZValidationResponse",
    "QualityInfo",
    "SideInfoOut",
]
