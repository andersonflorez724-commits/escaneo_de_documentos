"""Reexporta el motor RapidOCR del backend (compatibilidad con benchmarks antiguos)."""

from __future__ import annotations

from app.services.ocr_engine import RapidOCREngine

__all__ = ["RapidOCREngine"]
