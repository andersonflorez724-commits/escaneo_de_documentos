"""Dobles de prueba para aislar el pipeline del modelo real.

Permiten probar el contrato del escaner (recorte, MRZ, parser, respuestas) en
milisegundos, sin cargar PyTorch.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from app.services.ocr_engine import BaseOCREngine, TextBlock


class StubOCREngine(BaseOCREngine):
    """Motor que devuelve siempre las mismas lineas de texto."""

    name = "stub"

    def __init__(self, texts: Sequence[str], confidence: float = 0.95) -> None:
        self.texts = list(texts)
        self.confidence = confidence
        self.calls: list[tuple[int, int]] = []

    def read(self, image: np.ndarray) -> list[TextBlock]:
        height, width = image.shape[:2]
        self.calls.append((height, width))

        blocks: list[TextBlock] = []
        for index, text in enumerate(self.texts):
            position = (index + 1) / (len(self.texts) + 1)
            blocks.append(
                TextBlock(
                    text=text,
                    confidence=self.confidence,
                    box=(10, int(height * position), max(1, width - 20), 30),
                    rel_box=(0.01, position, 0.98, 0.05),
                )
            )
        return blocks


class FailingOCREngine(BaseOCREngine):
    """Motor que falla, para comprobar el manejo de errores."""

    name = "failing"

    def read(self, image: np.ndarray) -> list[TextBlock]:  # noqa: ARG002
        raise RuntimeError("fallo simulado del motor de OCR")


class RecordingHeuristicEngine(BaseOCREngine):
    """Motor sin reconocimiento que registra las imagenes recibidas."""

    name = "heuristic"
    supports_recognition = False

    def __init__(self) -> None:
        self.calls: int = 0

    def read(self, image: np.ndarray) -> list[TextBlock]:  # noqa: ARG002
        self.calls += 1
        return []


__all__ = ["FailingOCREngine", "RecordingHeuristicEngine", "StubOCREngine"]
