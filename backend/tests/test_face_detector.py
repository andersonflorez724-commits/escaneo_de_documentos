"""Pruebas de la deteccion del rostro del documento."""

from __future__ import annotations

import numpy as np
import pytest

from app.services.face_detector import (
    CV2_AVAILABLE,
    HAAR_AVAILABLE,
    FaceDetection,
    HaarFaceDetector,
    HeuristicFaceDetector,
    _evaluate,
)
from tests.fixtures import render_synthetic_document, skin_tone_face_image

pytestmark = pytest.mark.skipif(not CV2_AVAILABLE, reason="OpenCV no esta instalado")


class TestHaarFaceDetector:
    def test_cascades_are_loaded(self) -> None:
        detector = HaarFaceDetector()
        if not HAAR_AVAILABLE:
            pytest.skip("Esta version de OpenCV no incluye las cascadas de Haar.")
        assert detector._load(), "No se cargo ninguna cascada de Haar"

    def test_returns_structured_negative_result_for_non_face_image(self) -> None:
        # Los clasificadores de Haar estan entrenados con fotografias reales:
        # sobre un dibujo sintetico lo correcto es no encontrar rostro.
        image = render_synthetic_document(with_face=False)
        result = HaarFaceDetector().detect(image)

        assert isinstance(result, FaceDetection)
        assert result.method == "opencv-haar"
        assert result.detected is False
        assert result.valid_photo is False
        assert result.warnings

    def test_detects_on_skin_tone_region_is_not_a_false_positive(self) -> None:
        # Un rectangulo de color piel no es un rostro: no debe validarse.
        result = HaarFaceDetector().detect(skin_tone_face_image())
        assert result.valid_photo is False


class TestHeuristicFaceDetector:
    def test_detects_the_photo_zone_of_a_document(self) -> None:
        image = render_synthetic_document(with_face=True)
        result = HeuristicFaceDetector().detect(image)

        assert result.detected is True
        assert result.method == "heuristic-skin"
        assert result.area_ratio > 0

    def test_does_not_detect_on_plain_document(self) -> None:
        image = render_synthetic_document(with_face=False)
        result = HeuristicFaceDetector().detect(image)
        assert result.detected is False


class TestPhotoValidityRules:
    """Reglas de `valid_photo` sobre el recorte del rostro."""

    def _image(self, *, blurry: bool = False, size: int = 600) -> np.ndarray:
        rng = np.random.default_rng(seed=3)
        image = np.full((size, size, 3), 200, dtype=np.uint8)
        image[150:400, 200:400] = rng.integers(0, 255, (250, 200, 3), dtype=np.uint8)

        if blurry:
            import cv2

            image[150:400, 200:400] = cv2.GaussianBlur(image[150:400, 200:400], (41, 41), 0)
        return image

    def test_sharp_face_region_is_valid(self) -> None:
        result = _evaluate(self._image(), (200, 150, 200, 250), confidence=0.9, method="test")

        assert result.detected is True
        assert result.valid_photo is True
        assert result.warnings == ()

    def test_blurry_face_region_is_rejected(self) -> None:
        result = _evaluate(self._image(blurry=True), (200, 150, 200, 250), confidence=0.9, method="test")

        assert result.detected is True
        assert result.valid_photo is False
        assert any("desenfocado" in warning for warning in result.warnings)

    def test_tiny_face_region_is_rejected(self) -> None:
        result = _evaluate(self._image(), (10, 10, 12, 12), confidence=0.9, method="test")

        assert result.valid_photo is False
        assert any("pequeno" in warning for warning in result.warnings)

    def test_huge_region_is_rejected(self) -> None:
        result = _evaluate(self._image(), (0, 0, 600, 600), confidence=0.9, method="test")

        assert result.valid_photo is False
        assert any("encuadre" in warning for warning in result.warnings)

    def test_as_dict_is_serializable(self) -> None:
        payload = _evaluate(self._image(), (200, 150, 200, 250), confidence=0.9, method="test").as_dict()

        assert payload["box"] == [200, 150, 200, 250]
        assert isinstance(payload["warnings"], list)
