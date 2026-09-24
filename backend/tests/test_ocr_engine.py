"""Pruebas de los motores de OCR (fabrica y RapidOCR, sin inferencia pesada)."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from app.core.config import Settings, get_settings
from app.services.ocr_engine import (
    EasyOCREngine,
    HeuristicOCREngine,
    OCRResult,
    RapidOCREngine,
    _build_engine,
    easyocr_is_available,
    get_ocr_engine,
    rapidocr_is_available,
    reset_ocr_engine,
    warmup_ocr_engine,
)
from app.services.ocr_engine import TextBlock, group_blocks_into_lines
from tests.fixtures import document_bytes, encode


def blank_image(height: int = 400, width: int = 600) -> np.ndarray:
    return np.full((height, width, 3), 240, dtype=np.uint8)


def sample_blocks() -> list[TextBlock]:
    return [
        TextBlock(text="REPUBLICA", confidence=0.9, box=(10, 10, 100, 20), rel_box=(0.1, 0.1, 0.2, 0.05)),
        TextBlock(text="DE COLOMBIA", confidence=0.8, box=(120, 12, 140, 20), rel_box=(0.4, 0.11, 0.3, 0.05)),
        TextBlock(text="NUMERO", confidence=0.7, box=(10, 80, 60, 20), rel_box=(0.1, 0.4, 0.2, 0.05)),
        TextBlock(text="12345678", confidence=0.95, box=(80, 82, 80, 20), rel_box=(0.3, 0.41, 0.3, 0.05)),
    ]


class TestGroupBlocksIntoLines:
    def test_groups_by_vertical_center(self) -> None:
        lines = group_blocks_into_lines(sample_blocks())

        assert lines == [["REPUBLICA", "DE COLOMBIA"], ["NUMERO", "12345678"]]

    def test_empty_input(self) -> None:
        assert group_blocks_into_lines([]) == []


class TestOCRResult:
    def test_text_and_lines(self) -> None:
        result = OCRResult(engine="stub", blocks=sample_blocks())

        assert "REPUBLICA" in result.text
        assert result.mean_confidence == pytest.approx((0.9 + 0.8 + 0.7 + 0.95) / 4)
        assert result.lines[0] == "REPUBLICA DE COLOMBIA"

    def test_zero_confidence_when_empty(self) -> None:
        assert OCRResult(engine="stub", blocks=[]).mean_confidence == 0.0


class TestBuildEngineFactory:
    def _settings(self, **overrides: object) -> Settings:
        base = get_settings()
        return dataclasses.replace(base, **overrides)  # type: ignore[arg-type]

    def test_explicit_rapidocr(self) -> None:
        engine = _build_engine(self._settings(ocr_engine="rapidocr"))
        assert isinstance(engine, RapidOCREngine)
        assert engine.name == "rapidocr"
        assert engine.supports_recognition is True

    def test_explicit_easyocr(self) -> None:
        engine = _build_engine(self._settings(ocr_engine="easyocr"))
        assert isinstance(engine, EasyOCREngine)

    def test_explicit_heuristic(self) -> None:
        engine = _build_engine(self._settings(ocr_engine="heuristic"))
        assert isinstance(engine, HeuristicOCREngine)
        assert engine.supports_recognition is False

    def test_auto_prefers_rapidocr_when_installed(self) -> None:
        if not rapidocr_is_available():
            pytest.skip("rapidocr no instalado")
        engine = _build_engine(self._settings(ocr_engine="auto"))
        assert isinstance(engine, RapidOCREngine)

    def test_auto_falls_back_to_easyocr_if_only_easyocr(self) -> None:
        if not easyocr_is_available():
            pytest.skip("easyocr no instalado")
        # Se fuerza un entorno sin rapidocr simulando find_spec = None.
        import app.services.ocr_engine as ocr_module

        original = ocr_module.rapidocr_is_available
        ocr_module.rapidocr_is_available = lambda: False  # type: ignore[assignment]
        try:
            engine = _build_engine(self._settings(ocr_engine="auto"))
            assert isinstance(engine, EasyOCREngine)
        finally:
            ocr_module.rapidocr_is_available = original  # type: ignore[assignment]

    def test_auto_falls_back_to_heuristic_if_nothing_installed(self) -> None:
        import app.services.ocr_engine as ocr_module

        original_rapid = ocr_module.rapidocr_is_available
        original_easy = ocr_module.easyocr_is_available
        ocr_module.rapidocr_is_available = lambda: False  # type: ignore[assignment]
        ocr_module.easyocr_is_available = lambda: False  # type: ignore[assignment]
        try:
            engine = _build_engine(self._settings(ocr_engine="auto"))
            assert isinstance(engine, HeuristicOCREngine)
        finally:
            ocr_module.rapidocr_is_available = original_rapid  # type: ignore[assignment]
            ocr_module.easyocr_is_available = original_easy  # type: ignore[assignment]

    def test_heuristic_engine_returns_no_text(self) -> None:
        engine = HeuristicOCREngine()
        assert engine.read(blank_image()) == []

    def test_rapidocr_defaults_use_settings(self) -> None:
        engine = _build_engine(self._settings(ocr_engine="rapidocr", ocr_min_confidence=0.42, max_image_dimension=900))
        assert engine.min_confidence == 0.42
        assert engine.max_dimension == 900


class TestRapidOCREngineUnit:
    """Unidades del motor RapidOCR sin cargar los pesos ONNX reales."""

    def test_not_loaded_until_warmup(self) -> None:
        engine = RapidOCREngine()
        assert engine.loaded is False

    def test_warmup_loads_engine(self) -> None:
        if not rapidocr_is_available():
            pytest.skip("rapidocr no instalado")
        engine = RapidOCREngine()
        engine.warmup()
        assert engine.loaded is True
        assert engine._engine is not None

    def test_read_empty_when_output_is_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        engine = RapidOCREngine()
        engine._engine = object()  # ya cargado, sin inferencia real

        class _NoneResult:
            def __call__(self, *args: object, **kwargs: object) -> None:
                return None

        monkeypatch.setattr(engine, "_engine", _NoneResult(), raising=False)
        # read llama engine(prepared): con un callable None no hay boxes.
        assert engine.read(blank_image()) == []

    def test_read_normalizes_boxes_and_scores(self, monkeypatch: pytest.MonkeyPatch) -> None:
        engine = RapidOCREngine(min_confidence=0.5)

        class _FakeOut:
            boxes = [
                np.array([[10, 10], [200, 10], [200, 50], [10, 50]], dtype=np.float32),
                np.array([[10, 100], [90, 100], [90, 130], [10, 130]], dtype=np.float32),
            ]
            txts = ["HOLA MUNDO", "baja"]
            scores = [0.95, 0.4]

        class _FakeEngine:
            def __call__(self, image: np.ndarray) -> _FakeOut:
                return _FakeOut()

        monkeypatch.setattr(engine, "_engine", _FakeEngine(), raising=False)
        blocks = engine.read(blank_image(400, 600))

        # La de confianza 0.4 se filtra por min_confidence=0.5.
        assert len(blocks) == 1
        assert blocks[0].text == "HOLA MUNDO"
        assert blocks[0].confidence == pytest.approx(0.95)
        assert blocks[0].box[0] == 10
        assert 0.0 <= blocks[0].rel_box[0] <= 1.0

    def test_read_respects_max_dimension_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        engine = RapidOCREngine(max_dimension=1600)
        seen: list[tuple[int, ...]] = []

        class _FakeOut:
            boxes = None
            txts = ()
            scores = ()

        class _FakeEngine:
            def __call__(self, image: np.ndarray) -> _FakeOut:
                seen.append(image.shape[:2])
                return _FakeOut()

        monkeypatch.setattr(engine, "_engine", _FakeEngine(), raising=False)
        big = np.full((2000, 3000, 3), 255, dtype=np.uint8)
        engine.read(big, max_dimension=500)

        assert seen, "el motor no se invoco"
        height, width = seen[0]
        assert max(height, width) <= 500


class TestGetCachedEngine:
    def test_get_ocr_engine_is_cached(self) -> None:
        reset_ocr_engine()
        try:
            first = get_ocr_engine()
            second = get_ocr_engine()
            assert first is second
        finally:
            reset_ocr_engine()

    def test_warmup_does_not_raise_without_model(self) -> None:
        reset_ocr_engine()
        try:
            # No debe propagar: si falla la carga se loguea y sigue.
            warmup_ocr_engine()
        finally:
            reset_ocr_engine()


@pytest.mark.slow
class TestRealRapidOCROnSyntheticDocument:
    """Inferencia real con RapidOCR sobre el documento sintetico del fixture."""

    @pytest.fixture(scope="class")
    def blocks(self) -> list[TextBlock]:
        if not rapidocr_is_available():
            pytest.skip("rapidocr no instalado")
        engine = RapidOCREngine(min_confidence=0.25, max_dimension=1280)
        engine.warmup()
        image = np.array(__import__("cv2").imdecode(np.frombuffer(document_bytes(), dtype=np.uint8), 1))
        return engine.read(image)

    def test_recognizes_text(self, blocks: list[TextBlock]) -> None:
        assert blocks, "RapidOCR no detecto texto"
        joined = " ".join(block.text for block in blocks)
        assert "COLOMBIA" in joined.upper() or "12345678" in joined

    def test_blocks_have_valid_geometry(self, blocks: list[TextBlock]) -> None:
        for block in blocks:
            x, y, w, h = block.box
            assert w > 0 and h > 0
            assert 0.0 <= block.rel_box[0] <= 1.0
            assert 0.0 <= block.rel_box[1] <= 1.0
            assert 0.0 < block.confidence <= 1.0
