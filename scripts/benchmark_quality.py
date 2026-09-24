"""Compara el texto crudo y el parser EasyOCR vs RapidOCR en las fotos reales."""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
os.chdir(ROOT / "backend")

import numpy as np  # noqa: E402

from app.services.document_parser import parse_document  # noqa: E402
from app.services.face_detector import HeuristicFaceDetector  # noqa: E402
from app.services.image_utils import decode_image  # noqa: E402
from app.services.ocr_engine import (  # noqa: E402
    EasyOCREngine,
    RapidOCREngine,
    group_blocks_into_lines,
)
from app.services.scanner import DocumentScanner  # noqa: E402


def show_engine(label: str, engine, images: list[Path]) -> None:
    print("\n" + "=" * 72)
    print(label)
    print("=" * 72)
    scanner = DocumentScanner(engine=engine, face_detector=HeuristicFaceDetector())
    engine.warmup()

    for path in images:
        print(f"\n--- {path.name} ---")
        image = decode_image(path.read_bytes())
        blocks = engine.read(image)
        lines = [" ".join(g) for g in group_blocks_into_lines(blocks)]
        print(f"bloques={len(blocks)} lineas={len(lines)}")
        for i, line in enumerate(lines):
            print(f"  {i:02d}| {line}")

        # Parser solo sobre el texto de esta cara
        fields = parse_document(lines)
        print("  FIELDS:")
        for name in (
            "document_number",
            "first_surname",
            "second_surname",
            "given_names",
            "name",
            "document_type",
            "birth_date",
            "birth_place",
            "issue_date",
            "expiry_date",
            "sex",
            "blood_type",
        ):
            value = getattr(fields, name, None)
            if value:
                print(f"    {name}={value!r}")

        scan = scanner.scan(path.read_bytes())
        print("  SCAN:")
        print(f"    num={scan.document_number!r} name={scan.name!r}")
        print(f"    type={scan.fields.document_type!r}")
        print(f"    birth={scan.fields.birth_date!r} expiry={scan.fields.expiry_date!r}")
        print(f"    issue={scan.fields.issue_date!r} conf={scan.confidence:.2f}")


def main() -> None:
    images = [ROOT / "parte_de_adelante.jpeg", ROOT / "parte_de_atras.jpeg"]
    images = [p for p in images if p.exists()]

    show_engine(
        "EASYOCR",
        EasyOCREngine(languages=("es", "en"), use_gpu=False, min_confidence=0.20),
        images,
    )
    show_engine("RAPIDOCR", RapidOCREngine(min_confidence=0.20), images)

    # Combinado: las dos caras en un solo scan como hace la API
    print("\n" + "=" * 72)
    print("SCAN COMBINADO (front + back via scan_sides)")
    print("=" * 72)
    front = ROOT / "parte_de_adelante.jpeg"
    back = ROOT / "parte_de_atras.jpeg"
    if front.exists() and back.exists():
        for label, eng in (
            ("easyocr", EasyOCREngine(languages=("es", "en"), use_gpu=False, min_confidence=0.20)),
            ("rapidocr", RapidOCREngine(min_confidence=0.20)),
        ):
            eng.warmup()
            scanner = DocumentScanner(engine=eng, face_detector=HeuristicFaceDetector())
            result = scanner.scan_sides(
                [("front", front.read_bytes()), ("back", back.read_bytes())]
            )
            print(f"\n[{label}]")
            print(f"  engine={result.engine}")
            print(f"  num={result.document_number!r} name={result.name!r}")
            print(f"  type={result.fields.document_type!r}")
            print(f"  birth={result.fields.birth_date!r}")
            print(f"  expiry={result.fields.expiry_date!r}")
            print(f"  issue={result.fields.issue_date!r}")
            print(f"  sex={result.fields.sex!r} blood={result.fields.blood_type!r}")
            print(f"  conf={result.confidence:.2f} photo={result.valid_photo}")
            print(f"  sources={result.fields.sources}")
            print(f"  warnings={result.warnings}")


if __name__ == "__main__":
    main()
