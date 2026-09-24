"""Benchmark local: EasyOCR vs RapidOCR vs heuristico.

Compara tiempo de carga, inferencia, memoria RSS y campos extraidos sobre
las fotos reales del proyecto y sobre el documento sintetico de los tests.

Uso:
    python scripts/benchmark_ocr.py
"""

from __future__ import annotations

import gc
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
os.chdir(ROOT / "backend")

import numpy as np  # noqa: E402


def rss_mb() -> float:
    """MB de RSS del proceso actual (Windows o POSIX)."""
    try:
        import ctypes
        from ctypes import wintypes

        class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
            _fields_ = [
                ("cb", wintypes.DWORD),
                ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        k32 = ctypes.windll.kernel32
        psapi = ctypes.windll.psapi
        psapi.GetProcessMemoryInfo.argtypes = [
            wintypes.HANDLE,
            ctypes.POINTER(PROCESS_MEMORY_COUNTERS),
            wintypes.DWORD,
        ]
        psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
        k32.GetCurrentProcess.argtypes = []
        k32.GetCurrentProcess.restype = wintypes.HANDLE

        counters = PROCESS_MEMORY_COUNTERS()
        counters.cb = ctypes.sizeof(counters)
        if psapi.GetProcessMemoryInfo(k32.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
            # PeakWorkingSetSize = pico historico; WorkingSetSize = actual.
            return counters.PeakWorkingSetSize / (1024 * 1024)
    except Exception:
        pass
    try:
        import resource

        value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return value / 1024 if value > 10_000 else value / 1024
    except Exception:
        return -1.0


def load_image(path: Path):
    import cv2

    data = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"No se pudo leer {path}")
    return image


def run_engine(name: str, image_paths: list[Path]) -> dict:
    from app.services.ocr_engine import (
        EasyOCREngine,
        HeuristicOCREngine,
        RapidOCREngine,
        reset_ocr_engine,
    )
    from app.services.scanner import DocumentScanner
    from app.services.face_detector import HeuristicFaceDetector

    reset_ocr_engine()
    gc.collect()

    rss_before = rss_mb()
    t0 = time.perf_counter()

    if name == "easyocr":
        engine = EasyOCREngine(languages=("es", "en"), use_gpu=False, min_confidence=0.20)
    elif name == "rapidocr":
        engine = RapidOCREngine(min_confidence=0.20)
    else:
        engine = HeuristicOCREngine()

    engine.warmup()
    load_s = time.perf_counter() - t0
    rss_after_load = rss_mb()

    scanner = DocumentScanner(engine=engine, face_detector=HeuristicFaceDetector())

    per_image = []
    scans = []
    for path in image_paths:
        t1 = time.perf_counter()
        scan = scanner.scan(path.read_bytes())
        infer_s = time.perf_counter() - t1
        per_image.append((path.name, infer_s))
        scans.append(scan)

    rss_peak = rss_mb()

    return {
        "engine": name,
        "load_s": load_s,
        "rss_before_mb": rss_before,
        "rss_after_load_mb": rss_after_load,
        "rss_peak_mb": rss_peak,
        "per_image_s": per_image,
        "scans": scans,
    }


def summarize(result: dict) -> None:
    print("\n" + "=" * 72)
    print(f"ENGINE: {result['engine']}")
    print("=" * 72)
    print(f"  carga modelo:     {result['load_s']:.2f} s")
    print(f"  RSS antes:        {result['rss_before_mb']:.0f} MB")
    print(f"  RSS tras carga:   {result['rss_after_load_mb']:.0f} MB")
    print(f"  RSS pico:         {result['rss_peak_mb']:.0f} MB")
    for name, secs in result["per_image_s"]:
        print(f"  inferencia {name}: {secs:.2f} s")

    for scan in result["scans"]:
        print(f"\n  --- scan: {getattr(scan, 'engine', '?')} ---")
        print(f"  document_number: {scan.document_number!r}")
        print(f"  name:            {scan.name!r}")
        print(f"  valid_photo:     {scan.valid_photo}")
        print(f"  confidence:      {scan.confidence:.2f}")
        print(f"  document_type:   {scan.fields.document_type!r}")
        print(f"  birth_date:      {scan.fields.birth_date!r}")
        print(f"  expiry_date:     {scan.fields.expiry_date!r}")
        print(f"  mrz.detected:    {scan.mrz.detected}")
        lines = scan.text_lines[:12]
        print(f"  text_lines ({len(scan.text_lines)}):")
        for line in lines:
            print(f"    | {line}")


def main() -> None:
    # Adaptador RapidOCR con la misma interfaz que BaseOCREngine
    adapter = ROOT / "scripts" / "benchmark_rapidocr.py"
    if not adapter.exists():
        raise SystemExit(f"Falta {adapter}")

    images = [
        ROOT / "parte_de_adelante.jpeg",
        ROOT / "parte_de_atras.jpeg",
    ]
    images = [p for p in images if p.exists()]
    if not images:
        raise SystemExit("No hay fotos parte_de_*.jpeg en la raiz del proyecto")

    # Un motor por proceso: la memoria RSS no se contamina entre motores.
    only = sys.argv[1] if len(sys.argv) > 1 else None
    engines = [only] if only else ["heuristic", "easyocr", "rapidocr"]
    results = []
    for eng in engines:
        try:
            results.append(run_engine(eng, images))
        except Exception as exc:
            print(f"[ERROR] {eng}: {exc}", file=sys.stderr)

    for r in results:
        summarize(r)

    # Tabla resumen
    print("\n" + "=" * 72)
    print("RESUMEN")
    print("=" * 72)
    print(f"{'engine':<12} {'carga_s':>8} {'pico_MB':>8} {'infer_s':>8} {'campos':>40}")
    for r in results:
        infer_total = sum(s for _, s in r["per_image_s"])
        scan = r["scans"][0]
        campos = f"num={scan.document_number} name={scan.name!r}"
        print(
            f"{r['engine']:<12} {r['load_s']:8.2f} {r['rss_peak_mb']:8.0f} "
            f"{infer_total:8.2f} {campos:>40}"
        )


if __name__ == "__main__":
    main()
