"""Prueba de humo de extremo a extremo contra los servidores en ejecucion.

Levanta ambos servicios, genera un documento ficticio y recorre el flujo
completo que haria un usuario real: captura, lectura del documento y
validacion de la MRZ.

    # Terminal 1
    cd backend && uvicorn app.main:app --port 8001

    # Terminal 2
    cd frontend && python manage.py runserver 8000

    # Terminal 3
    python scripts/smoke_test.py

Salida: un informe con el resultado de cada comprobacion. Codigo de salida 0
si todo pasa, 1 si algo falla (util para encadenarlo en CI).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

import requests

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BACKEND_DIR = PROJECT_ROOT / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from tests.fixtures import build_td3_mrz, document_bytes  # noqa: E402

DEFAULT_FASTAPI = os.getenv("FASTAPI_BASE_URL", "http://127.0.0.1:8001")
DEFAULT_DJANGO = os.getenv("DJANGO_BASE_URL", "http://127.0.0.1:8000")

GREEN, RED, YELLOW, BLUE, DIM, RESET = "\033[92m", "\033[91m", "\033[93m", "\033[94m", "\033[2m", "\033[0m"

results: list[tuple[str, bool, str]] = []
notes: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> bool:
    results.append((name, condition, detail))
    icon = f"{GREEN}PASA{RESET}" if condition else f"{RED}FALLA{RESET}"
    print(f"  [{icon}] {name}" + (f"  {DIM}{detail}{RESET}" if detail else ""))
    return condition


def note(text: str) -> None:
    """Comprobacion informativa: no cuenta como fallo del sistema."""
    notes.append(text)
    print(f"  [{BLUE}NOTA{RESET}] {text}")


def section(title: str) -> None:
    print(f"\n{YELLOW}{title}{RESET}")


def extract_csrf(html: str, cookies: dict[str, str]) -> str:
    match = re.search(r'name="csrfmiddlewaretoken" value="([^"]+)"', html)
    if match:
        return match.group(1)
    return cookies.get("csrftoken", "")


def run(
    fastapi_url: str,
    django_url: str,
    image_path: str | None = None,
) -> int:
    real_photo = bool(image_path)

    if real_photo:
        path = Path(image_path)  # type: ignore[arg-type]
        if not path.is_file():
            print(f"{RED}No existe la imagen indicada: {path}{RESET}")
            return 1
        document = path.read_bytes()
        filename = path.name
        print(f"{DIM}  Documento real: {path}{RESET}")
    else:
        document = document_bytes()
        filename = "documento-sintetico.jpg"
        print(
            f"{DIM}  Documento sintetico generado con OpenCV (usa --image para "
            f"probar con una foto real){RESET}"
        )

    lines = list(build_td3_mrz())

    # ------------------------------------------------------------------ API
    section("1. Backend FastAPI")
    api = requests.Session()

    try:
        response = api.get(f"{fastapi_url}/api/v1/health", timeout=10)
        check("GET /api/v1/health responde 200", response.status_code == 200, f"status={response.status_code}")
    except requests.RequestException as exc:
        check("GET /api/v1/health responde 200", False, str(exc)[:90])
        print(f"\n{RED}El backend no responde en {fastapi_url}. Abortando.{RESET}")
        return 1

    response = api.get(f"{fastapi_url}/openapi.json", timeout=30)
    paths = sorted(response.json().get("paths", {}))
    check("Swagger publica los endpoints del taller", len(paths) == 3, ", ".join(paths))

    print(f"{DIM}     Procesando el documento con el modelo (puede tardar)...{RESET}")
    response = api.post(
        f"{fastapi_url}/api/v1/scan-document",
        files={"file": (filename, document, "image/jpeg")},
        params={"include_preview": "true"},
        timeout=180,
    )
    scan: dict[str, Any] = {}
    if not check("POST /scan-document responde 200", response.status_code == 200, f"status={response.status_code}"):
        print(f"{DIM}     Respuesta: {response.text[:300]}{RESET}")
    else:
        scan = response.json()
        if real_photo:
            check("Extrae el numero de documento", bool(scan.get("document_number")), str(scan.get("document_number")))
            check("Extrae el nombre del titular", bool(scan.get("name")), str(scan.get("name")))
            check("Valida la foto del documento", scan.get("valid_photo") is True)
        else:
            check("Extrae el numero de documento", scan.get("document_number") == "12345678", str(scan.get("document_number")))
            check("Extrae el nombre del titular", scan.get("name") == "JUAN PEREZ", str(scan.get("name")))
            note(
                "La validacion de la foto no se comprueba con el documento sintetico: "
                "los clasificadores de Haar se entrenan con fotografias reales. "
                "Usa --image ruta/a/tu/documento.jpg para verificarla."
            )

        check("Ejecuta el analisis del rostro", bool(scan["face"].get("method")), str(scan["face"].get("method")))
        check("Detecta los digitos de control de la MRZ", scan["mrz"]["valid"] is True, json.dumps(scan["mrz"]["checks"]))
        check("El motor de OCR es el modelo preentrenado", scan.get("engine") == "easyocr", str(scan.get("engine")))
        check("Genera la vista previa del recorte", bool(scan.get("preview_base64")))
        check("Devuelve la calidad de la imagen", scan["quality"]["score"] > 0, f"score={scan['quality']['score']}")
        print(f"{DIM}     Tiempo de procesamiento: {scan.get('processing_ms'):.0f} ms{RESET}")

    response = api.post(
        f"{fastapi_url}/api/v1/validate-mrz",
        json={"mrz_lines": lines},
        timeout=30,
    )
    check("POST /validate-mrz acepta una MRZ valida", response.json().get("valid") is True)

    tampered = [lines[0], lines[1].replace("12345678<", "12345679<")]
    response = api.post(
        f"{fastapi_url}/api/v1/validate-mrz",
        json={"mrz_lines": tampered},
        timeout=30,
    )
    body = response.json()
    check(
        "POST /validate-mrz detecta un documento alterado",
        body.get("valid") is False and body.get("inconsistent_fields"),
        f"campos={body.get('inconsistent_fields')}",
    )

    # -------------------------------------------------------------- Django
    section("2. Cliente Django")
    web = requests.Session()

    response = web.get(f"{django_url}/", timeout=30)
    check("GET / renderiza la pantalla de escaneo", response.status_code == 200 and "camera-video" in response.text)

    csrf = extract_csrf(response.text, web.cookies.get_dict())

    print(f"{DIM}     Enviando el documento a traves del frontend...{RESET}")
    response = web.post(
        f"{django_url}/api/escanear/",
        files={"file": (filename, document, "image/jpeg")},
        data={"include_preview": "true"},
        headers={"X-CSRFToken": csrf, "Referer": f"{django_url}/"},
        timeout=180,
    )
    if not check("POST /api/escanear/ responde 200", response.status_code == 200, f"status={response.status_code}"):
        print(f"{DIM}     Respuesta: {response.text[:300]}{RESET}")
    else:
        proxy = response.json()
        expected_number = scan.get("document_number") if real_photo else "12345678"
        check("El frontend devuelve el numero de documento", proxy.get("document_number") == expected_number)
        check("El frontend devuelve el nombre", bool(proxy.get("name")))
        check("El frontend reenvia el analisis del rostro", "face" in proxy and "mrz" in proxy)

    response = web.post(f"{django_url}/api/escanear/", data={}, headers={"X-CSRFToken": csrf}, timeout=30)
    check("Una subida sin archivo se rechaza con 400", response.status_code == 400, f"status={response.status_code}")

    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Prueba de humo del sistema completo.")
    parser.add_argument("--fastapi", default=DEFAULT_FASTAPI, help="URL base del backend FastAPI.")
    parser.add_argument("--django", default=DEFAULT_DJANGO, help="URL base del cliente Django.")
    parser.add_argument(
        "--image",
        default=None,
        help=(
            "Foto real de un documento de identificacion. Si se omite se genera un "
            "documento sintetico, con el que la validacion de la fotografia no puede "
            "comprobarse (los clasificadores de Haar necesitan fotografias reales)."
        ),
    )
    args = parser.parse_args()

    print(f"{YELLOW}Prueba de humo del Lector e Inspector de Documentos{RESET}")
    print(f"{DIM}  FastAPI: {args.fastapi}{RESET}")
    print(f"{DIM}  Django : {args.django}{RESET}")

    exit_code = run(
        args.fastapi.rstrip("/"),
        args.django.rstrip("/"),
        args.image,
    )

    section("Resumen")
    passed = sum(1 for _, ok, _ in results if ok)
    total = len(results)
    failed = [name for name, ok, _ in results if not ok]

    print(f"  {passed}/{total} comprobaciones correctas")
    for name in failed:
        print(f"  {RED}x{RESET} {name}")

    if notes:
        print(f"\n  {BLUE}Notas{RESET}")
        for text in notes:
            print(f"  · {text}")

    if exit_code == 0 and not failed:
        print(f"\n{GREEN}Todo funciona correctamente.{RESET}")
        return 0

    print(f"\n{RED}Hay comprobaciones fallidas.{RESET}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
