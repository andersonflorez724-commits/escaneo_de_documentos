# 🪪 Lector e Inspector de Documentos de Identificación

Aplicación web que **lee e inspecciona documentos de identidad** (cédula de
ciudadanía y tarjeta de identidad) con OCR, para evitar la digitación manual
en recepción.

**Demo:** <https://escaneo-de-documentos.vercel.app/>

## Qué hace

1. Captura la **cara frontal** y el **reverso** con la cámara del dispositivo
   (o subiendo fotos).
2. El backend fusiona el texto de ambas caras y extrae los campos.
3. Valida el **código de verificación** del reverso (y MRZ ICAO 9293 cuando el
   documento la trae, p. ej. pasaportes).
4. Muestra el resultado listo para copiar, con advertencias de calidad.

| Cara | Campos principales |
| --- | --- |
| Frontal | `document_number` (NUIP), apellidos, nombres, tipo de documento, foto válida |
| Reverso | fecha y lugar de nacimiento / expedición, vencimiento, sexo, grupo sanguíneo, código de verificación |

```
┌────────────────────────────┐        ┌──────────────────────────────┐
│  Frontend (Django)         │        │  Backend (FastAPI)           │
│  · Plantillas HTML5        │  HTTP  │  · RapidOCR + OpenCV (ONNX)  │
│  · getUserMedia (cámara)   │ ─────► │  · Validación MRZ ICAO 9293  │
│  · Sin sesiones ni BD      │        │  · Pydantic + Swagger /docs  │
└────────────────────────────┘        └──────────────────────────────┘
```

## Stack

| Capa | Tecnología |
| --- | --- |
| Frontend | Django + HTML5 + JS (`getUserMedia`) |
| Backend | FastAPI + Pydantic |
| OCR / ML | RapidOCR (modelos PaddleOCR vía ONNX) + OpenCV headless |
| Despliegue | Vercel (una Serverless Function Python) |

## API

Swagger: <https://escaneo-de-documentos.vercel.app/docs>

| Método | Ruta | Descripción |
| --- | --- | --- |
| `POST` | `/api/v1/scan-document` | Extrae los campos (`file` = frontal, `back_file` = reverso) |
| `POST` | `/api/v1/validate-mrz` | Valida los dígitos de control de la MRZ |
| `GET` | `/api/v1/health` | Estado del servicio y motor OCR activo |

## Estructura

```
.
├── backend/          # API FastAPI + OCR, parser, MRZ, tests
├── frontend/         # Cliente Django (captura, resultados)
├── api/              # Entry point y requirements de Vercel
├── public/static/    # CSS/JS servidos en producción
├── vercel.json
└── Dockerfile        # Contenedor opcional con RapidOCR
```

## Desarrollo local

```bash
python -m venv .venv
source .venv/Scripts/activate   # Windows; en Linux/macOS: source .venv/bin/activate
pip install -r requirements-dev.txt
pip install -r frontend/requirements.txt
cp .env.example .env            # define DJANGO_SECRET_KEY

# Backend (puerto 8001)
cd backend && uvicorn app.main:app --reload --port 8001

# Frontend (puerto 8000, otra terminal)
cd frontend && python manage.py runserver 8000
```

La cámara solo funciona en contextos seguros: `http://localhost` o HTTPS.

### Pruebas

```bash
cd backend  && ../.venv/Scripts/python.exe -m pytest -v    # ~210 pruebas
cd frontend && ../.venv/Scripts/python.exe manage.py test apps
```

## Licencia

MIT
