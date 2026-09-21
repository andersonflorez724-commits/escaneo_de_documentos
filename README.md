# 🪪 Lector e Inspector de Documentos de Identificación

> **Taller 4 — Python + ML + FastAPI**
> Lectura e inspección automática de documentos de identificación mediante
> OCR con un modelo preentrenado, en lugar de la digitación manual.

Aplicación web que permite **tomar una fotografía de un documento de identidad
con la cámara del dispositivo** y extraer automáticamente los datos relevantes,
evitando errores de captura en recepción.

```
┌────────────────────────────┐        ┌──────────────────────────────┐
│  Frontend (Django)         │        │  Backend (FastAPI)           │
│  · Sesiones / login        │  HTTP  │  · JWT (OAuth2 password)     │
│  · Plantillas HTML5        │ ─────► │  · EasyOCR + OpenCV          │
│  · getUserMedia (cámara)   │        │  · Validación MRZ ICAO 9293  │
│  · Swagger en /docs        │        │  · Pydantic + /docs          │
└────────────────────────────┘        └──────────────────────────────┘
```

## Stack tecnológico

| Capa | Tecnología |
| --- | --- |
| Frontend | Django (vistas, plantillas HTML5, JS `getUserMedia`) |
| Backend | FastAPI + Pydantic (Swagger interactivo en `/docs`) |
| Modelo ML | **EasyOCR** (preentrenado) + **OpenCV** + PyTorch |
| Autenticación | Sesión Django (UI) + JWT en FastAPI (API) |
| Despliegue | Vercel (`vercel.json`, Serverless Functions Python) |

## Endpoints principales (Swagger: `/docs`)

| Método | Ruta | Descripción |
| --- | --- | --- |
| `POST` | `/api/v1/auth/login` | Obtiene el token JWT |
| `POST` | `/api/v1/scan-document` | Extrae `document_number`, `name` y `valid_photo` |
| `POST` | `/api/v1/validate-mrz` | Valida la consistencia de caracteres de la MRZ |
| `GET` | `/api/v1/health` | Estado del servicio |

## Estructura del proyecto

```
.
├── backend/                      # API FastAPI + modelo de ML
│   ├── app/
│   │   ├── api/v1/endpoints/     # health, auth, documents
│   │   ├── core/                 # configuración y seguridad
│   │   ├── schemas/              # modelos Pydantic
│   │   └── services/             # OCR, parser, MRZ, rostro
│   ├── tests/                    # pruebas con pytest
│   └── requirements.txt
├── frontend/                     # Cliente Django
│   ├── apps/accounts/            # login / logout / registro
│   ├── apps/scanner/             # captura y resultados
│   ├── config/                   # settings, urls, wsgi
│   ├── static/                   # CSS y JS (cámara, cliente API)
│   └── templates/                # plantillas HTML5
├── vercel.json
└── .env.example
```

## Puesta en marcha local

### 1. Entorno virtual e instalación

```bash
python -m venv .venv
source .venv/Scripts/activate      # Windows (Git Bash)
# source .venv/bin/activate        # Linux / macOS

pip install -r backend/requirements.txt
pip install -r frontend/requirements.txt
```

### 2. Variables de entorno

```bash
cp .env.example .env
# Edita SECRET_KEY y DJANGO_SECRET_KEY
```

### 3. Backend FastAPI (puerto 8001)

```bash
cd backend
uvicorn app.main:app --reload --port 8001
```

Swagger: <http://127.0.0.1:8001/docs>

### 4. Frontend Django (puerto 8000)

```bash
cd frontend
python manage.py migrate
python manage.py createsuperuser
python manage.py runserver 8000
```

Aplicación: <http://127.0.0.1:8000>

> **Nota sobre la cámara:** `getUserMedia` requiere un *contexto seguro*.
> Usa `http://localhost` o `http://127.0.0.1` (ambos se consideran seguros) o
> HTTPS en producción.

## Licencia

MIT
