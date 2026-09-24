# ===========================================================================
#  Backend de vision con RapidOCR (ONNX) + OpenCV. Sin PyTorch.
#
#  Motor ligero (~150 MB de dependencias, ~300-400 MB de RAM) apto para
#  contenedores gratuitos (Render free: 512 MB / 0.1 CPU).
#
#      docker build -t escaneo-backend .
#      docker run --rm -p 8001:8000 --env-file .env escaneo-backend
#
#  Despues apunta FASTAPI_BASE_URL del frontend (Vercel) a la URL publica
#  de este contenedor.
# ===========================================================================
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8000 \
    OCR_ENGINE=rapidocr \
    OCR_LANGUAGES=es,en \
    OCR_USE_GPU=false \
    MAX_IMAGE_DIMENSION=1280 \
    OCR_CANVAS_SIZE=0 \
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1

WORKDIR /srv

# Bibliotecas del sistema que necesitan OpenCV y ONNX Runtime.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libgomp1 \
        libglib2.0-0 \
        libgl1 \
        curl \
    && rm -rf /var/lib/apt/lists/*

# Dependencias antes que el codigo para aprovechar la cache.
COPY backend/requirements-rapidocr.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY backend/ ./

# Pre-carga los pesos ONNX dentro de la imagen (sin red en runtime) y
# verifica que el motor arranca. Los modelos van en site-packages/rapidocr/models.
RUN python -c "from app.services.ocr_engine import RapidOCREngine; RapidOCREngine().warmup(); print('RapidOCR OK')"

# Usuario sin privilegios.
RUN useradd --create-home --shell /bin/bash appuser \
    && chown -R appuser:appuser /srv
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -fsS "http://127.0.0.1:${PORT}/api/v1/health" || exit 1

# Un solo worker: el modelo se carga en memoria y cada proceso lo duplica.
# En Render free (0.1 CPU) un worker es obligatorio.
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT} --workers 1"]
