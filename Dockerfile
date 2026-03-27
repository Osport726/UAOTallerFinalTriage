# python:3.12-slim tiene soporte multi-arch (amd64 + arm64).
# Docker Desktop selecciona automáticamente la arquitectura correcta,
# por lo que el mismo Dockerfile funciona en Mac (Intel/Apple Silicon) y Windows.

# =========================================================
# DOCKERFILE OPTIMIZADO PARA REDUCIR TIEMPO DE BUILD
#
# MEJORAS CLAVE:
# 1. Copia primero pyproject.toml y uv.lock
#    para cachear la capa de dependencias.
# 2. Instala dependencias ANTES de copiar todo el código.
# 3. Usa cache interno de uv durante el build.
# 4. Mantiene el modelo opcional mediante ARG MODEL_PATH.
#
# REQUISITO:
# Este Dockerfile aprovecha mejor el cache cuando usas
# Docker BuildKit / buildx, como en el .gitlab-ci.yml propuesto.
# =========================================================

FROM python:3.12-slim

# Ruta opcional del modelo a inyectar en la imagen.
# Si no envías MODEL_PATH desde CI, usa este valor por defecto.
ARG MODEL_PATH=models/classifier.pkl

# ---------------------------------------------------------
# Instalar uv desde la imagen oficial de Astral
# ---------------------------------------------------------
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# ---------------------------------------------------------
# Directorio de trabajo
# ---------------------------------------------------------
WORKDIR /app

# ---------------------------------------------------------
# Variables de entorno
# ---------------------------------------------------------
# PATH: permite ejecutar uvicorn/python desde el .venv del proyecto
ENV PATH="/app/.venv/bin:$PATH"

# Salida Python sin buffering, útil para logs de contenedor
ENV PYTHONUNBUFFERED=1

# Evita intentos de descarga desde Hugging Face en runtime
# asumiendo que ya tienes lo necesario localmente/cacheado
ENV TRANSFORMERS_OFFLINE=1
ENV HF_DATASETS_OFFLINE=1

# Hace más estable el comportamiento de uv dentro del contenedor
ENV UV_LINK_MODE=copy

# ---------------------------------------------------------
# 1) Copiar SOLO archivos que definen dependencias
# ---------------------------------------------------------
# Esta es la optimización más importante del Dockerfile:
# si solo cambias código fuente y no cambian estas dos rutas,
# Docker puede reutilizar la capa de dependencias.
COPY pyproject.toml uv.lock ./

# ---------------------------------------------------------
# 2) Instalar dependencias de producción
# ---------------------------------------------------------
# --mount=type=cache permite que uv reutilice su cache
# durante builds repetidos con BuildKit/buildx.
#
# --no-dev evita instalar dependencias de desarrollo.
# --no-install-project evita depender del código fuente aún.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --no-install-project

# ---------------------------------------------------------
# 3) Copiar solo lo necesario para runtime
# ---------------------------------------------------------
# En lugar de COPY . /app, copiamos únicamente lo que la app usa
# en producción. Eso reduce invalidaciones de capas y evita meter
# archivos innecesarios como tests, docs o scripts.
COPY src ./src
COPY main.py ./main.py
COPY proto ./proto
COPY models ./models

# Si tu aplicación usa README/licencia no hace falta copiarlos
# al runtime, así que se omiten para mantener la imagen más limpia.

# ---------------------------------------------------------
# 4) Inyectar modelo final si CI entrega MODEL_PATH
# ---------------------------------------------------------
# Si MODEL_PATH apunta a otro archivo durante CI, reemplazará
# /app/models/classifier.pkl.
ADD ${MODEL_PATH} /app/models/classifier.pkl

# ---------------------------------------------------------
# 5) Instalar el proyecto local (rápido)
# ---------------------------------------------------------
# Se ejecuta después de copiar el código.
# Como las dependencias ya quedaron resueltas antes, esta capa
# suele ser mucho más ligera que una instalación completa.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev

# ---------------------------------------------------------
# Puerto expuesto por FastAPI / Uvicorn
# ---------------------------------------------------------
EXPOSE 8000

# ---------------------------------------------------------
# Healthcheck
# ---------------------------------------------------------
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

# ---------------------------------------------------------
# Comando de inicio
# ---------------------------------------------------------
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]