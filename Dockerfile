# python:3.12-slim tiene soporte multi-arch (amd64 + arm64).
# Docker Desktop selecciona automáticamente la arquitectura correcta,
# por lo que el mismo Dockerfile funciona en Mac (Intel/Apple Silicon) y Windows.
FROM python:3.12-slim

ARG MODEL_PATH=models/classifier.pkl

# Instalar uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Copiar código fuente y artefactos
# .dockerignore excluye: .venv, data/, embeddings_cache.npz,
# mlruns/, tests/, scripts/, .git, IDEs, *.md
COPY . /app
ADD ${MODEL_PATH} /app/models/classifier.pkl
WORKDIR /app

# Instalar dependencias de producción
RUN uv sync --locked --no-dev

# Variables de entorno
ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONUNBUFFERED=1
# Evita que HuggingFace intente descargar el modelo si ya está en models/hf_cache
ENV TRANSFORMERS_OFFLINE=1
ENV HF_DATASETS_OFFLINE=1

# Puerto de la aplicación
EXPOSE 8000

# Healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

# Comando de inicio
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
