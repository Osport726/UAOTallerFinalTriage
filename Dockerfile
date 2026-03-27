FROM python:3.12-slim

# Instalar uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# Copiar primero solo archivos de dependencias para aprovechar cache
COPY pyproject.toml uv.lock ./

# Instalar dependencias de producción
RUN uv sync --locked --no-dev

# Copiar luego solo lo necesario de la app
COPY src ./src
COPY proto ./proto
COPY models ./models
COPY main.py ./

# Variables de entorno
ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONUNBUFFERED=1
ENV TRANSFORMERS_OFFLINE=1
ENV HF_DATASETS_OFFLINE=1

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]