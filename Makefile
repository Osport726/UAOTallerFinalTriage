.PHONY: help install start test lint format pre-commit-install train evaluate grpc

help:
	@echo ""
	@echo "Comandos disponibles:"
	@echo "  make install  Instala dependencias con uv"
	@echo "  make start    Levanta FastAPI en http://localhost:8000"
	@echo "  make grpc     Levanta el servidor gRPC en el puerto 50051"
	@echo "  make test     Ejecuta tests con pytest"
	@echo "  make lint     Verifica el código con ruff"
	@echo "  make format   Formatea el código con ruff"
	@echo "  make train    Entrena el clasificador y registra en MLflow"
	@echo "  make evaluate Evalúa el modelo entrenado"
	@echo ""

install:
	uv sync --group dev

start:
	uv run main.py

grpc:
	uv run python -m src.grpc.server

test:
	USE_DUMMY_MODEL=1 uv run --locked --group dev pytest

lint:
	uv run --locked --group dev ruff check src tests scripts main.py --exclude src/grpc/stubs

format:
	uv run ruff format src tests scripts main.py --exclude src/grpc/stubs

pre-commit-install:
	uv run pre-commit install

train:
	uv run python scripts/train.py

evaluate:
	uv run python scripts/evaluate.py
