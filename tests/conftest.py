"""
Fixtures compartidas para toda la suite de pruebas.

Establece USE_DUMMY_MODEL=1 antes de importar la aplicación para que
TriageService opere en modo sintético, evitando la descarga del modelo
DeiT-Tiny y la necesidad de `models/classifier.pkl` durante los tests.

Fixtures disponibles:
    client            : TestClient de FastAPI (scope=session).
    sample_image_bytes: PNG mínimo válido generado en memoria.
    sample_image_file : Tupla lista para usar en `files=` de TestClient.
"""

from __future__ import annotations

import io
import os

import pytest
from fastapi.testclient import TestClient
from PIL import Image

os.environ["USE_DUMMY_MODEL"] = "1"

from src.api.main import app  # noqa: E402


@pytest.fixture(scope="session")
def client() -> TestClient:
    """
    Cliente HTTP de pruebas para la aplicación FastAPI.

    Usa scope='session' para que el servidor (y el modelo dummy) se inicialice
    una sola vez por sesión de pytest, reduciendo el tiempo total de ejecución.

    Yields:
        TestClient listo para hacer peticiones a todos los endpoints.
    """
    return TestClient(app)


@pytest.fixture
def sample_image_bytes() -> bytes:
    """
    Imagen PNG mínima válida generada completamente en memoria.

    Genera un PNG RGB de 224×224 px sin necesidad de archivos en disco.
    Útil como payload directo cuando solo se necesitan los bytes.

    Returns:
        Bytes de un PNG válido de 224×224 px color azul acero.
    """
    img = Image.new("RGB", (224, 224), color=(100, 149, 237))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture
def sample_image_file(sample_image_bytes: bytes) -> tuple[str, bytes, str]:
    """
    Tupla lista para usar como valor en el dict `files=` de TestClient.

    Returns:
        Tupla (filename, bytes, content_type) compatible con requests/httpx.

    Example:
        def test_algo(client, sample_image_file):
            r = client.post("/api/predict", files={"image": sample_image_file})
    """
    return ("sample.png", sample_image_bytes, "image/png")
