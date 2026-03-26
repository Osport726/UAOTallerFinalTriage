"""
Pruebas del endpoint GET /health (RF-02).

Verifica que el servicio responda correctamente al healthcheck,
lo que es requisito del pipeline CI y del despliegue en producción.
"""

from __future__ import annotations

from fastapi.testclient import TestClient


def test_health_returns_ok(client: TestClient) -> None:
    """
    GET /health debe retornar HTTP 200 con body {"status": "ok"}.

    Es el check mínimo que usa el pipeline CI para verificar que la
    aplicación arrancó correctamente antes de correr el resto de pruebas.
    """
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_health_content_type_is_json(client: TestClient) -> None:
    """
    GET /health debe retornar Content-Type application/json.

    Garantiza que los clientes y load balancers que inspeccionan el tipo
    de contenido del healthcheck puedan parsearlo correctamente.
    """
    response = client.get("/health")

    assert "application/json" in response.headers.get("content-type", "")
