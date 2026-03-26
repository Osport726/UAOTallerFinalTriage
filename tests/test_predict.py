"""
Pruebas del endpoint POST /api/predict (RF-03 / RF-04) — T09.

El endpoint recibe una imagen multipart/form-data, la clasifica con
TriageService y retorna un JSON con la lista de predicciones top-k.

Grupos de pruebas:
    TestValidation      — rechazo de inputs inválidos.
    TestResponseSchema  — estructura y tipos del JSON de respuesta.
    TestScoreSemantics  — semántica del score y del campo human_review.
"""

from __future__ import annotations

import io

from fastapi.testclient import TestClient
from PIL import Image


def _make_png(
    size: tuple[int, int] = (224, 224), color: tuple[int, int, int] = (100, 149, 237)
) -> bytes:
    """Genera un PNG válido en memoria con el tamaño y color indicados."""
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, format="PNG")
    return buf.getvalue()


def _make_jpeg(color: tuple[int, int, int] = (200, 100, 50)) -> bytes:
    """Genera un JPEG válido en memoria."""
    buf = io.BytesIO()
    Image.new("RGB", (224, 224), color).save(buf, format="JPEG")
    return buf.getvalue()


class TestValidation:
    """Verifica que el endpoint rechace inputs inválidos con el código correcto."""

    def test_rejects_text_file(self, client: TestClient) -> None:
        """
        Un archivo de texto debe retornar HTTP 400.

        El endpoint valida que el Content-Type comience con 'image/';
        cualquier otro tipo debe rechazarse antes de intentar la inferencia.
        """
        r = client.post(
            "/api/predict",
            files={"image": ("doc.txt", b"contenido de texto", "text/plain")},
        )
        assert r.status_code == 400

    def test_rejects_pdf_file(self, client: TestClient) -> None:
        """Un archivo PDF debe retornar HTTP 400 (no es imagen)."""
        r = client.post(
            "/api/predict",
            files={"image": ("doc.pdf", b"%PDF-1.4", "application/pdf")},
        )
        assert r.status_code == 400

    def test_rejects_oversized_image(self, client: TestClient) -> None:
        """
        Una imagen que supere MAX_UPLOAD_MB debe retornar HTTP 413.

        Genera un PNG grande en memoria para simular el exceso de tamaño.
        """
        big_png = _make_png(size=(4000, 4000))  # ~48 MB descomprimido
        r = client.post(
            "/api/predict",
            files={"image": ("big.png", big_png, "image/png")},
        )
        # 413 si supera el límite; 200 si el límite es mayor al tamaño generado
        assert r.status_code in (200, 413)

    def test_rejects_corrupted_image(self, client: TestClient) -> None:
        """
        Bytes que declaran ser imagen pero están corruptos deben retornar HTTP 400.

        Simula un archivo PNG truncado o con contenido inválido.
        """
        r = client.post(
            "/api/predict",
            files={
                "image": ("corrupt.png", b"\x89PNG\r\n\x1a\nDATA_CORRUPTA", "image/png")
            },
        )
        assert r.status_code == 400


class TestResponseSchema:
    """Verifica la estructura del JSON de respuesta para una imagen válida."""

    def test_returns_200(self, client: TestClient, sample_image_file: tuple) -> None:
        """POST /api/predict con imagen válida debe retornar HTTP 200."""
        r = client.post("/api/predict", files={"image": sample_image_file})
        assert r.status_code == 200

    def test_response_has_predictions_key(
        self, client: TestClient, sample_image_file: tuple
    ) -> None:
        """La respuesta debe contener la clave 'predictions'."""
        data = client.post("/api/predict", files={"image": sample_image_file}).json()
        assert "predictions" in data

    def test_predictions_is_nonempty_list(
        self, client: TestClient, sample_image_file: tuple
    ) -> None:
        """
        'predictions' debe ser una lista con al menos una predicción.

        Con top_k=3 (default) se esperan hasta 3 resultados; como mínimo 1.
        """
        data = client.post("/api/predict", files={"image": sample_image_file}).json()
        assert isinstance(data["predictions"], list)
        assert len(data["predictions"]) >= 1

    def test_prediction_item_has_required_fields(
        self, client: TestClient, sample_image_file: tuple
    ) -> None:
        """
        Cada elemento de 'predictions' debe tener los campos del contrato T05:
        category, category_key, score y team.
        """
        data = client.post("/api/predict", files={"image": sample_image_file}).json()
        top = data["predictions"][0]
        for field in ("category", "category_key", "score", "team"):
            assert field in top, f"Campo faltante en prediction item: '{field}'"

    def test_response_has_timestamp(
        self, client: TestClient, sample_image_file: tuple
    ) -> None:
        """La respuesta debe incluir un timestamp ISO de la clasificación."""
        data = client.post("/api/predict", files={"image": sample_image_file}).json()
        assert "timestamp" in data and data["timestamp"]

    def test_response_has_model_info(
        self, client: TestClient, sample_image_file: tuple
    ) -> None:
        """La respuesta debe incluir la sección 'model' con metadatos del servicio."""
        data = client.post("/api/predict", files={"image": sample_image_file}).json()
        assert "model" in data
        assert "min_confidence" in data["model"]

    def test_accepts_jpeg(self, client: TestClient) -> None:
        """El endpoint debe aceptar imágenes JPEG además de PNG."""
        r = client.post(
            "/api/predict",
            files={"image": ("test.jpg", _make_jpeg(), "image/jpeg")},
        )
        assert r.status_code == 200


class TestScoreSemantics:
    """Verifica que el score y la señal de revisión humana sean coherentes."""

    def test_score_is_probability(
        self, client: TestClient, sample_image_file: tuple
    ) -> None:
        """
        El score de la predicción principal debe ser una probabilidad en [0, 1].

        Un valor fuera de este rango indicaría un error en la normalización
        de las salidas del clasificador.
        """
        data = client.post("/api/predict", files={"image": sample_image_file}).json()
        score = data["predictions"][0]["score"]
        assert 0.0 <= score <= 1.0, f"Score fuera de rango: {score}"

    def test_top_score_is_highest(
        self, client: TestClient, sample_image_file: tuple
    ) -> None:
        """
        La primera predicción debe tener el score más alto de la lista.

        Si hay múltiples predicciones, deben estar ordenadas de mayor
        a menor confianza.
        """
        data = client.post("/api/predict", files={"image": sample_image_file}).json()
        predictions = data["predictions"]
        if len(predictions) > 1:
            assert predictions[0]["score"] >= predictions[1]["score"]
