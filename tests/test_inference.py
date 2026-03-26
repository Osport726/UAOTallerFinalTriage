"""
Pruebas unitarias de la capa de inferencia — T09.

Cubre TriageService en modo dummy (USE_DUMMY_MODEL=1), que es el único
modo disponible en CI sin necesidad de descargar pesos ni entrenar el modelo.

Grupos de pruebas:
    TestPredictionOutput   — estructura y tipos de la respuesta.
    TestConfidenceGate     — regla de seguridad: score bajo → "otros".
    TestServiceInfo        — metadatos que expone el servicio.
    TestImageVariants      — distintos tamaños y modos de imagen.
"""

from __future__ import annotations

import os

import pytest
from PIL import Image
from src.config import AppConfig
from src.domain.categories import CATEGORY_KEYS

os.environ["USE_DUMMY_MODEL"] = "1"

from src.inference.service import TriageService  # noqa: E402


@pytest.fixture(scope="module")
def service() -> TriageService:
    """
    Instancia de TriageService en modo dummy compartida por todo el módulo.

    Usa scope='module' para inicializar el servicio una sola vez y reutilizarlo
    en todas las pruebas del archivo, evitando overhead de inicialización.
    """
    svc = TriageService()
    svc.load()
    return svc


@pytest.fixture
def rgb_image() -> Image.Image:
    """Imagen RGB mínima válida de 224×224 px para pruebas de inferencia."""
    return Image.new("RGB", (224, 224), color=(255, 0, 0))


class TestPredictionOutput:
    """Verifica la estructura y tipos de datos de la respuesta de predict()."""

    def test_returns_at_least_one_prediction(
        self, service: TriageService, rgb_image: Image.Image
    ) -> None:
        """
        predict() debe retornar al menos una predicción.

        Con top_k >= 1 (default 3), la lista nunca debe estar vacía.
        """
        predictions = service.predict(rgb_image)
        assert len(predictions) >= 1

    def test_category_key_belongs_to_catalog(
        self, service: TriageService, rgb_image: Image.Image
    ) -> None:
        """
        La clave de categoría de cada predicción debe pertenecer al catálogo.

        Garantiza que el modelo no retorna categorías inventadas o fuera
        del dominio definido en src/domain/categories.py.
        """
        predictions = service.predict(rgb_image)
        for pred in predictions:
            assert pred.category_key in CATEGORY_KEYS, (
                f"'{pred.category_key}' no es una clave válida del catálogo."
            )

    def test_score_is_probability(
        self, service: TriageService, rgb_image: Image.Image
    ) -> None:
        """
        El score de cada predicción debe ser una probabilidad en [0.0, 1.0].

        Un score fuera de este rango indicaría un error en predict_proba()
        o en la normalización de las salidas del modelo.
        """
        predictions = service.predict(rgb_image)
        for pred in predictions:
            assert 0.0 <= pred.score <= 1.0, f"Score fuera de rango: {pred.score}"

    def test_team_is_nonempty_string(
        self, service: TriageService, rgb_image: Image.Image
    ) -> None:
        """Cada predicción debe tener un equipo destinatario no vacío."""
        predictions = service.predict(rgb_image)
        for pred in predictions:
            assert isinstance(pred.team, str) and pred.team.strip()

    def test_category_label_is_nonempty_string(
        self, service: TriageService, rgb_image: Image.Image
    ) -> None:
        """Cada predicción debe tener una etiqueta de categoría no vacía."""
        predictions = service.predict(rgb_image)
        for pred in predictions:
            assert isinstance(pred.category_label, str) and pred.category_label.strip()

    def test_predictions_ordered_by_score_desc(
        self, service: TriageService, rgb_image: Image.Image
    ) -> None:
        """
        Las predicciones deben estar ordenadas de mayor a menor score.

        La predicción en índice 0 siempre es la más probable; las siguientes
        son alternativas en orden descendente de confianza.
        """
        predictions = service.predict(rgb_image)
        scores = [p.score for p in predictions]
        assert scores == sorted(scores, reverse=True), (
            "Las predicciones no están ordenadas por score descendente."
        )


class TestConfidenceGate:
    """Verifica la regla: si score < min_confidence entonces categoría 'otros'."""

    def test_dummy_top_prediction_score_above_zero(
        self, service: TriageService, rgb_image: Image.Image
    ) -> None:
        """
        En modo dummy la predicción principal tiene score > 0.

        El modelo sintético asigna probs=[0.52, 0.23, 0.15, 0.10],
        por lo que el score top siempre es 0.52.
        """
        predictions = service.predict(rgb_image)
        assert predictions[0].score > 0.0

    def test_low_confidence_returns_otros(self) -> None:
        """
        Cuando el score top es menor al umbral, el servicio debe retornar
        la categoría 'otros' con la misma puntuación.

        Se prueba creando un servicio con min_confidence=1.0 para forzar
        que cualquier predicción quede por debajo del umbral.
        """

        cfg = AppConfig.__new__(AppConfig)
        object.__setattr__(cfg, "app_title", "Test")
        object.__setattr__(cfg, "model_id", "facebook/deit-tiny-patch16-224")
        object.__setattr__(cfg, "top_k", 3)
        object.__setattr__(cfg, "max_upload_mb", 8)
        object.__setattr__(cfg, "min_confidence", 1.0)  # umbral imposible
        object.__setattr__(cfg, "classifier_path", "models/classifier.pkl")
        object.__setattr__(cfg, "labels_path", "models/labels.json")
        object.__setattr__(cfg, "use_grpc_backend", False)
        object.__setattr__(cfg, "grpc_target", "127.0.0.1:50051")

        svc = TriageService(cfg=cfg)
        svc.load()
        image = Image.new("RGB", (224, 224), color=(0, 255, 0))
        predictions = svc.predict(image)

        assert predictions[0].category_key == "otros"


class TestServiceInfo:
    """Verifica que info() retorne los metadatos esperados del modelo."""

    def test_info_has_required_keys(self, service: TriageService) -> None:
        """
        info() debe retornar un dict con todas las claves documentadas en T05.

        Estas claves son las que se incluyen en ModelInfoResponse y se
        exponen al cliente en la respuesta de /api/predict.
        """
        info = service.info()
        required = {
            "app_title",
            "model_id",
            "top_k",
            "max_upload_mb",
            "min_confidence",
            "ready",
            "dummy_mode",
            "labels",
            "classifier_path",
            "labels_path",
            "categories",
        }
        assert required.issubset(info.keys()), (
            f"Claves faltantes en info(): {required - info.keys()}"
        )

    def test_info_ready_is_true_after_load(self, service: TriageService) -> None:
        """Después de load(), el servicio debe reportar ready=True."""
        assert service.info()["ready"] is True

    def test_info_dummy_mode_is_true(self, service: TriageService) -> None:
        """En el entorno de pruebas, dummy_mode debe ser True."""
        assert service.info()["dummy_mode"] is True


class TestImageVariants:
    """Verifica que el servicio acepte distintos tamaños y modos de imagen."""

    @pytest.mark.parametrize("size", [(64, 64), (224, 224), (512, 512)])
    def test_accepts_various_sizes(
        self, service: TriageService, size: tuple[int, int]
    ) -> None:
        """
        El servicio debe procesar imágenes de distintas resoluciones sin error.

        DeiT-Tiny redimensiona internamente a 224×224, por lo que el tamaño
        de entrada no debe afectar la capacidad de inferencia.
        """
        image = Image.new("RGB", size, color=(128, 128, 128))
        predictions = service.predict(image)
        assert len(predictions) >= 1

    def test_accepts_rgba_image(self, service: TriageService) -> None:
        """
        El servicio debe convertir imágenes RGBA a RGB sin lanzar excepción.

        Las capturas de pantalla en sistemas macOS suelen venir en RGBA;
        TriageService._embed() las convierte antes de extraer el embedding.
        """
        image = Image.new("RGBA", (224, 224), color=(100, 100, 100, 200))
        predictions = service.predict(image)
        assert len(predictions) >= 1
