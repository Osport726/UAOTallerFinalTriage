"""
Pruebas del endpoint POST /api/report y del módulo pdf_builder — RF-06 / T09.

Cubre dos niveles:
  1. Unitarias — DTO ReportData y construcción del PDF sin HTTP.
  2. Integración — endpoint /api/report con TestClient.

Criterio de aceptación RF-06: mínimo 3 PDFs generados consecutivamente
deben ser válidos y abribles en un lector estándar (verificado por firma).
"""

from __future__ import annotations

import io
from datetime import datetime
from datetime import UTC

import pytest
from fastapi.testclient import TestClient
from PIL import Image
from src.api.pdf_builder import _PDFBuilder
from src.api.pdf_builder import build_report_pdf
from src.api.pdf_builder import PredictionRow
from src.api.pdf_builder import ReportData


def _png_bytes(color: tuple[int, int, int] = (100, 149, 237)) -> bytes:
    """Genera un PNG válido en memoria con el color indicado."""
    buf = io.BytesIO()
    Image.new("RGB", (224, 224), color).save(buf, format="PNG")
    return buf.getvalue()


def _sample_report_data(score: float = 0.87) -> ReportData:
    """
    Construye un ReportData de prueba con una predicción principal.

    Args:
        score: Confianza de la predicción principal (default 0.87).

    Returns:
        ReportData listo para pasar a build_report_pdf() en tests.
    """
    return ReportData(
        predictions=[
            PredictionRow(
                "Red / Conectividad", "red_conectividad", score, "Equipo Redes"
            ),
            PredictionRow(
                "Correo / Office 365", "correo_office365", 0.08, "Equipo O365"
            ),
            PredictionRow("Otros / No clasifica", "otros", 0.05, "Revisión humana"),
        ],
        image_filename="captura_incidente.png",
        timestamp=datetime(2025, 6, 15, 10, 30, tzinfo=UTC),
        min_confidence=0.40,
    )


class TestReportDataDTO:
    """Verifica la construcción y campos del DTO ReportData."""

    def test_fields_assigned_correctly(self) -> None:
        """Los campos del DTO deben reflejar exactamente los valores pasados."""
        d = _sample_report_data()
        assert d.predictions[0].category == "Red / Conectividad"
        assert d.predictions[0].score == pytest.approx(0.87)
        assert d.image_filename == "captura_incidente.png"
        assert d.min_confidence == pytest.approx(0.40)

    def test_from_service_factory(self) -> None:
        """
        from_service() debe construir correctamente el DTO a partir del
        formato de lista de dicts que retorna predict.py.
        """
        raw = [
            {
                "category": "Red / Conectividad",
                "category_key": "red_conectividad",
                "score": 0.72,
                "team": "Equipo Redes",
            },
            {
                "category": "Otros / No clasifica",
                "category_key": "otros",
                "score": 0.28,
                "team": "Revisión humana",
            },
        ]
        d = ReportData.from_service(raw, "test.png", min_confidence=0.40)
        assert len(d.predictions) == 2
        assert d.predictions[0].category_key == "red_conectividad"
        assert isinstance(d.timestamp, datetime)

    def test_from_service_default_timestamp(self) -> None:
        """Si timestamp es None, from_service() debe usar datetime.now(UTC)."""
        d = ReportData.from_service(
            [{"category": "X", "category_key": "otros", "score": 0.3, "team": "R"}],
            "img.png",
        )
        assert isinstance(d.timestamp, datetime)
        assert d.timestamp.tzinfo is not None

    def test_low_score_prediction_row(self) -> None:
        """Un PredictionRow con score < min_confidence es un caso válido."""
        row = PredictionRow("Otros / No clasifica", "otros", 0.25, "Revisión humana")
        assert row.score < 0.40
        assert row.category_key == "otros"


class TestPDFBuilder:
    """Verifica el generador interno de PDF sin pasar por HTTP."""

    def test_build_returns_bytes(self) -> None:
        """build() debe retornar un objeto bytes no vacío."""
        result = _PDFBuilder(_sample_report_data()).build()
        assert isinstance(result, bytes)
        assert len(result) > 0

    def test_pdf_has_valid_signature(self) -> None:
        """
        Los primeros 5 bytes deben ser '%PDF-' — firma estándar del formato PDF.

        Esta es la verificación mínima para garantizar que el archivo es
        abribles en cualquier lector PDF.
        """
        pdf = _PDFBuilder(_sample_report_data()).build()
        assert pdf[:5] == b"%PDF-", "El PDF no tiene la firma estándar '%PDF-'"

    def test_pdf_minimum_size(self) -> None:
        """
        El PDF debe superar 1 KB para garantizar que tiene contenido real.

        Un PDF vacío o con errores de generación suele tener menos de 500 bytes.
        """
        pdf = _PDFBuilder(_sample_report_data()).build()
        assert len(pdf) > 1024, f"PDF demasiado pequeño: {len(pdf)} bytes"

    def test_build_with_low_confidence_banner(self) -> None:
        """
        Con score < min_confidence el builder debe generar el banner rojo
        de revisión humana sin lanzar excepción.
        """
        data = _sample_report_data(score=0.25)  # debajo del umbral 0.40
        pdf = _PDFBuilder(data).build()
        assert pdf[:5] == b"%PDF-"

    def test_build_with_single_prediction(self) -> None:
        """El builder debe funcionar correctamente con una sola predicción."""
        data = ReportData(
            predictions=[
                PredictionRow("Otros / No clasifica", "otros", 0.30, "Revisión humana")
            ],
            image_filename="solo.png",
            timestamp=datetime.now(UTC),
            min_confidence=0.40,
        )
        pdf = _PDFBuilder(data).build()
        assert pdf[:5] == b"%PDF-"

    def test_score_bar_proportional(self) -> None:
        """La barra de score debe tener exactamente `width` caracteres."""
        width = 12
        for score in (0.0, 0.5, 1.0):
            bar = _PDFBuilder._score_bar(score, width)
            assert len(bar) == width, (
                f"Barra con score={score} tiene {len(bar)} chars, esperado {width}"
            )

    def test_score_bar_full_at_one(self) -> None:
        """Con score=1.0 la barra debe estar completamente llena."""
        assert _PDFBuilder._score_bar(1.0, 10) == "█" * 10

    def test_score_bar_empty_at_zero(self) -> None:
        """Con score=0.0 la barra debe estar completamente vacía."""
        assert _PDFBuilder._score_bar(0.0, 10) == "░" * 10


class TestBuildReportPDF:
    """Verifica la función pública build_report_pdf()."""

    def test_returns_valid_pdf(self) -> None:
        """build_report_pdf() debe retornar bytes con firma PDF válida."""
        assert build_report_pdf(_sample_report_data())[:5] == b"%PDF-"

    def test_multiple_calls_produce_independent_pdfs(self) -> None:
        """
        Llamadas consecutivas con datos distintos deben producir PDFs distintos.

        Verifica que el builder no tiene estado compartido entre llamadas
        (el BytesIO interno se crea nuevo en cada build()).
        """
        pdf1 = build_report_pdf(_sample_report_data(score=0.87))
        pdf2 = build_report_pdf(_sample_report_data(score=0.30))
        assert pdf1 != pdf2


class TestAPIReport:
    """Pruebas de integración del endpoint POST /api/report."""

    def test_returns_200(self, client: TestClient) -> None:
        """POST /api/report con imagen válida debe retornar HTTP 200."""
        r = client.post(
            "/api/report",
            files={"image": ("test.png", _png_bytes(), "image/png")},
        )
        assert r.status_code == 200

    def test_content_type_is_pdf(self, client: TestClient) -> None:
        """La respuesta debe tener Content-Type: application/pdf."""
        r = client.post(
            "/api/report",
            files={"image": ("test.png", _png_bytes(), "image/png")},
        )
        assert r.headers["content-type"] == "application/pdf"

    def test_response_body_is_valid_pdf(self, client: TestClient) -> None:
        """El cuerpo de la respuesta debe comenzar con la firma '%PDF-'."""
        r = client.post(
            "/api/report",
            files={"image": ("test.png", _png_bytes(), "image/png")},
        )
        assert r.content[:5] == b"%PDF-"

    def test_content_disposition_is_attachment(self, client: TestClient) -> None:
        """
        El header Content-Disposition debe indicar 'attachment' para que el
        navegador descargue el archivo en lugar de mostrarlo en línea.
        """
        r = client.post(
            "/api/report",
            files={"image": ("test.png", _png_bytes(), "image/png")},
        )
        assert "attachment" in r.headers.get("content-disposition", "")

    def test_filename_has_pdf_extension(self, client: TestClient) -> None:
        """El nombre de archivo en Content-Disposition debe terminar en .pdf."""
        r = client.post(
            "/api/report",
            files={"image": ("test.png", _png_bytes(), "image/png")},
        )
        cd = r.headers.get("content-disposition", "")
        assert ".pdf" in cd

    def test_rejects_non_image(self, client: TestClient) -> None:
        """Un archivo que no es imagen debe retornar HTTP 400."""
        r = client.post(
            "/api/report",
            files={"image": ("doc.txt", b"texto plano", "text/plain")},
        )
        assert r.status_code == 400

    def test_accepts_jpeg(self, client: TestClient) -> None:
        """El endpoint debe aceptar imágenes JPEG además de PNG."""
        buf = io.BytesIO()
        Image.new("RGB", (224, 224), (200, 100, 50)).save(buf, format="JPEG")
        r = client.post(
            "/api/report",
            files={"image": ("test.jpg", buf.getvalue(), "image/jpeg")},
        )
        assert r.status_code == 200
        assert r.content[:5] == b"%PDF-"

    def test_three_consecutive_pdfs_are_valid(self, client: TestClient) -> None:
        """
        Criterio de aceptación RF-06: tres PDFs generados consecutivamente
        deben ser válidos (firma correcta y tamaño > 1 KB).

        Simula el uso real donde distintos operadores generan reportes
        de distintas imágenes sin reiniciar el servidor.
        """
        colors = [(80, 100, 200), (200, 80, 100), (100, 200, 80)]
        for i, color in enumerate(colors, start=1):
            r = client.post(
                "/api/report",
                files={"image": (f"inc_{i}.png", _png_bytes(color), "image/png")},
            )
            assert r.status_code == 200, f"PDF #{i}: status inesperado {r.status_code}"
            assert r.content[:5] == b"%PDF-", f"PDF #{i}: firma inválida"
            assert len(r.content) > 1024, f"PDF #{i}: tamaño sospechosamente pequeño"
