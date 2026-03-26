"""
src/api/pdf_builder.py

Generación del reporte PDF para el triage de soporte TI.

Responsabilidad única: construir el documento PDF a partir de los datos de
predicción entregados por TriageService. No depende de FastAPI ni de la capa
de inferencia; recibe un DTO ya resuelto desde el router.

Uso típico desde routes/report.py:
    data = ReportData.from_service(predictions, filename, timestamp)
    pdf_bytes = build_report_pdf(data)
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from dataclasses import field
from datetime import datetime
from datetime import UTC

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import HRFlowable
from reportlab.platypus import Paragraph
from reportlab.platypus import SimpleDocTemplate
from reportlab.platypus import Spacer
from reportlab.platypus import Table
from reportlab.platypus import TableStyle


@dataclass
class PredictionRow:
    """Una fila de resultado: categoría, clave, score y equipo."""

    category: str
    category_key: str
    score: float
    team: str


@dataclass
class ReportData:
    """
    Datos completos necesarios para construir el PDF de triage.

    Attributes:
        predictions    : Lista ordenada de predicciones (top-k del modelo).
        image_filename : Nombre del archivo de imagen analizado.
        timestamp      : Momento UTC de la clasificación.
        min_confidence : Umbral de confianza mínima (default: 0.40).
    """

    predictions: list[PredictionRow]
    image_filename: str = "imagen_incidente"
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))
    min_confidence: float = 0.40

    @classmethod
    def from_service(
        cls,
        predictions: list[dict],
        image_filename: str,
        timestamp: datetime | None = None,
        min_confidence: float = 0.40,
    ) -> ReportData:
        """
        Construye ReportData a partir de la salida de TriageService.predict().

        Args:
            predictions    : Lista de dicts con keys category, category_key,
                             score, team — formato directo de predict.py.
            image_filename : Nombre del archivo original subido.
            timestamp      : Fecha/hora de clasificación; si None usa utcnow.
            min_confidence : Umbral de confianza del modelo (de AppConfig).

        Returns:
            Instancia de ReportData lista para pasar a build_report_pdf().
        """
        rows = [
            PredictionRow(
                category=p["category"],
                category_key=p["category_key"],
                score=p["score"],
                team=p["team"],
            )
            for p in predictions
        ]
        return cls(
            predictions=rows,
            image_filename=image_filename,
            timestamp=timestamp or datetime.now(UTC),
            min_confidence=min_confidence,
        )


class _P:
    NAVY = colors.HexColor("#0F2645")
    BLUE = colors.HexColor("#1A4A8A")
    STEEL = colors.HexColor("#4A7FC1")
    RED = colors.HexColor("#C0392B")
    GREEN = colors.HexColor("#1E7E4A")
    ORANGE = colors.HexColor("#D4860A")
    OFF_WHITE = colors.HexColor("#F7F9FC")
    RULE = colors.HexColor("#D0DAE8")
    BODY = colors.HexColor("#1A1A2E")
    MUTED = colors.HexColor("#6B7A99")
    WHITE = colors.white


class _PDFBuilder:
    """
    Construye el PDF de triage usando ReportLab Platypus.

    No se instancia directamente desde fuera del módulo; usar build_report_pdf().
    """

    PAGE = A4
    MARGIN = 2.2 * cm

    def __init__(self, data: ReportData) -> None:
        self._d = data
        self._st = getSampleStyleSheet()
        self._buf = io.BytesIO()
        self._story: list = []

    def build(self) -> bytes:
        """Construye el documento y retorna los bytes del PDF."""
        doc = SimpleDocTemplate(
            self._buf,
            pagesize=self.PAGE,
            leftMargin=self.MARGIN,
            rightMargin=self.MARGIN,
            topMargin=3.2 * cm,
            bottomMargin=2.0 * cm,
            title=f"Reporte Triage TI — {self._d.timestamp:%Y-%m-%d}",
            author="Sistema de Triage de Soporte TI — UAO",
        )
        self._build_story()
        doc.build(
            self._story,
            onFirstPage=self._draw_chrome,
            onLaterPages=self._draw_chrome,
        )
        return self._buf.getvalue()

    def _build_story(self) -> None:
        s = self._story
        s.clear()
        self._section("Datos del análisis")
        s.append(Spacer(1, 0.3 * cm))
        self._meta_table()
        s.append(Spacer(1, 0.6 * cm))
        self._section("Resultados de clasificación")
        s.append(Spacer(1, 0.3 * cm))
        self._results_table()
        s.append(Spacer(1, 0.6 * cm))
        self._action_banner()
        s.append(Spacer(1, 0.8 * cm))
        self._footer_note()

    def _section(self, text: str) -> None:
        """Encabezado de sección: texto en mayúsculas con línea inferior."""
        self._story.append(
            Paragraph(
                text.upper(),
                ParagraphStyle(
                    "Sec",
                    parent=self._st["Normal"],
                    fontName="Helvetica-Bold",
                    fontSize=7,
                    textColor=_P.STEEL,
                    letterSpacing=1.5,
                ),
            )
        )
        self._story.append(
            HRFlowable(width="100%", thickness=0.75, color=_P.RULE, spaceAfter=0)
        )

    def _meta_table(self) -> None:
        """Tabla de dos columnas con metadatos: archivo, fecha, modelo, ID."""
        d = self._d
        lbl = ParagraphStyle(
            "ML", parent=self._st["Normal"], fontSize=8, textColor=_P.MUTED
        )
        val = ParagraphStyle(
            "MV",
            parent=self._st["Normal"],
            fontSize=8.5,
            textColor=_P.BODY,
            fontName="Helvetica-Bold",
        )
        rows = [
            [
                Paragraph("Archivo analizado", lbl),
                Paragraph(d.image_filename, val),
                Paragraph("Fecha y hora", lbl),
                Paragraph(f"{d.timestamp:%d %b %Y  •  %H:%M:%S} UTC", val),
            ],
            [
                Paragraph("Modelo", lbl),
                Paragraph("DeiT-Tiny · facebook/deit-tiny-patch16-224", val),
                Paragraph("Umbral de confianza", lbl),
                Paragraph(f"{d.min_confidence:.0%}", val),
            ],
        ]
        tbl = Table(rows, colWidths=[3.2 * cm, 6.3 * cm, 3.2 * cm, 4.3 * cm])
        tbl.setStyle(
            TableStyle(
                [
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("TOPPADDING", (0, 0), (-1, -1), 5),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                    ("LEFTPADDING", (0, 0), (-1, -1), 0),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                    ("LINEBELOW", (0, 0), (-1, -2), 0.3, _P.RULE),
                ]
            )
        )
        self._story.append(tbl)

    def _results_table(self) -> None:
        """
        Tabla principal con todas las predicciones top-k del modelo.
        Cada fila muestra categoría, equipo, barra de confianza y porcentaje.
        """
        hdr_st = ParagraphStyle(
            "H",
            parent=self._st["Normal"],
            fontSize=8,
            fontName="Helvetica-Bold",
            textColor=_P.WHITE,
        )
        rows = [
            [
                Paragraph("CATEGORÍA", hdr_st),
                Paragraph("EQUIPO DESTINATARIO", hdr_st),
                Paragraph("CONFIANZA", hdr_st),
                Paragraph("%", hdr_st),
            ]
        ]

        for i, pred in enumerate(self._d.predictions):
            score = pred.score
            pct = f"{score * 100:.1f} %"
            bar = self._score_bar(score)

            if score >= 0.70:
                score_color = _P.GREEN
            elif score >= self._d.min_confidence:
                score_color = _P.ORANGE
            else:
                score_color = _P.RED

            cell_st = ParagraphStyle(
                f"C{i}",
                parent=self._st["Normal"],
                fontSize=8.5 if i == 0 else 8,
                fontName="Helvetica-Bold" if i == 0 else "Helvetica",
                textColor=_P.BODY,
            )
            score_st = ParagraphStyle(
                f"S{i}",
                parent=self._st["Normal"],
                fontSize=8,
                fontName="Helvetica-Bold",
                textColor=score_color,
            )
            rows.append(
                [
                    Paragraph(pred.category, cell_st),
                    Paragraph(pred.team, cell_st),
                    Paragraph(bar, score_st),
                    Paragraph(pct, score_st),
                ]
            )

        col_w = [5 * cm, 5 * cm, 4.5 * cm, 2.5 * cm]
        tbl = Table(rows, colWidths=col_w, repeatRows=1)

        style = [
            ("BACKGROUND", (0, 0), (-1, 0), _P.NAVY),
            ("TOPPADDING", (0, 0), (-1, -1), 7),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
            ("LEFTPADDING", (0, 0), (-1, -1), 10),
            ("RIGHTPADDING", (0, 0), (-1, -1), 10),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("BOX", (0, 0), (-1, -1), 0.5, _P.RULE),
            ("LINEAFTER", (0, 0), (-2, -1), 0.3, _P.RULE),
            ("LINEBELOW", (0, 0), (-1, 0), 1.5, _P.STEEL),
        ]
        # Fila ganadora con fondo ligeramente destacado
        if len(rows) > 1:
            style.append(("BACKGROUND", (0, 1), (-1, 1), _P.OFF_WHITE))
        # Filas alternas del resto
        for row_idx in range(2, len(rows)):
            bg = colors.HexColor("#FFFFFF") if row_idx % 2 == 0 else _P.OFF_WHITE
            style.append(("BACKGROUND", (0, row_idx), (-1, row_idx), bg))

        tbl.setStyle(TableStyle(style))
        self._story.append(tbl)

    def _action_banner(self) -> None:
        """
        Banner de acción recomendada según el score de la predicción principal.
        Verde si la confianza es suficiente para enrutar; rojo si requiere revisión.
        """
        top = self._d.predictions[0]
        score = top.score

        if score >= self._d.min_confidence:
            bg, border = colors.HexColor("#F2FAF5"), _P.GREEN
            icon, title = "✓", f"Enrutar a: {top.team}"
            detail = (
                f"Clasificación aceptada con una confianza del {score:.0%}. "
                "El ticket puede asignarse automáticamente al equipo indicado."
            )
        else:
            bg, border = colors.HexColor("#FDF2F2"), _P.RED
            icon, title = "⚠", "Requiere revisión humana"
            detail = (
                f"El score ({score:.0%}) es inferior al umbral mínimo "
                f"({self._d.min_confidence:.0%}). "
                "Este ticket debe revisarse manualmente antes de ser asignado."
            )

        title_st = ParagraphStyle(
            "BT",
            parent=self._st["Normal"],
            fontSize=10,
            fontName="Helvetica-Bold",
            textColor=border,
            leading=14,
        )
        detail_st = ParagraphStyle(
            "BD",
            parent=self._st["Normal"],
            fontSize=8.5,
            textColor=_P.MUTED,
            leading=13,
        )

        content = [
            Paragraph(f"{icon}  {title}", title_st),
            Spacer(1, 4),
            Paragraph(detail, detail_st),
        ]
        banner = Table([[content]], colWidths=[17 * cm])
        banner.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, -1), bg),
                    ("LINEBEFORE", (0, 0), (0, -1), 4, border),
                    ("BOX", (0, 0), (-1, -1), 0.4, _P.RULE),
                    ("TOPPADDING", (0, 0), (-1, -1), 12),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 12),
                    ("LEFTPADDING", (0, 0), (-1, -1), 14),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 14),
                ]
            )
        )
        self._story.append(banner)

    def _footer_note(self) -> None:
        """Nota de descargo al pie de la última sección."""
        self._story.append(HRFlowable(width="100%", thickness=0.4, color=_P.RULE))
        self._story.append(Spacer(1, 0.2 * cm))
        self._story.append(
            Paragraph(
                "Este reporte es generado automáticamente con carácter orientativo. "
                "El modelo puede cometer errores; ante dudas, escalar a revisión "
                "humana.",
                ParagraphStyle(
                    "FN",
                    parent=self._st["Normal"],
                    fontSize=7,
                    textColor=_P.MUTED,
                    leading=10,
                ),
            )
        )

    @staticmethod
    def _draw_chrome(canvas, doc) -> None:
        """Dibuja el encabezado y pie de página en cada hoja."""
        canvas.saveState()
        w, h = A4
        m = 2.2 * cm

        # Banda superior
        canvas.setFillColor(_P.NAVY)
        canvas.rect(0, h - 1.8 * cm, w, 1.8 * cm, fill=1, stroke=0)
        canvas.setFillColor(_P.STEEL)
        canvas.rect(0, h - 1.98 * cm, w, 0.18 * cm, fill=1, stroke=0)

        canvas.setFont("Helvetica-Bold", 11)
        canvas.setFillColor(_P.WHITE)
        canvas.drawString(m, h - 1.25 * cm, "Reporte de Triage de Soporte TI")

        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(colors.HexColor("#A8C4E0"))
        canvas.drawRightString(w - m, h - 1.25 * cm, f"UAO · {datetime.now():%d/%m/%Y}")

        # Pie de página
        canvas.setStrokeColor(_P.RULE)
        canvas.setLineWidth(0.4)
        canvas.line(m, 1.5 * cm, w - m, 1.5 * cm)
        canvas.setFont("Helvetica", 7)
        canvas.setFillColor(_P.MUTED)
        canvas.drawString(
            m, 1.1 * cm, "Triage de Soporte TI — Desarrollo de Proyectos de IA"
        )
        canvas.drawCentredString(w / 2, 1.1 * cm, f"Página {doc.page}")
        canvas.drawRightString(w - m, 1.1 * cm, f"Generado: {datetime.now():%H:%M:%S}")
        canvas.restoreState()

    @staticmethod
    def _score_bar(score: float, width: int = 12) -> str:
        """Genera una barra visual de texto proporcional al score."""
        filled = round(score * width)
        return "█" * filled + "░" * (width - filled)


def build_report_pdf(data: ReportData) -> bytes:
    """
    Genera el PDF de reporte de triage y retorna sus bytes.

    Args:
        data: DTO con predicciones, metadatos y configuración del modelo.

    Returns:
        Bytes del PDF listo para enviar como StreamingResponse o guardar en disco.

    Example:
        data = ReportData.from_service(predictions, "captura.png", timestamp)
        pdf_bytes = build_report_pdf(data)
    """
    return _PDFBuilder(data).build()
