"""
RF-06: POST /api/report — genera y descarga el reporte PDF de triage.

Responsabilidad única: recibir la imagen, delegar la inferencia a
TriageService (igual que predict.py) y retornar el PDF como descarga.
No contiene lógica de modelo ni de construcción de documentos.
"""

from __future__ import annotations

import io
from datetime import datetime
from datetime import UTC

from fastapi import APIRouter
from fastapi import UploadFile
from fastapi.responses import StreamingResponse

from src.api.deps import cfg
from src.api.deps import triage_service
from src.api.pdf_builder import build_report_pdf
from src.api.pdf_builder import ReportData
from src.api.utils import read_uploaded_image

router = APIRouter()


@router.post("/report", summary="Genera y descarga el reporte PDF de triage")
def report_pdf(image: UploadFile) -> StreamingResponse:
    """
    Clasifica la imagen y retorna un PDF descargable con el reporte de triage.

    El flujo es idéntico al de /api/predict: valida la imagen, invoca
    TriageService.predict() y delega la construcción del PDF a pdf_builder.

    El PDF incluye:
    - Metadatos del análisis (archivo, fecha, modelo, umbral)
    - Tabla con predicciones top-k (categoría, equipo, score, barra visual)
    - Banner de acción recomendada (enrutar o revisar manualmente)

    Args:
        image: Imagen de la incidencia TI (multipart/form-data).
               Formatos aceptados: image/*.

    Returns:
        StreamingResponse con el PDF adjunto para descarga directa.

    Raises:
        HTTPException 400: archivo no es imagen o está corrupto.
        HTTPException 413: imagen supera el límite de tamaño configurado.
    """
    img = read_uploaded_image(image, cfg)
    timestamp = datetime.now(UTC)

    predictions = [
        {
            "category": p.category_label,
            "category_key": p.category_key,
            "score": p.score,
            "team": p.team,
        }
        for p in triage_service.predict(img)
    ]

    data = ReportData.from_service(
        predictions=predictions,
        image_filename=image.filename or "imagen_incidente",
        timestamp=timestamp,
        min_confidence=cfg.min_confidence,
    )

    pdf_bytes = build_report_pdf(data)
    filename = f"triage_{timestamp:%Y%m%d_%H%M%S}.pdf"

    return StreamingResponse(
        io.BytesIO(pdf_bytes),
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )
