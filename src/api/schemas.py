from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel
from pydantic import Field


class PredictionItem(BaseModel):
    category: str = Field(..., description="Etiqueta legible de la categoría")
    category_key: str = Field(..., description="Clave interna de la categoría")
    score: float = Field(..., ge=0.0, le=1.0)
    team: str = Field(..., description="Equipo sugerido para enrutar")


class ModelInfoResponse(BaseModel):
    app_title: str
    model_id: str
    top_k: int
    max_upload_mb: int
    min_confidence: float
    ready: bool
    dummy_mode: bool
    labels: list[str]
    classifier_path: str
    labels_path: str
    load_seconds: float | None = None
    categories: list[str]


class PredictionResponse(BaseModel):
    predictions: list[PredictionItem]
    model: ModelInfoResponse
    timestamp: datetime
