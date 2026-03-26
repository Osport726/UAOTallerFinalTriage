from __future__ import annotations

from datetime import datetime

from fastapi import APIRouter
from fastapi import UploadFile
from fastapi.responses import JSONResponse

from src.api.deps import cfg
from src.api.deps import grpc_client
from src.api.deps import triage_service
from src.api.schemas import PredictionResponse
from src.api.utils import image_to_png_bytes
from src.api.utils import read_uploaded_image
from src.domain.categories import get_category_by_label

router = APIRouter()


def _predict(upload: UploadFile) -> PredictionResponse:
    image = read_uploaded_image(upload, cfg)
    if grpc_client is not None:
        response = grpc_client.predict_image(
            image_bytes=image_to_png_bytes(image),
            filename=upload.filename or "",
            content_type=upload.content_type or "image/png",
        )
        predictions = []
        for item in response.predictions:
            category = get_category_by_label(item.category)
            predictions.append(
                {
                    "category": item.category,
                    "category_key": category.key,
                    "score": item.score,
                    "team": item.routed_to,
                }
            )
        model_info = {
            "app_title": cfg.app_title,
            "model_id": response.model.model_id,
            "top_k": response.model.top_k,
            "max_upload_mb": cfg.max_upload_mb,
            "min_confidence": response.model.min_confidence,
            "ready": response.model.ready,
            "dummy_mode": response.model.dummy_mode,
            "labels": [],
            "classifier_path": cfg.classifier_path,
            "labels_path": cfg.labels_path,
            "load_seconds": None,
            "categories": [],
        }
    else:
        predictions = [
            {
                "category": item.category_label,
                "category_key": item.category_key,
                "score": item.score,
                "team": item.team,
            }
            for item in triage_service.predict(image)
        ]
        model_info = triage_service.info()

    return PredictionResponse(
        predictions=predictions,
        model=model_info,
        timestamp=datetime.utcnow(),
    )


@router.post("/predict", response_model=PredictionResponse)
def predict_api(image: UploadFile) -> JSONResponse:
    response = _predict(image)
    return JSONResponse(content=response.model_dump(mode="json"))
