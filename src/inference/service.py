from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from src.config import AppConfig
from src.domain.categories import CATEGORIES
from src.domain.categories import get_category
from src.inference import model as embedding_model

USE_DUMMY_MODEL = os.getenv("USE_DUMMY_MODEL", "0") == "1"


@dataclass(frozen=True)
class Prediction:
    category_key: str
    category_label: str
    score: float
    team: str


class TriageService:
    def __init__(self, cfg: AppConfig | None = None) -> None:
        self.cfg = cfg or AppConfig()
        self._ready = False
        self._clf_artifact: dict[str, Any] | None = None
        self._labels: list[str] = []
        self._load_seconds: float | None = None

    def load(self) -> None:
        if self._ready:
            return

        if USE_DUMMY_MODEL:
            self._labels = [
                "red_conectividad",
                "correo_office365",
                "aplicacion_errores",
                "otros",
            ]
            self._ready = True
            return

        started = time.time()
        labels_path = Path(self.cfg.labels_path)
        if not labels_path.exists():
            raise FileNotFoundError(
                f"No se encontró '{labels_path}'. "
                "Ejecuta el pipeline de entrenamiento primero."
            )
        info = json.loads(labels_path.read_text(encoding="utf-8"))
        self._labels = list(info["labels"])

        from src.inference import classifier

        self._clf_artifact = classifier.load(Path(self.cfg.classifier_path))
        self._load_seconds = time.time() - started
        self._ready = True

    def _embed(self, image: Image.Image) -> np.ndarray:
        if not self._ready:
            self.load()

        if image.mode not in ("RGB", "RGBA"):
            image = image.convert("RGB")
        if image.mode == "RGBA":
            image = image.convert("RGB")

        if USE_DUMMY_MODEL:
            rng = np.random.default_rng(42)
            return rng.normal(size=(192,)).astype(np.float32)

        return embedding_model.extract_embedding(image).astype(np.float32)

    def predict(self, image: Image.Image) -> list[Prediction]:
        emb = self._embed(image).reshape(1, -1)

        if USE_DUMMY_MODEL:
            probs = np.array([[0.52, 0.23, 0.15, 0.10]], dtype=np.float32)
            labels = self._labels
        else:
            assert self._clf_artifact is not None
            clf = self._clf_artifact["clf"]
            if not hasattr(clf, "predict_proba"):
                raise RuntimeError(
                    "El clasificador no soporta predict_proba. Usa LogisticRegression."
                )
            probs = clf.predict_proba(emb)
            labels = self._labels

        top_k = min(self.cfg.top_k, probs.shape[1])
        idxs = np.argsort(-probs[0])[:top_k]
        preds: list[Prediction] = []
        for idx in idxs:
            key = labels[int(idx)]
            category = get_category(key)
            score = float(probs[0, int(idx)])
            preds.append(
                Prediction(
                    category_key=category.key,
                    category_label=category.label,
                    score=score,
                    team=category.team,
                )
            )

        if preds and preds[0].score < self.cfg.min_confidence:
            fallback = get_category("otros")
            preds = [
                Prediction(
                    category_key=fallback.key,
                    category_label=fallback.label,
                    score=float(preds[0].score),
                    team=fallback.team,
                )
            ]

        return preds

    def info(self) -> dict[str, Any]:
        return {
            "app_title": self.cfg.app_title,
            "model_id": self.cfg.model_id,
            "top_k": self.cfg.top_k,
            "max_upload_mb": self.cfg.max_upload_mb,
            "min_confidence": self.cfg.min_confidence,
            "ready": self._ready,
            "dummy_mode": USE_DUMMY_MODEL,
            "labels": self._labels,
            "classifier_path": self.cfg.classifier_path,
            "labels_path": self.cfg.labels_path,
            "load_seconds": self._load_seconds,
            "categories": [c.key for c in CATEGORIES],
        }
