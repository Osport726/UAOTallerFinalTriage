"""
Clasificador sklearn entrenado sobre embeddings de DeiT-Tiny.

Responsabilidad única: dado un vector de embeddings, retorna
la categoría predicha y su score de confianza. No sabe nada
de FastAPI ni de cómo se extraen los embeddings.

Flujo de uso:
  1. train()  → entrena el clasificador con embeddings etiquetados
  2. save()   → persiste el artefacto en disk como .pkl
  3. load()   → carga el artefacto desde disk
  4. predict() → retorna (category_key, score) para un embedding

Si score < CONFIDENCE_THRESHOLD → retorna ("otros", score)
independientemente de lo que prediga el modelo.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

from src.domain.categories import CONFIDENCE_THRESHOLD

# Tipo del artefacto serializado
Artifact = dict[str, Any]

DEFAULT_MODEL_PATH = Path("models/classifier.pkl")


def train(
    embeddings: np.ndarray,
    labels: list[str],
) -> Artifact:
    """
    Entrena un clasificador LogisticRegression sobre embeddings.

    Args:
        embeddings: matriz (N, 192) con un embedding por imagen.
        labels    : lista de N claves de categoría (ej. "red_conectividad").

    Returns:
        Artefacto con el clasificador y el encoder listos para save().
    """
    encoder = LabelEncoder()
    y = encoder.fit_transform(labels)

    clf = LogisticRegression(max_iter=1000, random_state=42, C=1.0)
    clf.fit(embeddings, y)

    return {"clf": clf, "encoder": encoder}


def save(artifact: Artifact, path: Path = DEFAULT_MODEL_PATH) -> None:
    """Serializa el artefacto entrenado en un archivo .pkl."""
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, path)


def load(path: Path = DEFAULT_MODEL_PATH) -> Artifact:
    """Carga el artefacto .pkl desde disk."""
    if not path.exists():
        raise FileNotFoundError(
            f"No se encontró el clasificador en '{path}'. "
            "Ejecuta 'make train' para entrenarlo primero."
        )
    return joblib.load(path)


def predict(artifact: Artifact, embedding: np.ndarray) -> tuple[str, float]:
    """
    Predice la categoría de un embedding.

    Args:
        artifact : artefacto cargado con load().
        embedding: vector de 192 dimensiones (salida de model.extract_embedding).

    Returns:
        Tupla (category_key, score) donde score es la probabilidad de la
        clase predicha. Si score < CONFIDENCE_THRESHOLD retorna ("otros", score).
    """
    clf: LogisticRegression = artifact["clf"]
    encoder: LabelEncoder = artifact["encoder"]

    proba = clf.predict_proba(embedding.reshape(1, -1))[0]
    top_idx = int(proba.argmax())
    score = float(proba[top_idx])
    category_key: str = encoder.inverse_transform([top_idx])[0]

    if score < CONFIDENCE_THRESHOLD:
        return "otros", score

    return category_key, score
