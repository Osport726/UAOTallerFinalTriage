from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
from PIL import Image
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.domain.categories import CATEGORY_KEYS
from src.inference import model

DATA_DIR = Path("data/test")
MODELS_DIR = Path("models")
VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def load_test_dataset() -> tuple[np.ndarray, list[str]]:
    embeddings: list[np.ndarray] = []
    labels: list[str] = []
    for category_key in CATEGORY_KEYS:
        category_dir = DATA_DIR / category_key
        if not category_dir.is_dir():
            continue
        for image_path in category_dir.iterdir():
            if image_path.suffix.lower() not in VALID_EXTENSIONS:
                continue
            image = Image.open(image_path)
            embeddings.append(model.extract_embedding(image))
            labels.append(category_key)
    if not embeddings:
        raise RuntimeError("No se encontraron imágenes de prueba en data/test/.")
    return np.array(embeddings), labels


def main() -> None:
    classifier_artifact = joblib.load(MODELS_DIR / "classifier.pkl")
    labels_info = json.loads((MODELS_DIR / "labels.json").read_text(encoding="utf-8"))
    known_labels = labels_info["labels"]
    X_test, y_test = load_test_dataset()
    clf = classifier_artifact["clf"]
    y_pred = clf.predict(X_test)

    print(classification_report(y_test, y_pred, labels=known_labels, zero_division=0))
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"Macro-F1: {f1_score(y_test, y_pred, average='macro'):.4f}")
    print("Matriz de confusión:")
    print(confusion_matrix(y_test, y_pred, labels=known_labels))


if __name__ == "__main__":
    main()
