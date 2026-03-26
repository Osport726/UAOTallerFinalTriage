from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
from PIL import Image
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.domain.categories import CATEGORY_KEYS
from src.inference import model

DATA_DIR = Path("data/raw")
TEST_DIR = Path("data/test")
MODELS_DIR = Path("models")
VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
EMBEDDINGS_CACHE = MODELS_DIR / "embeddings_cache.npz"

# Split ratios — train:val:test = 70:15:15
VAL_RATIO = 0.15
TEST_RATIO = 0.15


def mlflow_defaults() -> tuple[str, str, str]:
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns")
    experiment = os.getenv("MLFLOW_EXPERIMENT_NAME", "triage-soporte-ti")
    run_name = os.getenv("MLFLOW_RUN_NAME", "")
    return tracking_uri, experiment, run_name


def load_dataset(use_cache: bool = True) -> tuple[np.ndarray, list[str], list[Path]]:
    """Carga embeddings, etiquetas y rutas.

    Si existe un caché en EMBEDDINGS_CACHE y use_cache=True, lo usa para
    evitar re-extraer embeddings (ahorra ~2 min por ejecución).
    El caché se invalida automáticamente si hay nuevas imágenes en data/raw/.
    """
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # Recopilar rutas primero (siempre, para conocer el estado actual del dataset)
    all_paths: list[Path] = []
    all_labels: list[str] = []
    for category_key in CATEGORY_KEYS:
        category_dir = DATA_DIR / category_key
        if not category_dir.is_dir():
            print(f"[aviso] Carpeta no encontrada: {category_dir}")
            continue
        image_paths = sorted(
            p for p in category_dir.iterdir() if p.suffix.lower() in VALID_EXTENSIONS
        )
        if not image_paths:
            print(f"[aviso] Sin imágenes en: {category_dir}")
            continue
        print(f"{category_key}: {len(image_paths)} imágenes")
        all_paths.extend(image_paths)
        all_labels.extend([category_key] * len(image_paths))

    if not all_paths:
        raise RuntimeError("No se encontraron imágenes válidas en data/raw/.")

    # Intentar usar caché
    if use_cache and EMBEDDINGS_CACHE.exists():
        cached = np.load(EMBEDDINGS_CACHE, allow_pickle=True)
        cached_paths = list(cached["paths"])
        if cached_paths == [str(p) for p in all_paths]:
            print(f"[caché] Embeddings cargados desde {EMBEDDINGS_CACHE}")
            return cached["X"], list(cached["y"]), all_paths

    # Extraer embeddings
    embeddings: list[np.ndarray] = []
    valid_paths: list[Path] = []
    valid_labels: list[str] = []
    for image_path, label in zip(all_paths, all_labels, strict=True):
        try:
            image = Image.open(image_path)
            embeddings.append(model.extract_embedding(image))
            valid_paths.append(image_path)
            valid_labels.append(label)
        except Exception as exc:
            print(f"[error] {image_path.name}: {exc}")

    X = np.array(embeddings)
    np.savez(
        EMBEDDINGS_CACHE,
        X=X,
        y=np.array(valid_labels),
        paths=np.array([str(p) for p in valid_paths]),
    )
    print(f"[caché] Embeddings guardados en {EMBEDDINGS_CACHE}")
    return X, valid_labels, valid_paths


def export_test_images(test_paths: list[Path], test_labels: list[str]) -> None:
    """Copia las imágenes del conjunto de test a data/test/{categoria}/."""
    if TEST_DIR.exists():
        shutil.rmtree(TEST_DIR)

    for path, label in zip(test_paths, test_labels, strict=True):
        dest_dir = TEST_DIR / label
        dest_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, dest_dir / path.name)

    print(f"Imágenes de test exportadas a {TEST_DIR}/ ({len(test_paths)} archivos)")


def build_classifier(name: str) -> Pipeline:
    """Devuelve el pipeline StandardScaler + clasificador según el nombre."""
    import argparse  # noqa: PLC0415

    classifiers: dict = {
        "svm_rbf_c10": SVC(
            kernel="rbf", C=10, gamma="scale", probability=True, random_state=42
        ),
        "svm_rbf_c5": SVC(
            kernel="rbf", C=5, gamma="scale", probability=True, random_state=42
        ),
        "svm_rbf_auto": SVC(
            kernel="rbf", C=10, gamma="auto", probability=True, random_state=42
        ),
        "svm_rbf_c100": SVC(
            kernel="rbf", C=100, gamma="scale", probability=True, random_state=42
        ),
        "svm_linear": SVC(kernel="linear", C=1, probability=True, random_state=42),
        "svm_ovr": OneVsRestClassifier(
            SVC(kernel="rbf", C=10, gamma="scale", probability=True, random_state=42)
        ),
        "rf": RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1),
        "rf500": RandomForestClassifier(n_estimators=500, random_state=42, n_jobs=-1),
        "et": ExtraTreesClassifier(n_estimators=300, random_state=42, n_jobs=-1),
        "gbt": GradientBoostingClassifier(n_estimators=200, random_state=42),
        "knn5": KNeighborsClassifier(n_neighbors=5, metric="cosine"),
        "knn7": KNeighborsClassifier(n_neighbors=7, metric="cosine"),
        "ensemble": VotingClassifier(
            estimators=[
                (
                    "rf",
                    RandomForestClassifier(
                        n_estimators=300, random_state=42, n_jobs=-1
                    ),
                ),
                (
                    "et",
                    ExtraTreesClassifier(n_estimators=300, random_state=42, n_jobs=-1),
                ),
                (
                    "svm",
                    SVC(
                        kernel="rbf",
                        C=10,
                        gamma="scale",
                        probability=True,
                        random_state=42,
                    ),
                ),
            ],
            voting="soft",
        ),
    }
    if name not in classifiers:
        raise argparse.ArgumentTypeError(
            f"Clasificador desconocido: '{name}'. Opciones: {list(classifiers)}"
        )
    return Pipeline([("scaler", StandardScaler()), ("clf", classifiers[name])])


def main() -> None:
    import argparse  # noqa: PLC0415

    parser = argparse.ArgumentParser(
        description="Pipeline de entrenamiento — Triage TI"
    )
    parser.add_argument(
        "--clf",
        default="svm_rbf_c10",
        choices=[
            "svm_rbf_c10",
            "svm_rbf_c5",
            "svm_rbf_auto",
            "svm_rbf_c100",
            "svm_linear",
            "svm_ovr",
            "rf",
            "rf500",
            "et",
            "gbt",
            "knn5",
            "knn7",
            "ensemble",
        ],
        help="Clasificador a usar (default: svm_rbf_c10)",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Ignorar caché de embeddings y re-extraer desde cero",
    )
    args = parser.parse_args()

    print("=== Pipeline de entrenamiento — Triage TI ===")
    X, y, paths = load_dataset(use_cache=not args.no_cache)
    print(f"Total imágenes: {len(y)} | Categorías: {len(set(y))}")

    # Paso 1: separar test (15 %)
    X_trainval, X_test, y_trainval, y_test, paths_trainval, paths_test = (
        train_test_split(
            X,
            y,
            paths,
            test_size=TEST_RATIO,
            random_state=42,
            stratify=y,
        )
    )

    # Paso 2: separar validación del resto (15 % del total ≈ 17.6 % del trainval)
    val_relative = VAL_RATIO / (1.0 - TEST_RATIO)
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval,
        y_trainval,
        test_size=val_relative,
        random_state=42,
        stratify=y_trainval,
    )

    print(
        f"Split → entrenamiento: {len(y_train)} | "
        f"validación: {len(y_val)} | "
        f"test: {len(y_test)}"
    )

    # Exportar imágenes de test para scripts/evaluate.py
    export_test_images(paths_test, y_test)

    # Entrenamiento
    # Pipeline: StandardScaler + clasificador elegido.
    # predict_proba() funciona igual en service.py sin cambios.
    clf = build_classifier(args.clf)
    print(f"Clasificador: {args.clf}")
    clf.fit(X_train, y_train)

    # Métricas de validación
    y_pred_val = clf.predict(X_val)
    macro_f1_val = float(f1_score(y_val, y_pred_val, average="macro"))
    acc_val = float(accuracy_score(y_val, y_pred_val))
    report_val = classification_report(y_val, y_pred_val, zero_division=0)
    cm_val = confusion_matrix(y_val, y_pred_val, labels=sorted(set(y)))

    # Persistencia
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump({"clf": clf, "encoder": None}, MODELS_DIR / "classifier.pkl")
    (MODELS_DIR / "labels.json").write_text(
        json.dumps({"labels": sorted(set(y))}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (MODELS_DIR / "val_report.txt").write_text(report_val, encoding="utf-8")
    np.savetxt(MODELS_DIR / "confusion_matrix.csv", cm_val, delimiter=",", fmt="%d")

    # MLflow
    tracking_uri, experiment, run_name = mlflow_defaults()
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment)
    if not run_name:
        run_name = (
            f"{args.clf}_{len(set(y))}c_"
            f"{len(y_train)}tr_{len(y_val)}val_{len(y_test)}test"
        )

    with mlflow.start_run(run_name=run_name):
        mlflow.log_param("classifier", args.clf)
        mlflow.log_param("model_id", model.MODEL_ID)
        mlflow.log_param("train_samples", len(y_train))
        mlflow.log_param("val_samples", len(y_val))
        mlflow.log_param("test_samples", len(y_test))
        mlflow.log_param("n_classes", len(set(y)))
        mlflow.log_param("val_ratio", VAL_RATIO)
        mlflow.log_param("test_ratio", TEST_RATIO)
        mlflow.log_metric("val_macro_f1", macro_f1_val)
        mlflow.log_metric("val_accuracy", acc_val)
        mlflow.log_artifact(str(MODELS_DIR / "val_report.txt"), artifact_path="reports")
        mlflow.log_artifact(
            str(MODELS_DIR / "confusion_matrix.csv"), artifact_path="reports"
        )
        mlflow.log_artifact(
            str(MODELS_DIR / "labels.json"), artifact_path="model_files"
        )
        mlflow.log_artifact(
            str(MODELS_DIR / "classifier.pkl"), artifact_path="model_files"
        )
        mlflow.sklearn.log_model(clf, artifact_path="sklearn_model")

    print(report_val)
    print(f"Macro-F1 (validación): {macro_f1_val:.4f}")
    print(f"Accuracy  (validación): {acc_val:.4f}")
    print("Artefactos guardados en models/")
    print(f"Imágenes de test disponibles en {TEST_DIR}/ — ejecuta scripts/evaluate.py")


if __name__ == "__main__":
    main()
