import os
from dataclasses import dataclass
from dataclasses import field


def _env(name: str, default: str) -> str:
    return os.getenv(name, default)


def _env_int(name: str, default: int) -> int:
    try:
        return int(_env(name, str(default)))
    except Exception:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(_env(name, str(default)))
    except Exception:
        return default


@dataclass(frozen=True)
class AppConfig:
    app_title: str = field(
        default_factory=lambda: _env(
            "APP_TITLE",
            "Triage de soporte TI mediante clasificación de imágenes",
        )
    )
    model_id: str = field(
        default_factory=lambda: _env("MODEL_ID", "facebook/deit-tiny-patch16-224")
    )
    top_k: int = field(default_factory=lambda: _env_int("TOP_K", 3))
    max_upload_mb: int = field(default_factory=lambda: _env_int("MAX_UPLOAD_MB", 8))
    min_confidence: float = field(
        default_factory=lambda: _env_float("MIN_CONFIDENCE", 0.40)
    )
    classifier_path: str = field(
        default_factory=lambda: _env("CLASSIFIER_PATH", "models/classifier.pkl")
    )
    labels_path: str = field(
        default_factory=lambda: _env("LABELS_PATH", "models/labels.json")
    )
    use_grpc_backend: bool = field(
        default_factory=lambda: _env("USE_GRPC_BACKEND", "0") == "1"
    )
    grpc_target: str = field(
        default_factory=lambda: _env("GRPC_TARGET", "127.0.0.1:50051")
    )
