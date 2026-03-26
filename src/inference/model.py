"""
Carga del modelo DeiT-Tiny y extracción de embeddings visuales.

Responsabilidad única: dado una imagen PIL, retorna el vector de embeddings
del token CLS (192 dimensiones). No sabe nada de FastAPI ni de sklearn.

Usa carga diferida (lazy loading): el modelo se descarga de HuggingFace
la primera vez que se invoca extract_embedding() y se reutiliza en las
llamadas siguientes sin volver a cargarlo.

Modelo: facebook/deit-tiny-patch16-224
  - Arquitectura : Vision Transformer (ViT), parches 16x16 px
  - Entrada      : imágenes RGB 224x224
  - Embedding    : vector CLS de 192 dimensiones
  - Tamaño disco : ~22 MB
  - RAM en uso   : ~200 MB (CPU)
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch

if TYPE_CHECKING:
    from PIL import Image

MODEL_ID = "facebook/deit-tiny-patch16-224"
CACHE_DIR = Path("models/hf_cache")

# Estado interno del módulo — inicializado una sola vez
_processor = None
_model = None


def _load() -> None:
    """Descarga y carga el modelo en memoria (solo la primera vez)."""
    global _processor, _model

    if _processor is not None:
        return

    from transformers import AutoImageProcessor
    from transformers import AutoModel

    _processor = AutoImageProcessor.from_pretrained(MODEL_ID, cache_dir=CACHE_DIR)
    _model = AutoModel.from_pretrained(MODEL_ID, cache_dir=CACHE_DIR)
    _model.eval()


def extract_embedding(image: Image.Image) -> np.ndarray:
    """
    Extrae el embedding del token CLS para una imagen PIL.

    Args:
        image: imagen RGB en formato PIL.Image.

    Returns:
        Vector numpy de 192 dimensiones (float32).
    """
    _load()

    inputs = _processor(images=image.convert("RGB"), return_tensors="pt")

    with torch.no_grad():
        outputs = _model(**inputs)

    # Token CLS: índice 0 del último estado oculto → shape (192,)
    embedding: np.ndarray = outputs.last_hidden_state[:, 0, :].squeeze().cpu().numpy()
    return embedding


def embedding_dim() -> int:
    """Retorna la dimensión del embedding (192 para DeiT-Tiny)."""
    _load()
    return _model.config.hidden_size
