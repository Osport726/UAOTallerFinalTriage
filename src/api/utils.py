from __future__ import annotations

import io

from fastapi import HTTPException
from fastapi import UploadFile
from PIL import Image

from src.config import AppConfig


def read_uploaded_image(upload: UploadFile, cfg: AppConfig) -> Image.Image:
    if not upload.content_type or not upload.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="El archivo debe ser una imagen (image/*).",
        )

    data = upload.file.read()
    if len(data) > cfg.max_upload_mb * 1024 * 1024:
        raise HTTPException(
            status_code=413,
            detail=f"Imagen demasiado grande (máx {cfg.max_upload_mb} MB).",
        )

    try:
        image = Image.open(io.BytesIO(data))
        image.load()
        return image
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail="No se pudo leer la imagen. Formato inválido o corrupto.",
        ) from exc


def image_to_png_bytes(image: Image.Image) -> bytes:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()
