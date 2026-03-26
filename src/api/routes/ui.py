"""
RF-05: rutas HTML — interfaz web para triage de imágenes TI.

Responsabilidad única: renderizar templates Jinja2. No contiene
lógica de inferencia; la UI llama al endpoint /api/predict vía fetch.
"""

from __future__ import annotations

from fastapi import APIRouter
from fastapi import Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from src.domain.categories import CATEGORIES

router = APIRouter()
templates = Jinja2Templates(directory="src/api/templates")


@router.get("/", response_class=HTMLResponse)
async def index(request: Request) -> HTMLResponse:
    """Renderiza la página principal de triage."""
    return templates.TemplateResponse(
        "index.html",
        {"request": request, "categories": CATEGORIES},
    )
