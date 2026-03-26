"""
Punto de entrada de la aplicación FastAPI.

Responsabilidad única: configurar la app, registrar routers, montar
archivos estáticos y gestionar el ciclo de vida del servidor.
No contiene lógica de negocio ni de inferencia.
"""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from src.api.deps import triage_service
from src.api.routes import health
from src.api.routes import predict
from src.api.routes import report
from src.api.routes import ui

app = FastAPI(
    title="Triage de Soporte TI cambios 123",
    description="Clasificación de imágenes de incidencias TI mediante DeiT-Tiny",
    version="0.1.0",
)

app.mount("/static", StaticFiles(directory="src/api/static"), name="static")

app.include_router(health.router)
app.include_router(predict.router, prefix="/api")
app.include_router(report.router, prefix="/api")
app.include_router(ui.router)


@app.on_event("startup")
def startup_event() -> None:
    """
    Precarga el modelo al iniciar el servidor para evitar latencia en el primer request.
    """
    triage_service.load()
