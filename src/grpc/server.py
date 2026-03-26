from __future__ import annotations

import io
import os
from concurrent import futures

from PIL import Image
from src.config import AppConfig
from src.grpc.stubs import triage_pb2
from src.grpc.stubs import triage_pb2_grpc
from src.inference.service import TriageService

import grpc


def _read_image_bytes(image_bytes: bytes) -> Image.Image:
    try:
        image = Image.open(io.BytesIO(image_bytes))
        image.load()
        if image.mode not in ("RGB", "RGBA"):
            image = image.convert("RGB")
        if image.mode == "RGBA":
            image = image.convert("RGB")
        return image
    except Exception as exc:
        raise ValueError(
            "No se pudo leer la imagen. Formato inválido o corrupto."
        ) from exc


class TriageGrpcService(triage_pb2_grpc.TriageServiceServicer):
    def __init__(self, cfg: AppConfig | None = None) -> None:
        self.cfg = cfg or AppConfig()
        self.triage = TriageService(cfg=self.cfg)
        self.triage.load()

    def PredictImage(self, request, context):
        if not request.image_bytes:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, "image_bytes vacío.")
        try:
            image = _read_image_bytes(request.image_bytes)
            predictions = self.triage.predict(image)
            info = self.triage.info()
            return triage_pb2.PredictImageResponse(
                predictions=[
                    triage_pb2.Prediction(
                        category=item.category_label,
                        score=item.score,
                        routed_to=item.team,
                    )
                    for item in predictions
                ],
                model=triage_pb2.ModelInfo(
                    model_id=str(info.get("model_id", "")),
                    top_k=int(info.get("top_k", 0)),
                    min_confidence=float(info.get("min_confidence", 0.0)),
                    dummy_mode=bool(info.get("dummy_mode", False)),
                    ready=bool(info.get("ready", False)),
                ),
            )
        except ValueError as exc:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(exc))
        except FileNotFoundError as exc:
            context.abort(grpc.StatusCode.FAILED_PRECONDITION, str(exc))
        except Exception as exc:
            context.abort(grpc.StatusCode.INTERNAL, f"Error interno: {exc}")

    def Health(self, request, context):
        info = self.triage.info()
        return triage_pb2.HealthResponse(
            status="ok",
            model=triage_pb2.ModelInfo(
                model_id=str(info.get("model_id", "")),
                top_k=int(info.get("top_k", 0)),
                min_confidence=float(info.get("min_confidence", 0.0)),
                dummy_mode=bool(info.get("dummy_mode", False)),
                ready=bool(info.get("ready", False)),
            ),
        )


def serve() -> None:
    host = os.getenv("GRPC_HOST", "0.0.0.0")
    port = int(os.getenv("GRPC_PORT", "50051"))
    workers = int(os.getenv("GRPC_WORKERS", "4"))

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=workers))
    triage_pb2_grpc.add_TriageServiceServicer_to_server(TriageGrpcService(), server)
    server.add_insecure_port(f"{host}:{port}")
    server.start()
    print(f"[gRPC] TriageService escuchando en {host}:{port}")
    server.wait_for_termination()


if __name__ == "__main__":
    serve()
