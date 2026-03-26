from __future__ import annotations

from src.grpc.stubs import triage_pb2
from src.grpc.stubs import triage_pb2_grpc

import grpc


class GrpcTriageClient:
    def __init__(self, target: str) -> None:
        self.target = target
        self._channel = grpc.insecure_channel(self.target)
        self._stub = triage_pb2_grpc.TriageServiceStub(self._channel)

    def health(self):
        return self._stub.Health(triage_pb2.HealthRequest())

    def predict_image(
        self,
        image_bytes: bytes,
        filename: str = "",
        content_type: str = "image/png",
    ):
        request = triage_pb2.PredictImageRequest(
            image_bytes=image_bytes,
            filename=filename,
            content_type=content_type,
            top_k_override=0,
        )
        return self._stub.PredictImage(request)
