import io
import os
from concurrent import futures

from PIL import Image

import grpc

os.environ["USE_DUMMY_MODEL"] = "1"

from src.grpc.server import TriageGrpcService
from src.grpc.stubs import triage_pb2
from src.grpc.stubs import triage_pb2_grpc


def _make_image_bytes() -> bytes:
    image = Image.new("RGB", (64, 64), (120, 200, 80))
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


def test_grpc_health_ok() -> None:
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=2))
    triage_pb2_grpc.add_TriageServiceServicer_to_server(TriageGrpcService(), server)
    server.add_insecure_port("127.0.0.1:50055")
    server.start()
    try:
        channel = grpc.insecure_channel("127.0.0.1:50055")
        stub = triage_pb2_grpc.TriageServiceStub(channel)
        health = stub.Health(triage_pb2.HealthRequest())
        assert health.status == "ok"
    finally:
        server.stop(0)


def test_grpc_predict_ok() -> None:
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=2))
    triage_pb2_grpc.add_TriageServiceServicer_to_server(TriageGrpcService(), server)
    server.add_insecure_port("127.0.0.1:50056")
    server.start()
    try:
        channel = grpc.insecure_channel("127.0.0.1:50056")
        stub = triage_pb2_grpc.TriageServiceStub(channel)
        response = stub.PredictImage(
            triage_pb2.PredictImageRequest(
                image_bytes=_make_image_bytes(),
                filename="sample.png",
                content_type="image/png",
            )
        )
        assert len(response.predictions) >= 1
        assert response.predictions[0].category != ""
    finally:
        server.stop(0)
