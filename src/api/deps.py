from __future__ import annotations

from src.config import AppConfig
from src.grpc.client import GrpcTriageClient
from src.inference.service import TriageService

cfg = AppConfig()
triage_service = TriageService(cfg=cfg)
grpc_client = GrpcTriageClient(cfg.grpc_target) if cfg.use_grpc_backend else None
