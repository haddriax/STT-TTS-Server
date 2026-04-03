from __future__ import annotations

from pydantic import BaseModel


class Audio2FaceConfig(BaseModel):
    host: str = "localhost"
    port: int = 52000
    timeout: float = 30.0  # seconds -- covers full bidirectional streaming call
