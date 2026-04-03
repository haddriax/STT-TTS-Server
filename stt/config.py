from __future__ import annotations

from typing import Optional

from pydantic import BaseModel


class ModelConfig(BaseModel):
    name: str = "base"
    device: str = "auto"
    compute_type: str = "default"


class TranscribeConfig(BaseModel):
    beam_size: int = 5
    temperature: float = 0.0
    vad_filter: bool = True
    language: Optional[str] = None
    condition_on_previous_text: bool = True


class TranslateConfig(BaseModel):
    beam_size: int = 5
    temperature: float = 0.0
    vad_filter: bool = True
    condition_on_previous_text: bool = True
