from __future__ import annotations

from typing import Optional

from pydantic import BaseModel


class OccConfig(BaseModel):
    enabled: bool = False
    mode: str = "ollama"                    # "ollama" | "lora"
    ollama_url: str = "http://localhost:11434"
    ollama_model: str = "mistral"
    lora_model_path: str = ""               # path to LoRA adapter dir (mode="lora" only)
    max_vram_gb: Optional[float] = None     # cap VRAM for LoRA model; excess layers spill to RAM
