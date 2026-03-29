from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Literal, Optional

import yaml
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

DEFAULT_CONFIG_PATH = Path(__file__).parent / "config.yaml"


# ---------------------------------------------------------------------------
# Config schemas
# ---------------------------------------------------------------------------

class ServerConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8000
    log_level: str = "info"


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


class KokoroConfig(BaseModel):
    lang_code: str = "a"
    default_voice: str = "am_adam"
    default_speed: float = 1.0
    output_mode: Literal["arkit", "words"] = "arkit"
    fps: int = 60
    output_sample_rate: int = 48000
    phoneme_mapping_path: str = "phoneme_to_arkit.json"
    ipa_to_viseme_path: str = "ipa_to_viseme.json"
    viseme_to_arkit_path: str = "viseme_to_arkit.json"
    use_viseme_pipeline: bool = True
    phoneme_durations_path: str = "phoneme_durations.json"


class AppConfig(BaseModel):
    server: ServerConfig = Field(default_factory=ServerConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    transcribe: TranscribeConfig = Field(default_factory=TranscribeConfig)
    translate: TranslateConfig = Field(default_factory=TranslateConfig)
    kokoro: KokoroConfig = Field(default_factory=KokoroConfig)


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

def load_config(path: Path) -> AppConfig:
    """Load config from YAML, then apply env var overrides."""
    raw: dict[str, Any] = {}
    if path.exists():
        with path.open() as f:
            raw = yaml.safe_load(f) or {}
        logger.info("Loaded config from %s", path)
    else:
        logger.warning("Config file not found at %s, using defaults.", path)

    cfg = AppConfig.model_validate(raw)

    # Env var overrides (take precedence over YAML)
    if v := os.getenv("WHISPER_MODEL"):
        cfg.model.name = v
    if v := os.getenv("WHISPER_DEVICE"):
        cfg.model.device = v
    if v := os.getenv("WHISPER_COMPUTE"):
        cfg.model.compute_type = v
    if v := os.getenv("WHISPER_HOST"):
        cfg.server.host = v
    if v := os.getenv("WHISPER_PORT"):
        cfg.server.port = int(v)
    if v := os.getenv("WHISPER_LOG_LEVEL"):
        cfg.server.log_level = v

    return cfg
