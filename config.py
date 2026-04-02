from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Optional

import yaml
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

DEFAULT_CONFIG_PATH = Path(__file__).parent / "config.yaml"

# BCP-47 language code → (kokoro_lang_code, kokoro_default_voice)
_LANG_MAP: dict[str, tuple[str, str]] = {
    "fr":    ("f", "ff_siwis"),
    "en":    ("a", "am_adam"),
    "en-gb": ("b", "bf_emma"),
}

# kokoro lang_code → IPA-to-viseme mapping file
_IPA_VISEME_PATH_MAP: dict[str, str] = {
    "a": "ipa_to_viseme.json",     # American English
    "b": "ipa_to_viseme.json",     # British English
    "f": "ipa_to_viseme_fr.json",  # French
}


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
    activate_base_arkit: bool = True
    activate_words: bool = True
    activate_audio2face: bool = False
    fps: int = 60
    output_sample_rate: int = 48000
    phoneme_mapping_path: str = "phoneme_to_arkit.json"
    ipa_to_viseme_path: str = "ipa_to_viseme.json"
    viseme_to_arkit_path: str = "viseme_to_arkit.json"
    use_viseme_pipeline: bool = True
    phoneme_durations_path: str = "phoneme_durations.json"
    arkit_level: int = Field(default=2, ge=1, le=3)  # 1=basic  2=advanced  3=full suit
    debug_dump_arkit: bool = False  # dump each /tts/arkit response as JSON to ./tmp/


class Audio2FaceConfig(BaseModel):
    host: str = "localhost"
    port: int = 52000
    timeout: float = 30.0  # seconds — covers full bidirectional streaming call


class OccConfig(BaseModel):
    enabled: bool = False
    mode: str = "ollama"                    # "ollama" | "lora"
    ollama_url: str = "http://localhost:11434"
    ollama_model: str = "mistral"
    lora_model_path: str = ""               # path to LoRA adapter dir (mode="lora" only)
    max_vram_gb: Optional[float] = None     # cap VRAM for LoRA model; excess layers spill to RAM
                                            # e.g. 4.0 → keep 4 GB on GPU, rest on CPU RAM
                                            # null = no cap, fill VRAM first then spill automatically


class AppConfig(BaseModel):
    language: str = "fr"                         # BCP-47: fr | en | en-gb
    server: ServerConfig = Field(default_factory=ServerConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    transcribe: TranscribeConfig = Field(default_factory=TranscribeConfig)
    translate: TranslateConfig = Field(default_factory=TranslateConfig)
    kokoro: KokoroConfig = Field(default_factory=KokoroConfig)
    audio2face: Audio2FaceConfig = Field(default_factory=Audio2FaceConfig)
    occ: OccConfig = Field(default_factory=OccConfig)


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

    # Derive per-component language settings from the top-level language key.
    # Per-component keys explicitly set in YAML always take precedence.
    lang = cfg.language.lower()
    kokoro_lang_code, kokoro_default_voice = _LANG_MAP.get(lang, (None, None))
    if kokoro_lang_code is None:
        logger.warning("language '%s' not in known codes %s — falling back to 'fr'", lang, list(_LANG_MAP))
        kokoro_lang_code, kokoro_default_voice = _LANG_MAP["fr"]

    if "language" not in (raw.get("transcribe") or {}):
        cfg.transcribe.language = lang
    if "lang_code" not in (raw.get("kokoro") or {}):
        cfg.kokoro.lang_code = kokoro_lang_code
    if "default_voice" not in (raw.get("kokoro") or {}):
        cfg.kokoro.default_voice = kokoro_default_voice
    if "ipa_to_viseme_path" not in (raw.get("kokoro") or {}):
        cfg.kokoro.ipa_to_viseme_path = _IPA_VISEME_PATH_MAP.get(
            cfg.kokoro.lang_code, "ipa_to_viseme.json"
        )

    # Env var overrides (take precedence over YAML)
    if v := os.getenv("WHISPER_MODEL"):
        logger.info("Config override: model.name = %r (WHISPER_MODEL)", v)
        cfg.model.name = v
    if v := os.getenv("WHISPER_DEVICE"):
        logger.info("Config override: model.device = %r (WHISPER_DEVICE)", v)
        cfg.model.device = v
    if v := os.getenv("WHISPER_COMPUTE"):
        logger.info("Config override: model.compute_type = %r (WHISPER_COMPUTE)", v)
        cfg.model.compute_type = v
    if v := os.getenv("WHISPER_HOST"):
        logger.info("Config override: server.host = %r (WHISPER_HOST)", v)
        cfg.server.host = v
    if v := os.getenv("WHISPER_PORT"):
        logger.info("Config override: server.port = %r (WHISPER_PORT)", v)
        cfg.server.port = int(v)
    if v := os.getenv("WHISPER_LOG_LEVEL"):
        logger.info("Config override: server.log_level = %r (WHISPER_LOG_LEVEL)", v)
        cfg.server.log_level = v

    return cfg
