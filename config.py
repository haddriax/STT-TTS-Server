from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field

from occ.config import OccConfig
from stt.config import ModelConfig, TranscribeConfig, TranslateConfig
from tts.config import IPA_VISEME_PATH_MAP, KokoroConfig

logger = logging.getLogger(__name__)

DEFAULT_CONFIG_PATH = Path(__file__).parent / "config.yaml"

# BCP-47 language code -> (kokoro_lang_code, kokoro_default_voice)
_LANG_MAP: dict[str, tuple[str, str]] = {
    "fr":    ("f", "ff_siwis"),
    "en":    ("a", "am_adam"),
    "en-gb": ("b", "bf_emma"),
}


# ---------------------------------------------------------------------------
# Config schemas
# ---------------------------------------------------------------------------

class ServerConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8000
    log_level: str = "info"


class AppConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")

    language: str = "fr"                         # BCP-47: fr | en | en-gb
    server: ServerConfig = Field(default_factory=ServerConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    transcribe: TranscribeConfig = Field(default_factory=TranscribeConfig)
    translate: TranslateConfig = Field(default_factory=TranslateConfig)
    kokoro: KokoroConfig = Field(default_factory=KokoroConfig)
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
        cfg.kokoro.ipa_to_viseme_path = IPA_VISEME_PATH_MAP.get(
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
