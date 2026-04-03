from __future__ import annotations

from pydantic import BaseModel, Field

# kokoro lang_code -> IPA-to-viseme mapping file
IPA_VISEME_PATH_MAP: dict[str, str] = {
    "a": "ipa_to_viseme.json",     # American English
    "b": "ipa_to_viseme.json",     # British English
    "f": "ipa_to_viseme_fr.json",  # French
}


class KokoroConfig(BaseModel):
    device: str = "auto"
    lang_code: str = "a"
    default_voice: str = "am_adam"
    default_speed: float = 1.0
    activate_base_arkit: bool = True
    activate_words: bool = True
    fps: int = 60
    output_sample_rate: int = 48000
    phoneme_mapping_path: str = "phoneme_to_arkit.json"
    ipa_to_viseme_path: str = "ipa_to_viseme.json"
    viseme_to_arkit_path: str = "viseme_to_arkit.json"
    use_viseme_pipeline: bool = True
    phoneme_durations_path: str = "phoneme_durations.json"
    arkit_level: int = Field(default=2, ge=1, le=3)  # 1=basic  2=advanced  3=full suit
    debug_dump_arkit: bool = False  # dump each /tts/arkit response as JSON to ./tmp/
