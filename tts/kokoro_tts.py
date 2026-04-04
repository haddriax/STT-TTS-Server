from __future__ import annotations

import asyncio
import json
import logging
import re as _re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from pydantic import BaseModel

from tts.arkit import (
    BlendShapeFrame,
    _SAMPLE_RATE,
    _encode_wav_base64,
    _estimate_word_timestamps,
    _generate_arkit_frames,
    _load_phoneme_durations,
    _load_phoneme_mapping,
    _load_viseme_pipeline,
    _rescale_model_timestamps,
    _tokenize_ipa,
    _trim_trailing_silence,
)
from tts.config import KokoroConfig

logger = logging.getLogger(__name__)


def clean_llm_text(text: str) -> str:
    """Strip common LLM/markdown artefacts from text before TTS synthesis."""
    # Markdown links: keep label, drop URL
    text = _re.sub(r'\[([^\]]+)\]\([^)]*\)', r'\1', text)
    # Code spans: keep content, drop backticks
    text = _re.sub(r'`+([^`]*)`+', r'\1', text)
    # Bold/italic markers: ***, **, *, ___, __, _
    text = _re.sub(r'\*{1,3}([^*]*)\*{1,3}', r'\1', text)
    text = _re.sub(r'_{1,3}([^_]*)_{1,3}', r'\1', text)
    # Remaining lone asterisks / underscores (orphaned markers)
    text = _re.sub(r'(?<!\w)[*_]+(?!\w)', '', text)
    # Markdown headers: leading # symbols
    text = _re.sub(r'^\s*#{1,6}\s+', '', text, flags=_re.MULTILINE)
    # List bullets: leading -, *, •
    text = _re.sub(r'^\s*[-*•]\s+', '', text, flags=_re.MULTILINE)
    # Collapse excess whitespace
    text = _re.sub(r' {2,}', ' ', text)
    text = _re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


# ---------------------------------------------------------------------------
# Request / response schemas
# ---------------------------------------------------------------------------

class LipSyncRequest(BaseModel):
    text: str
    voice: Optional[str] = None   # falls back to cfg.default_voice
    speed: Optional[float] = None  # falls back to cfg.default_speed


class WordTimestamp(BaseModel):
    word: str
    start_time: float
    end_time: float


class TtsResponse(BaseModel):
    audio: str                                      # base64-encoded wav
    format: str
    duration: float
    frames: Optional[list[BlendShapeFrame]] = None  # arkit / audio2face modes
    timestamps: Optional[list[WordTimestamp]] = None  # words mode (deprecated)


# ---------------------------------------------------------------------------
# KokoroTTS class
# ---------------------------------------------------------------------------

class KokoroTTS:
    def __init__(self, cfg: KokoroConfig) -> None:
        from kokoro import KPipeline
        logger.info("Loading Kokoro pipeline (lang_code=%s)...", cfg.lang_code)
        self.pipeline = KPipeline(lang_code=cfg.lang_code, device=cfg.device)
        self.cfg = cfg
        if cfg.use_viseme_pipeline:
            _load_viseme_pipeline(cfg.ipa_to_viseme_path, cfg.viseme_to_arkit_path)
        else:
            _load_phoneme_mapping(cfg.phoneme_mapping_path)
        _load_phoneme_durations(cfg.phoneme_durations_path)
        logger.info("Kokoro pipeline ready.")

    @dataclass
    class SynthResult:
        audio: np.ndarray
        phoneme_parts: list[str]
        token_timestamps: list[dict] | None  # from model pred_dur

    def _synthesize(self, text: str, voice: str, speed: float) -> "KokoroTTS.SynthResult":
        """Run KPipeline synchronously; returns audio, phoneme strings, and model timestamps."""
        audio_chunks: list[np.ndarray] = []
        phoneme_parts: list[str] = []
        token_timestamps: list[dict] = []
        has_timestamps = True
        cumulative_offset = 0.0  # seconds of audio from previous chunks

        for result in self.pipeline(text, voice=voice, speed=speed):
            audio = result.audio
            if hasattr(audio, 'cpu'):
                audio = audio.cpu()
            chunk_np = audio.numpy() if hasattr(audio, 'numpy') else np.array(audio)
            audio_chunks.append(chunk_np)
            if result.phonemes:
                phoneme_parts.append(result.phonemes)
            if result.tokens:
                for tok in result.tokens:
                    if tok.phonemes and tok.start_ts is not None and tok.end_ts is not None:
                        token_timestamps.append({
                            "word": tok.text,
                            "phonemes": tok.phonemes,
                            "start_time": tok.start_ts + cumulative_offset,
                            "end_time": tok.end_ts + cumulative_offset,
                        })
            else:
                has_timestamps = False
            cumulative_offset += len(chunk_np) / _SAMPLE_RATE

        combined = np.concatenate(audio_chunks) if audio_chunks else np.zeros(0, dtype=np.float32)
        return KokoroTTS.SynthResult(
            audio=combined,
            phoneme_parts=phoneme_parts,
            token_timestamps=token_timestamps if has_timestamps and token_timestamps else None,
        )

    @dataclass
    class _Prepared:
        audio_b64: str
        duration: float
        raw_audio: np.ndarray
        phonemes: list[str]
        word_timestamps: list[dict]

    async def _prepare(self, req: LipSyncRequest) -> "_Prepared":
        """Shared synthesis + timestamp logic used by all lipsync_* methods."""
        voice = req.voice or self.cfg.default_voice
        speed = req.speed or self.cfg.default_speed
        text = clean_llm_text(req.text)
        logger.info("TTS synthesize — %d chars  voice=%s  speed=%.2f", len(text), voice, speed)

        synth = await asyncio.to_thread(self._synthesize, text, voice, speed)
        synth.audio = _trim_trailing_silence(synth.audio, _SAMPLE_RATE)
        duration = len(synth.audio) / _SAMPLE_RATE
        audio_b64 = _encode_wav_base64(synth.audio, self.cfg.output_sample_rate)
        logger.info("TTS synthesize — %.2fs audio", duration)

        if synth.token_timestamps:
            word_timestamps = _rescale_model_timestamps(synth.token_timestamps, duration)
            phonemes = _tokenize_ipa(" ".join(synth.phoneme_parts))
            logger.info("lipsync: using model-predicted timestamps (%d tokens)", len(word_timestamps))
        else:
            words = text.split()
            phonemes = _tokenize_ipa(" ".join(synth.phoneme_parts))
            word_timestamps = _estimate_word_timestamps(words, duration, phonemes)
            logger.info("lipsync: using estimated timestamps (no model data)")

        return KokoroTTS._Prepared(
            audio_b64=audio_b64,
            duration=duration,
            raw_audio=synth.audio,
            phonemes=phonemes,
            word_timestamps=word_timestamps,
        )

    async def lipsync_arkit(self, req: LipSyncRequest) -> TtsResponse:
        p = await self._prepare(req)
        frames = _generate_arkit_frames(p.phonemes, p.word_timestamps, self.cfg.fps, self.cfg.arkit_level)
        logger.info(
            "lipsync_arkit: %d words, %d phonemes, %d frames @ %d fps",
            len(p.word_timestamps), len(p.phonemes), len(frames), self.cfg.fps,
        )
        result = TtsResponse(audio=p.audio_b64, format="wav", duration=p.duration, frames=frames)
        if self.cfg.debug_dump_arkit:
            self._dump_arkit(req.text, result)
        return result

    def _dump_arkit(self, text: str, result: TtsResponse) -> None:
        import datetime
        tmp = Path(__file__).parent / "tmp"
        tmp.mkdir(exist_ok=True)
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        slug = _re.sub(r"[^\w]+", "_", text[:40]).strip("_")
        path = tmp / f"{ts}_{slug}.json"
        payload = result.model_dump(exclude_none=True)
        payload.pop("audio", None)  # omit base64 blob — not useful for visualisation
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        logger.debug("debug_dump_arkit: wrote %s", path)

    async def lipsync_words(self, req: LipSyncRequest) -> TtsResponse:
        p = await self._prepare(req)
        return TtsResponse(
            audio=p.audio_b64,
            format="wav",
            duration=p.duration,
            timestamps=[
                WordTimestamp(**{k: v for k, v in t.items() if k in ("word", "start_time", "end_time")})
                for t in p.word_timestamps
            ],
        )
