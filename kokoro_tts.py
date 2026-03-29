from __future__ import annotations

import asyncio
import base64
import io
import json
import logging
import math
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import numpy as np
from pydantic import BaseModel
from scipy.signal import resample_poly

from config import KokoroConfig

logger = logging.getLogger(__name__)


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


class WordsLipSyncResponse(BaseModel):
    audio: str          # base64-encoded wav
    format: str
    timestamps: list[WordTimestamp]


class BlendShapeFrame(BaseModel):
    time: float                    # seconds from audio start
    blendshapes: dict[str, float]  # ARKit name → weight 0.0–1.0


class ARKitLipSyncResponse(BaseModel):
    audio: str          # base64-encoded wav
    format: str
    duration: float
    frames: list[BlendShapeFrame]


# ---------------------------------------------------------------------------
# IPA → ARKit blend shape table (loaded from external JSON)
# ---------------------------------------------------------------------------

_REST: dict[str, float] = {}

# Populated by _load_phoneme_mapping() or _load_viseme_pipeline() during KokoroTTS init
_IPA_LOOKUP: dict[str, dict[str, float]] = {}
_IPA_SORTED: list[str] = []


def _resolve_path(path: str) -> Path:
    """Resolve a path relative to this file's directory if not absolute."""
    resolved = Path(path)
    if not resolved.is_absolute():
        resolved = Path(__file__).parent / resolved
    return resolved


def _load_phoneme_mapping(path: str) -> None:
    """Load grouped phoneme→ARKit mapping from JSON (legacy/flat mode)."""
    global _IPA_LOOKUP, _IPA_SORTED

    resolved = _resolve_path(path)
    with open(resolved, encoding="utf-8") as f:
        groups = json.load(f)

    mapping: dict[str, dict[str, float]] = {}
    for group in groups.values():
        bs = group["blendshapes"]
        for ph in group["phonemes"]:
            mapping[ph] = bs

    _IPA_LOOKUP = mapping
    _IPA_SORTED = sorted(_IPA_LOOKUP.keys(), key=len, reverse=True)
    logger.info("Loaded %d phoneme→ARKit mappings (flat) from %s", len(_IPA_LOOKUP), resolved)


def _load_viseme_pipeline(ipa_to_viseme_path: str, viseme_to_arkit_path: str) -> None:
    """Load two-stage IPA→viseme→ARKit mapping from JSON files."""
    global _IPA_LOOKUP, _IPA_SORTED

    ipa_path = _resolve_path(ipa_to_viseme_path)
    viseme_path = _resolve_path(viseme_to_arkit_path)

    with open(ipa_path, encoding="utf-8") as f:
        ipa_to_viseme: dict[str, str] = {
            k: v for k, v in json.load(f).items() if not k.startswith("_")
        }

    with open(viseme_path, encoding="utf-8") as f:
        viseme_to_arkit: dict[str, dict[str, float]] = {
            k: v for k, v in json.load(f).items() if not k.startswith("_")
        }

    mapping: dict[str, dict[str, float]] = {}
    for ipa_sym, viseme_id in ipa_to_viseme.items():
        if viseme_id in viseme_to_arkit:
            mapping[ipa_sym] = viseme_to_arkit[viseme_id]
        else:
            logger.warning("Viseme %s (for IPA '%s') not found in viseme_to_arkit", viseme_id, ipa_sym)

    _IPA_LOOKUP = mapping
    _IPA_SORTED = sorted(_IPA_LOOKUP.keys(), key=len, reverse=True)
    logger.info(
        "Loaded viseme pipeline: %d IPA→viseme, %d visemes→ARKit, %d total mappings",
        len(ipa_to_viseme), len(viseme_to_arkit), len(_IPA_LOOKUP),
    )


# ---------------------------------------------------------------------------
# IPA helpers
# ---------------------------------------------------------------------------

def _tokenize_ipa(ipa: str) -> list[str]:
    tokens: list[str] = []
    i = 0
    while i < len(ipa):
        matched = False
        for sym in _IPA_SORTED:
            if ipa[i:].startswith(sym):
                tokens.append(sym)
                i += len(sym)
                matched = True
                break
        if not matched:
            i += 1
    return tokens


def _lerp_weights(a: dict[str, float], b: dict[str, float], t: float) -> dict[str, float]:
    keys = set(a) | set(b)
    return {k: a.get(k, 0.0) * (1 - t) + b.get(k, 0.0) * t for k in keys}


def _distribute_phonemes_in_word(
    word_phonemes: list[str], w_start: float, w_end: float
) -> list[tuple[float, float, str]]:
    """Distribute phonemes within a word's time window using duration heuristics or equal spacing."""
    w_total = w_end - w_start
    if not word_phonemes or w_total <= 0:
        return []

    if _PHONEME_DURATIONS:
        raw = [_PHONEME_DURATIONS.get(s, _DEFAULT_PHONEME_DURATION) for s in word_phonemes]
        raw_sum = sum(raw) or 1.0
        timeline = []
        t_cursor = w_start
        for j, sym in enumerate(word_phonemes):
            ph_dur = w_total * raw[j] / raw_sum
            timeline.append((t_cursor, t_cursor + ph_dur, sym))
            t_cursor += ph_dur
        return timeline

    w_dur = w_total / len(word_phonemes)
    return [(w_start + j * w_dur, w_start + (j + 1) * w_dur, sym)
            for j, sym in enumerate(word_phonemes)]


def _generate_arkit_frames(
    phonemes: list[str],
    word_timestamps: list[dict],
    fps: int,
) -> list[BlendShapeFrame]:
    if not word_timestamps:
        return []

    total_duration = word_timestamps[-1]["end_time"]
    has_model_phonemes = "phonemes" in word_timestamps[0]

    phoneme_timeline: list[tuple[float, float, str]] = []

    if has_model_phonemes:
        # Model tokens: each word has its own phoneme string — use it directly
        for wt in word_timestamps:
            word_phonemes = _tokenize_ipa(wt.get("phonemes", ""))
            if word_phonemes:
                phoneme_timeline.extend(
                    _distribute_phonemes_in_word(word_phonemes, wt["start_time"], wt["end_time"])
                )
    else:
        # Fallback: distribute global phoneme list across words by character count
        n_phonemes = len(phonemes)
        n_words = len(word_timestamps)
        word_lengths = [len(w["word"]) for w in word_timestamps]
        total_chars = sum(word_lengths) or 1

        ph_idx = 0
        for wi, wt in enumerate(word_timestamps):
            share = max(1, round(n_phonemes * word_lengths[wi] / total_chars))
            if wi == n_words - 1:
                share = n_phonemes - ph_idx
            word_phonemes = phonemes[ph_idx: ph_idx + share]
            ph_idx += share
            if word_phonemes:
                phoneme_timeline.extend(
                    _distribute_phonemes_in_word(word_phonemes, wt["start_time"], wt["end_time"])
                )

    if not phoneme_timeline:
        return []

    TRANSITION = 0.05

    def _weights_at(t: float) -> dict[str, float]:
        for k, (ps, pe, sym) in enumerate(phoneme_timeline):
            if ps <= t < pe:
                current = _IPA_LOOKUP.get(sym, _REST)
                fade_start = pe - TRANSITION
                if t >= fade_start and k + 1 < len(phoneme_timeline):
                    next_w = _IPA_LOOKUP.get(phoneme_timeline[k + 1][2], _REST)
                    ease = (1 - math.cos((t - fade_start) / TRANSITION * math.pi)) / 2
                    return _lerp_weights(current, next_w, ease)
                return dict(current)
        return {}

    frames: list[BlendShapeFrame] = []
    tick = 1.0 / fps
    t = 0.0
    while t <= total_duration + tick:
        frames.append(BlendShapeFrame(time=round(t, 6), blendshapes=_weights_at(t)))
        t += tick
    return frames


# ---------------------------------------------------------------------------
# Audio helpers
# ---------------------------------------------------------------------------

_SAMPLE_RATE = 24000  # Kokoro outputs at 24 kHz


def _trim_trailing_silence(
    audio: np.ndarray,
    sample_rate: int,
    threshold: float = 0.01,
    tail_seconds: float = 0.06,
) -> np.ndarray:
    """Remove trailing silence so animation doesn't outlive audible speech."""
    above = np.where(np.abs(audio) > threshold)[0]
    if len(above) == 0:
        return audio
    last_loud = int(above[-1])
    end = min(last_loud + int(tail_seconds * sample_rate), len(audio))
    return audio[:end]

def _encode_wav_base64(audio: np.ndarray, output_sample_rate: int) -> str:
    """Encode a float32 numpy audio array to a base64 WAV string, resampling if needed."""
    if output_sample_rate != _SAMPLE_RATE:
        from math import gcd
        g = gcd(output_sample_rate, _SAMPLE_RATE)
        audio = resample_poly(audio, output_sample_rate // g, _SAMPLE_RATE // g).astype(np.float32)
    buf = io.BytesIO()
    pcm = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(output_sample_rate)
        wf.writeframes(pcm.tobytes())
    return base64.b64encode(buf.getvalue()).decode()


# Populated by _load_phoneme_durations() during KokoroTTS init
_PHONEME_DURATIONS: dict[str, float] = {}
_DEFAULT_PHONEME_DURATION = 0.060  # fallback for unmapped symbols


def _load_phoneme_durations(path: str) -> None:
    """Load per-phoneme typical durations from JSON."""
    global _PHONEME_DURATIONS
    resolved = _resolve_path(path)
    with open(resolved, encoding="utf-8") as f:
        _PHONEME_DURATIONS = {
            k: v for k, v in json.load(f).items() if not k.startswith("_")
        }
    logger.info("Loaded %d phoneme durations from %s", len(_PHONEME_DURATIONS), resolved)


def _estimate_word_duration(word_phonemes: list[str]) -> float:
    """Sum typical durations for a word's phonemes."""
    return sum(_PHONEME_DURATIONS.get(ph, _DEFAULT_PHONEME_DURATION) for ph in word_phonemes)


def _estimate_word_timestamps(
    words: list[str],
    duration: float,
    phonemes: list[str] | None = None,
) -> list[dict]:
    """Distribute word timestamps proportionally.

    If phonemes are provided and _PHONEME_DURATIONS is loaded, uses phoneme-level
    duration heuristics for better accuracy. Falls back to character-count
    distribution otherwise.
    """
    if phonemes and _PHONEME_DURATIONS:
        return _estimate_word_timestamps_phoneme(words, duration, phonemes)

    # Fallback: character-count proportional
    total_chars = sum(len(w) for w in words) or 1
    timestamps, t = [], 0.0
    for word in words:
        word_dur = duration * len(word) / total_chars
        timestamps.append({"word": word, "start_time": round(t, 3), "end_time": round(t + word_dur, 3)})
        t += word_dur
    return timestamps


def _estimate_word_timestamps_phoneme(
    words: list[str],
    duration: float,
    phonemes: list[str],
) -> list[dict]:
    """Distribute phonemes across words, then allocate time by phoneme duration sums."""
    n_phonemes = len(phonemes)
    n_words = len(words)
    word_lengths = [len(w) for w in words]
    total_chars = sum(word_lengths) or 1

    # Assign phonemes to words proportionally by character count
    word_phoneme_groups: list[list[str]] = []
    ph_idx = 0
    for wi in range(n_words):
        if wi == n_words - 1:
            share = n_phonemes - ph_idx
        else:
            share = max(1, round(n_phonemes * word_lengths[wi] / total_chars))
        word_phoneme_groups.append(phonemes[ph_idx: ph_idx + share])
        ph_idx += share

    # Compute raw durations per word from phoneme heuristics
    raw_durations = [_estimate_word_duration(wph) for wph in word_phoneme_groups]
    total_raw = sum(raw_durations) or 1.0

    # Scale to fit actual audio duration
    timestamps, t = [], 0.0
    for wi, word in enumerate(words):
        word_dur = duration * raw_durations[wi] / total_raw
        timestamps.append({"word": word, "start_time": round(t, 3), "end_time": round(t + word_dur, 3)})
        t += word_dur
    return timestamps


# ---------------------------------------------------------------------------
# Model timestamp helpers
# ---------------------------------------------------------------------------

def _rescale_model_timestamps(
    token_timestamps: list[dict],
    actual_duration: float,
) -> list[dict]:
    """Clamp model timestamps to actual audio duration.

    The model timestamps are already aligned with the audio — pred_dur drives
    synthesis directly. Trailing silence is trimmed *after* synthesis, so word
    timestamps in the middle of the audio are correct as-is. We only need to
    clamp the last token's end_time to avoid frames past the audio end.
    """
    if not token_timestamps:
        return []

    result = []
    for t in token_timestamps:
        if t["start_time"] >= actual_duration:
            break  # token is entirely in trimmed silence — drop it
        result.append({
            **t,
            "end_time": min(t["end_time"], actual_duration),
        })
    return result


# ---------------------------------------------------------------------------
# KokoroTTS class
# ---------------------------------------------------------------------------

class KokoroTTS:
    def __init__(self, cfg: KokoroConfig) -> None:
        from kokoro import KPipeline
        logger.info("Loading Kokoro pipeline (lang_code=%s)...", cfg.lang_code)
        self.pipeline = KPipeline(lang_code=cfg.lang_code)
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

    async def lipsync(self, req: LipSyncRequest) -> Union[WordsLipSyncResponse, ARKitLipSyncResponse]:
        voice = req.voice or self.cfg.default_voice
        speed = req.speed or self.cfg.default_speed

        synth = await asyncio.to_thread(
            self._synthesize, req.text, voice, speed
        )

        synth.audio = _trim_trailing_silence(synth.audio, _SAMPLE_RATE)
        duration = len(synth.audio) / _SAMPLE_RATE
        audio_b64 = _encode_wav_base64(synth.audio, self.cfg.output_sample_rate)

        # Use model timestamps when available, fall back to estimation
        if synth.token_timestamps:
            word_timestamps = _rescale_model_timestamps(synth.token_timestamps, duration)
            phonemes = _tokenize_ipa(" ".join(synth.phoneme_parts))
            logger.info("lipsync: using model-predicted timestamps (%d tokens)", len(word_timestamps))
        else:
            words = req.text.split()
            phonemes = _tokenize_ipa(" ".join(synth.phoneme_parts))
            word_timestamps = _estimate_word_timestamps(words, duration, phonemes)
            logger.info("lipsync: using estimated timestamps (no model data)")

        if self.cfg.output_mode == "words":
            return WordsLipSyncResponse(
                audio=audio_b64,
                format="wav",
                timestamps=[WordTimestamp(**{k: v for k, v in t.items() if k in ("word", "start_time", "end_time")}) for t in word_timestamps],
            )
        frames = _generate_arkit_frames(phonemes, word_timestamps, self.cfg.fps)
        logger.info(
            "lipsync: %d words, %d phonemes, %d frames @ %d fps",
            len(word_timestamps), len(phonemes), len(frames), self.cfg.fps,
        )
        return ARKitLipSyncResponse(
            audio=audio_b64,
            format="wav",
            duration=duration,
            frames=frames,
        )
