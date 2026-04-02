from __future__ import annotations

import asyncio
import io
import logging
from typing import Any, Optional, Union

from fastapi.responses import JSONResponse, PlainTextResponse
from faster_whisper import WhisperModel
from pydantic import BaseModel

from config import ModelConfig, TranscribeConfig, TranslateConfig

logger = logging.getLogger(__name__)

# Kokoro single-letter lang codes → BCP-47, in case a client sends the wrong code
_KOKORO_TO_BCP47: dict[str, str] = {"a": "en", "b": "en", "f": "fr"}


def _normalize_language(lang: Optional[str]) -> Optional[str]:
    if lang is None:
        return None
    normalized = _KOKORO_TO_BCP47.get(lang.lower(), lang)
    if normalized != lang:
        logger.warning("Language code %r is a Kokoro code — using %r for Whisper", lang, normalized)
    return normalized

SUPPORTED_FORMATS = {"json", "verbose_json", "text", "srt", "vtt"}


# ---------------------------------------------------------------------------
# Response schemas
# ---------------------------------------------------------------------------

class WhisperSegment(BaseModel):
    id: int
    start: float
    end: float
    text: str
    tokens: Optional[list[int]] = None
    temperature: Optional[float] = None
    avg_logprob: Optional[float] = None
    compression_ratio: Optional[float] = None
    no_speech_prob: Optional[float] = None


class AudioJsonResponse(BaseModel):
    text: str
    emotion: Optional[str] = None


class AudioVerboseJsonResponse(BaseModel):
    task: str
    language: str
    duration: float
    text: str
    segments: list[WhisperSegment]
    emotion: Optional[str] = None


# ---------------------------------------------------------------------------
# Format helpers
# ---------------------------------------------------------------------------

def _fmt_timestamp(seconds: float) -> str:
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int(round((seconds % 1) * 1000))
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def segments_to_srt(segments: list) -> str:
    lines = []
    for i, seg in enumerate(segments, start=1):
        lines.append(
            f"{i}\n{_fmt_timestamp(seg.start)} --> {_fmt_timestamp(seg.end)}\n{seg.text.strip()}\n"
        )
    return "\n".join(lines)


def segments_to_vtt(segments: list) -> str:
    lines = ["WEBVTT\n"]
    for seg in segments:
        start = _fmt_timestamp(seg.start).replace(",", ".")
        end = _fmt_timestamp(seg.end).replace(",", ".")
        lines.append(f"{start} --> {end}\n{seg.text.strip()}\n")
    return "\n".join(lines)


def build_response(
    response_format: str,
    task: str,
    segments_list: list,
    info: Any,
    emotion: Optional[str] = None,
) -> Union[JSONResponse, PlainTextResponse]:
    full_text = "".join(seg.text for seg in segments_list)

    if response_format == "text":
        return PlainTextResponse(full_text)
    if response_format == "srt":
        return PlainTextResponse(segments_to_srt(segments_list))
    if response_format == "vtt":
        return PlainTextResponse(segments_to_vtt(segments_list), media_type="text/vtt")
    if response_format == "verbose_json":
        resp = AudioVerboseJsonResponse(
            task=task,
            language=info.language,
            duration=info.duration,
            text=full_text,
            segments=[
                WhisperSegment(
                    id=seg.id,
                    start=seg.start,
                    end=seg.end,
                    text=seg.text,
                    tokens=seg.tokens,
                    temperature=seg.temperature,
                    avg_logprob=seg.avg_logprob,
                    compression_ratio=seg.compression_ratio,
                    no_speech_prob=seg.no_speech_prob,
                )
                for seg in segments_list
            ],
            emotion=emotion,
        )
        return JSONResponse(resp.model_dump(exclude_none=True))
    return JSONResponse(AudioJsonResponse(text=full_text, emotion=emotion).model_dump(exclude_none=True))


# ---------------------------------------------------------------------------
# WhisperSTT class
# ---------------------------------------------------------------------------

class WhisperSTT:
    def __init__(self, cfg: ModelConfig) -> None:
        logger.info("Loading Whisper model '%s' on %s (%s)...", cfg.name, cfg.device, cfg.compute_type)
        self.model = WhisperModel(cfg.name, device=cfg.device, compute_type=cfg.compute_type)
        self.cfg = cfg
        logger.info("Model '%s' ready.", cfg.name)

    def _run(
        self,
        audio_bytes: bytes,
        task: str,
        task_cfg: TranscribeConfig | TranslateConfig,
        language: Optional[str],
        prompt: Optional[str],
        temperature: Optional[float],
    ) -> tuple:
        kwargs: dict[str, Any] = {
            "task": task,
            "beam_size": task_cfg.beam_size,
            "temperature": temperature if temperature is not None else task_cfg.temperature,
            "vad_filter": task_cfg.vad_filter,
            "condition_on_previous_text": task_cfg.condition_on_previous_text,
            "initial_prompt": prompt,
        }
        if task == "transcribe":
            kwargs["language"] = _normalize_language(language if language is not None else getattr(task_cfg, "language", None))

        segments_gen, info = self.model.transcribe(io.BytesIO(audio_bytes), **kwargs)
        segments = list(segments_gen)
        word_count = sum(len(seg.text.split()) for seg in segments)
        logger.info(
            "Whisper %s — lang=%s (%.0f%%) %.2fs audio %d segs %d words",
            task, info.language, info.language_probability * 100,
            info.duration, len(segments), word_count,
        )
        return segments, info

    async def transcribe(
        self,
        audio_bytes: bytes,
        task_cfg: TranscribeConfig,
        language: Optional[str] = None,
        prompt: Optional[str] = None,
        temperature: Optional[float] = None,
    ) -> tuple:
        return await asyncio.to_thread(
            self._run, audio_bytes, "transcribe", task_cfg, language, prompt, temperature
        )

    async def translate(
        self,
        audio_bytes: bytes,
        task_cfg: TranslateConfig,
        prompt: Optional[str] = None,
        temperature: Optional[float] = None,
    ) -> tuple:
        return await asyncio.to_thread(
            self._run, audio_bytes, "translate", task_cfg, None, prompt, temperature
        )
