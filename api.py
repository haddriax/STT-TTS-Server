from __future__ import annotations

import gc
import logging
import os
import time

import torch
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional, Union

import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.exceptions import HTTPException as StarletteHTTPException, RequestValidationError
from fastapi.responses import JSONResponse, PlainTextResponse

from audio2face_client import Audio2FaceClient, Audio2FaceError
from config import DEFAULT_CONFIG_PATH, AppConfig, load_config
from healthcheck import run_checks
from kokoro_tts import KokoroTTS, LipSyncRequest, TtsResponse
from occ_emotion import OccClassifyRequest, OccClassifyResponse, OccEmotionClient
from whisper_stt import SUPPORTED_FORMATS, WhisperSTT, build_response

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

config_path = Path(os.getenv("WHISPER_CONFIG", DEFAULT_CONFIG_PATH))
cfg = load_config(config_path)


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    mc = cfg.model
    sc = cfg.server
    kc = cfg.kokoro

    active_routes = []
    if kc.activate_base_arkit:
        active_routes.append("/tts/arkit")
    if kc.activate_words:
        active_routes.append("/tts/words")
    if kc.activate_audio2face:
        active_routes.append("/tts/audio2face")

    logger.info("=" * 60)
    logger.info("whispeer-server starting up")
    logger.info("  Config file : %s", config_path if config_path.exists() else f"{config_path} (not found, using defaults)")
    logger.info("  Listen      : http://%s:%d", sc.host, sc.port)
    logger.info("  Model       : %s  [%s / %s]", mc.name, mc.device, mc.compute_type)
    logger.info("  Kokoro      : voice=%s  routes=%s", kc.default_voice, ", ".join(active_routes) or "(none)")
    logger.info("  Language    : %s → stt=%s  tts=%s  voice=%s",
                cfg.language, cfg.transcribe.language, kc.lang_code, kc.default_voice)
    logger.info("=" * 60)

    app.state.stt = WhisperSTT(cfg.model)
    app.state.tts = KokoroTTS(cfg.kokoro)

    if kc.activate_audio2face:
        app.state.a2f = Audio2FaceClient(cfg.audio2face)
    else:
        app.state.a2f = None

    if cfg.occ.enabled:
        app.state.occ = OccEmotionClient(cfg.occ)
        logger.info("  OCC emotion: enabled (mode=%s)", cfg.occ.mode)
    else:
        app.state.occ = None

    app.state.cfg = cfg

    await run_checks(app, cfg)

    logger.info(
        "Endpoints: POST /stt/transcribe  POST /stt/translate%s  GET /health",
        "".join(f"  POST {r}" for r in active_routes),
    )
    logger.info("  API docs -> http://%s:%d/docs", sc.host, sc.port)

    gc.collect()
    torch.cuda.empty_cache()

    yield

    logger.info("Shutting down — releasing models...")
    app.state.stt = None
    app.state.tts = None
    app.state.a2f = None
    app.state.occ = None
    gc.collect()
    torch.cuda.empty_cache()
    logger.info("Shutdown complete.")


app = FastAPI(title="whispeer-server", version="0.1.0", lifespan=lifespan)


# ---------------------------------------------------------------------------
# Error handlers
# ---------------------------------------------------------------------------

_ROUTES = [
    ("GET",  "/health",         "Server health check"),
    ("GET",  "/models",         "List loaded Whisper model"),
    ("POST", "/stt/transcribe", "Transcribe audio to text"),
    ("POST", "/stt/translate",  "Transcribe audio and translate to English"),
]
if cfg.kokoro.activate_base_arkit:
    _ROUTES.append(("POST", "/tts/arkit",      "Generate speech + ARKit blendshapes (viseme pipeline)"))
if cfg.kokoro.activate_words:
    _ROUTES.append(("POST", "/tts/words",      "Generate speech + word-level timestamps"))
if cfg.kokoro.activate_audio2face:
    _ROUTES.append(("POST", "/tts/audio2face", "Generate speech + ARKit blendshapes (Audio2Face-3D)"))
if cfg.occ.enabled:
    _ROUTES.append(("POST", "/emotion/classify", "Classify text into an OCC emotion label"))

@app.exception_handler(StarletteHTTPException)
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    if exc.status_code == 404:
        return JSONResponse(
            status_code=404,
            content={
                "error": f"Route '{request.method} {request.url.path}' not found.",
                "available_routes": [
                    {"method": m, "path": p, "description": d} for m, p, d in _ROUTES
                ],
                "docs": str(request.url.replace(path="/docs", query="")),
            },
        )
    logger.error("HTTP %d — %s %s — %s", exc.status_code, request.method, request.url.path, exc.detail)
    return JSONResponse(status_code=exc.status_code, content={"error": exc.detail})


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model": cfg.model.name}


@app.get("/models")
def list_models() -> dict:
    return {"data": [{"id": cfg.model.name, "object": "model"}]}


@app.post("/stt/transcribe", response_model=None)
async def transcriptions(
    file: UploadFile = File(...),
    language: Optional[str] = Form(None),
    prompt: Optional[str] = Form(None),
    response_format: str = Form("json"),
    temperature: Optional[float] = Form(None),
) -> Union[JSONResponse, PlainTextResponse]:
    if response_format not in SUPPORTED_FORMATS:
        raise HTTPException(
            status_code=400,
            detail={"error": {"message": f"Unsupported response_format '{response_format}'. Must be one of: {sorted(SUPPORTED_FORMATS)}", "type": "invalid_request_error"}},
        )
    audio_bytes = await file.read()
    logger.info("STT transcribe — %d bytes", len(audio_bytes))
    t0 = time.perf_counter()
    try:
        segments_list, info = await app.state.stt.transcribe(
            audio_bytes, app.state.cfg.transcribe, language, prompt, temperature
        )
    except Exception as exc:
        logger.exception("STT transcription error")
        raise HTTPException(
            status_code=422,
            detail={"error": {"message": f"Audio processing failed: {exc}", "type": "invalid_request_error"}},
        ) from exc
    emotion = None
    if app.state.occ and response_format in ("json", "verbose_json"):
        full_text = "".join(seg.text for seg in segments_list)
        emotion = await app.state.occ.classify(full_text)
    logger.info("STT transcribe — done in %dms", int((time.perf_counter() - t0) * 1000))
    return build_response(response_format, "transcribe", segments_list, info, emotion=emotion)


@app.post("/stt/translate", response_model=None)
async def translations(
    file: UploadFile = File(...),
    prompt: Optional[str] = Form(None),
    response_format: str = Form("json"),
    temperature: Optional[float] = Form(None),
) -> Union[JSONResponse, PlainTextResponse]:
    if response_format not in SUPPORTED_FORMATS:
        raise HTTPException(
            status_code=400,
            detail={"error": {"message": f"Unsupported response_format '{response_format}'. Must be one of: {sorted(SUPPORTED_FORMATS)}", "type": "invalid_request_error"}},
        )
    audio_bytes = await file.read()
    logger.info("STT translate — %d bytes", len(audio_bytes))
    t0 = time.perf_counter()
    try:
        segments_list, info = await app.state.stt.translate(
            audio_bytes, app.state.cfg.translate, prompt, temperature
        )
    except Exception as exc:
        logger.exception("STT translation error")
        raise HTTPException(
            status_code=422,
            detail={"error": {"message": f"Audio processing failed: {exc}", "type": "invalid_request_error"}},
        ) from exc
    emotion = None
    if app.state.occ and response_format in ("json", "verbose_json"):
        full_text = "".join(seg.text for seg in segments_list)
        emotion = await app.state.occ.classify(full_text)
    logger.info("STT translate — done in %dms", int((time.perf_counter() - t0) * 1000))
    return build_response(response_format, "translate", segments_list, info, emotion=emotion)


if cfg.kokoro.activate_base_arkit:
    @app.post("/tts/arkit", response_model=TtsResponse)
    async def tts_arkit(req: LipSyncRequest, request: Request) -> JSONResponse:
        logger.info("TTS arkit — %d chars", len(req.text))
        t0 = time.perf_counter()
        result = await request.app.state.tts.lipsync_arkit(req)
        logger.info("TTS arkit — done in %dms", int((time.perf_counter() - t0) * 1000))
        return JSONResponse(result.model_dump(exclude_none=True))

if cfg.kokoro.activate_words:
    @app.post("/tts/words", response_model=TtsResponse, deprecated=True)
    async def tts_words(req: LipSyncRequest, request: Request) -> JSONResponse:
        logger.info("TTS words — %d chars", len(req.text))
        t0 = time.perf_counter()
        result = await request.app.state.tts.lipsync_words(req)
        logger.info("TTS words — done in %dms", int((time.perf_counter() - t0) * 1000))
        return JSONResponse(result.model_dump(exclude_none=True))

if cfg.kokoro.activate_audio2face:
    @app.post("/tts/audio2face", response_model=TtsResponse)
    async def tts_audio2face(req: LipSyncRequest, request: Request) -> JSONResponse:
        logger.info("TTS audio2face — %d chars", len(req.text))
        t0 = time.perf_counter()
        try:
            result = await request.app.state.tts.lipsync_audio2face(
                req, request.app.state.a2f
            )
        except Audio2FaceError as exc:
            logger.exception("Audio2Face error")
            raise HTTPException(status_code=502, detail=str(exc))
        logger.info("TTS audio2face — done in %dms", int((time.perf_counter() - t0) * 1000))
        return JSONResponse(result.model_dump(exclude_none=True))

if cfg.occ.enabled:
    @app.post("/emotion/classify", response_model=OccClassifyResponse)
    async def emotion_classify(req: OccClassifyRequest, request: Request) -> JSONResponse:
        logger.info("OCC classify — %d chars", len(req.text))
        t0 = time.perf_counter()
        result = await request.app.state.occ.classify_full(req.text)
        logger.info("OCC classify — %s in %dms", result.emotion, int((time.perf_counter() - t0) * 1000))
        return JSONResponse(result.model_dump())


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    uvicorn.run(
        app,
        host=cfg.server.host,
        port=cfg.server.port,
        log_level=cfg.server.log_level,
    )
