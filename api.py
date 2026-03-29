from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional, Union

import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.exceptions import HTTPException as StarletteHTTPException, RequestValidationError
from fastapi.responses import JSONResponse, PlainTextResponse

from config import DEFAULT_CONFIG_PATH, AppConfig, load_config
from healthcheck import run_checks
from kokoro_tts import KokoroTTS, LipSyncRequest
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

    logger.info("=" * 60)
    logger.info("whispeer-server starting up")
    logger.info("  Config file : %s", config_path if config_path.exists() else f"{config_path} (not found, using defaults)")
    logger.info("  Listen      : http://%s:%d", sc.host, sc.port)
    logger.info("  Model       : %s  [%s / %s]", mc.name, mc.device, mc.compute_type)
    logger.info("  Kokoro      : voice=%s  mode=%s  fps=%s", cfg.kokoro.default_voice, cfg.kokoro.output_mode, cfg.kokoro.fps)
    logger.info("=" * 60)

    app.state.stt = WhisperSTT(cfg.model)
    app.state.tts = KokoroTTS(cfg.kokoro)
    app.state.cfg = cfg

    await run_checks(app)

    logger.info(
        "Endpoints: POST /stt/transcribe  POST /stt/translate  POST /tts/lipsync  GET /health"
    )
    print(f"\n  API docs -> http://{sc.host}:{sc.port}/docs\n")

    yield

    logger.info("Shutting down.")


app = FastAPI(title="whispeer-server", version="0.1.0", lifespan=lifespan)


# ---------------------------------------------------------------------------
# Error handlers
# ---------------------------------------------------------------------------

_ROUTES = [
    ("GET",  "/health",         "Server health check"),
    ("GET",  "/models",         "List loaded Whisper model"),
    ("POST", "/stt/transcribe", "Transcribe audio to text"),
    ("POST", "/stt/translate",  "Transcribe audio and translate to English"),
    ("POST", "/tts/lipsync",    "Generate speech + lip-sync animation from text"),
]

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
    try:
        segments_list, info = await app.state.stt.transcribe(
            audio_bytes, app.state.cfg.transcribe, language, prompt, temperature
        )
    except Exception as exc:
        raise HTTPException(
            status_code=422,
            detail={"error": {"message": f"Audio processing failed: {exc}", "type": "invalid_request_error"}},
        ) from exc
    return build_response(response_format, "transcribe", segments_list, info)


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
    try:
        segments_list, info = await app.state.stt.translate(
            audio_bytes, app.state.cfg.translate, prompt, temperature
        )
    except Exception as exc:
        raise HTTPException(
            status_code=422,
            detail={"error": {"message": f"Audio processing failed: {exc}", "type": "invalid_request_error"}},
        ) from exc
    return build_response(response_format, "translate", segments_list, info)


@app.post("/tts/lipsync", response_model=None)
async def lipsync(req: LipSyncRequest) -> JSONResponse:
    result = await app.state.tts.lipsync(req)
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
