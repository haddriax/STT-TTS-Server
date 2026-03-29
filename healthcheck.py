"""
Post-startup health checks for all API routes.

Called from the lifespan in api.py after both Whisper and Kokoro are loaded.
Uses httpx.ASGITransport to call the app in-process — no network, no port needed.
"""
from __future__ import annotations

import io
import logging
import wave

import httpx

logger = logging.getLogger("healthcheck")


def _silent_wav(duration_s: float = 0.1, sample_rate: int = 16000) -> bytes:
    """Generate a minimal silent WAV file for STT endpoint checks."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(b"\x00\x00" * int(sample_rate * duration_s))
    return buf.getvalue()


async def _check(client: httpx.AsyncClient, method: str, path: str, label: str, **kwargs) -> bool:
    try:
        resp = await client.request(method, path, **kwargs)
        ok = resp.status_code < 400
        level = logger.info if ok else logger.warning
        level("  [%s] %s %s → %d", "OK  " if ok else "FAIL", method, path, resp.status_code)
        if not ok:
            logger.warning("       response: %s", resp.text[:200])
        return ok
    except Exception as exc:
        logger.error("  [FAIL] %s %s → %s", method, path, exc)
        return False


async def run_checks(app) -> None:
    """Run a smoke-test against every route and log the results."""
    logger.info("Running startup health checks...")
    silent = _silent_wav()

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test", timeout=30.0) as client:
        results = [
            await _check(client, "GET",  "/health",        "health"),
            await _check(client, "GET",  "/models",        "models"),
            await _check(client, "POST", "/stt/transcribe", "stt/transcribe",
                files={"file": ("check.wav", silent, "audio/wav")}),
            await _check(client, "POST", "/stt/translate",  "stt/translate",
                files={"file": ("check.wav", silent, "audio/wav")}),
            await _check(client, "POST", "/tts/lipsync",    "tts/lipsync",
                json={"text": "hello"}),
        ]

    passed, total = sum(results), len(results)
    if passed == total:
        logger.info("Health checks passed: %d/%d", passed, total)
    else:
        logger.warning("Health checks: %d/%d passed — some routes may not be working.", passed, total)
