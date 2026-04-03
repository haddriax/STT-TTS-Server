"""
Post-startup health checks for all API routes.

Called from the lifespan in api.py after both Whisper and Kokoro are loaded.
Uses httpx.ASGITransport to call the app in-process — no network, no port needed.
"""
from __future__ import annotations

import asyncio
import logging

import httpx

logger = logging.getLogger("healthcheck")



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


async def run_checks(app, cfg) -> None:
    """Run a smoke-test against every active route and log the results."""
    logger.info("Running startup health checks...")

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test", timeout=30.0) as client:
        checks = [
            _check(client, "GET",  "/health",         "health"),
            _check(client, "GET",  "/models",         "models"),
        ]
        if cfg.kokoro.activate_base_arkit:
            checks.append(_check(client, "POST", "/tts/arkit",      "tts/arkit",      json={"text": "hello"}))
        if cfg.kokoro.activate_words:
            checks.append(_check(client, "POST", "/tts/words",      "tts/words",      json={"text": "hello"}))
        if cfg.kokoro.activate_audio2face:
            checks.append(_check(client, "POST", "/tts/audio2face", "tts/audio2face", json={"text": "hello"}))

        results = list(await asyncio.gather(*checks))

    passed, total = sum(results), len(results)
    if passed == total:
        logger.info("Health checks passed: %d/%d", passed, total)
    else:
        logger.warning("Health checks: %d/%d passed — some routes may not be working.", passed, total)
