#!/usr/bin/env python
"""
whispeer-server entry point.
"""
# ---------------------------------------------------------------------------
# Warning filters — must be set before any import that triggers them.
# pkg_resources fires on ctranslate2 import (happens inside `from api import app`).
# Torch warnings fire at Kokoro model load (lifespan), but early placement is fine.
# ---------------------------------------------------------------------------
import warnings
warnings.filterwarnings("ignore", message="pkg_resources is deprecated", category=UserWarning)
warnings.filterwarnings("ignore", message="dropout option adds dropout", category=UserWarning)
warnings.filterwarnings("ignore", message=".*weight_norm.*", category=FutureWarning)

import logging
import os
import site
import sys

# --- FIX FOR WINDOWS CUDA DLL LOADING ---
if sys.platform == "win32":
    venv_paths = site.getsitepackages() + [site.getusersitepackages()]

    for pkg in venv_paths:
        cublas_path = os.path.join(pkg, "nvidia", "cublas", "bin")
        cudnn_path = os.path.join(pkg, "nvidia", "cudnn", "bin")

        if os.path.exists(cublas_path):
            os.add_dll_directory(cublas_path)
            os.environ["PATH"] = cublas_path + os.pathsep + os.environ["PATH"]

        if os.path.exists(cudnn_path):
            os.add_dll_directory(cudnn_path)
            os.environ["PATH"] = cudnn_path + os.pathsep + os.environ["PATH"]
# ----------------------------------------

import uvicorn
from api import app, cfg  # cfg reused from api — no second load_config() call

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# HuggingFace token — set HF_TOKEN in your environment or .env file.
# Enables higher rate limits and faster downloads from the HF Hub.
# ---------------------------------------------------------------------------
if hf_token := os.getenv("HF_TOKEN"):
    from huggingface_hub import login
    login(token=hf_token, add_to_git_credential=False)
    logger.info("HuggingFace: authenticated via HF_TOKEN")
else:
    logger.debug("HuggingFace: no HF_TOKEN set — using anonymous access")

if __name__ == "__main__":
    try:
        uvicorn.run(
            app,
            host=cfg.server.host,
            port=cfg.server.port,
            log_level=cfg.server.log_level,
        )
    except KeyboardInterrupt:
        pass
