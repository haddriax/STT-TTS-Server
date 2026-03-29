#!/usr/bin/env python
"""
whispeer-server entry point.
"""
import os
import sys
import site
from pathlib import Path

# --- FIX FOR WINDOWS CUDA DLL LOADING ---
if sys.platform == "win32":
    # 1. Expand Python's DLL directory search
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
from api import app
from config import DEFAULT_CONFIG_PATH, load_config

cfg = load_config(Path(os.getenv("WHISPER_CONFIG", DEFAULT_CONFIG_PATH)))

if __name__ == "__main__":
    uvicorn.run(
        app,
        host=cfg.server.host,
        port=cfg.server.port,
        log_level=cfg.server.log_level,
    )
