"""Runtime helpers for optional system dependencies."""

from __future__ import annotations

import logging
import os
from pathlib import Path
import shutil

logger = logging.getLogger("memvid-mcp")


def _prepend_path(path: str) -> None:
    if not path:
        return
    current = os.environ.get("PATH", "")
    if current.startswith(path + os.pathsep) or current.split(os.pathsep, 1)[0] == path:
        return
    os.environ["PATH"] = path + os.pathsep + current if current else path


def ensure_ffmpeg() -> None:
    """Ensure ffmpeg is available, optionally via imageio-ffmpeg."""
    if shutil.which("ffmpeg"):
        return

    try:
        import imageio_ffmpeg  # type: ignore
    except Exception:
        logger.info(
            "ffmpeg not found in PATH; install system ffmpeg or add the 'runtime' extra (imageio-ffmpeg)."
        )
        return

    try:
        ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()
    except Exception as exc:
        logger.warning("Failed to resolve ffmpeg via imageio-ffmpeg: %s", exc)
        return

    ffmpeg_dir = str(Path(ffmpeg_path).parent)
    _prepend_path(ffmpeg_dir)
    os.environ.setdefault("IMAGEIO_FFMPEG_EXE", ffmpeg_path)
    os.environ.setdefault("FFMPEG_BINARY", ffmpeg_path)
