"""incident_clip.py - Opt-In Incident Video Clip Processing and Delivery for FallGuard.

Key Specifications:
- Checks ffmpeg availability at startup.
- Limits concurrent clip processing to 1 via asyncio Semaphore.
- Joins independent browser segments via ffmpeg concat demuxer.
- Trims exactly to [T - 10 s, T + 10 s] relative to fall onset T.
- Scales video to at most 480p height, encodes H.264 (yuv420p, crf 28, +faststart).
- Strictly removes audio (-an).
- Enforces strict size ceiling (< 20 MB).
- Executes ffmpeg with argument list (never shell=True) and 40s timeout.
- Sends video to caregiver's Telegram chat via sendVideo multipart API (60s timeout, 1 retry).
- Never logs or exposes video bytes, bot tokens, or unmasked chat IDs.
"""

import asyncio
import logging
import os
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests

logger = logging.getLogger("fallguard.incident_clip")

# Concurrency semaphore limiting clip processing to 1 worker at a time
clip_processing_semaphore = asyncio.Semaphore(1)


def get_ffmpeg_path() -> Optional[str]:
    """Discover ffmpeg executable in system PATH or common local installations."""
    # 1. Standard PATH lookup
    path = shutil.which("ffmpeg")
    if path:
        return path

    # 2. Check environment variable override
    env_path = os.getenv("FFMPEG_PATH", "").strip()
    if env_path and os.path.isfile(env_path):
        return env_path

    # 3. Common Windows binary locations (including user tools)
    candidate_paths = [
        r"D:\tools\ffmpeg\bin\ffmpeg.exe",
        r"C:\tools\ffmpeg\bin\ffmpeg.exe",
        r"C:\ffmpeg\bin\ffmpeg.exe",
    ]
    for cand in candidate_paths:
        if os.path.isfile(cand):
            # Also add to os.environ["PATH"] so child processes find it
            bin_dir = os.path.dirname(cand)
            if bin_dir not in os.environ.get("PATH", ""):
                os.environ["PATH"] = bin_dir + os.pathsep + os.environ.get("PATH", "")
            return cand

    return None


def is_ffmpeg_available() -> bool:
    """Check if ffmpeg executable is installed and available."""
    return get_ffmpeg_path() is not None


def _sanitize_string(text: str, token: Optional[str] = None) -> str:
    """Sanitize bot token or sensitive patterns from log messages."""
    if not text:
        return ""
    tok = token or os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    if tok and tok in text:
        text = text.replace(tok, "[REDACTED]")
    return text


def mask_chat_id(chat_id: Optional[str]) -> str:
    """Return masked chat ID showing last 4 digits only (e.g. '...1585')."""
    if not chat_id:
        return ""
    s = str(chat_id).strip()
    if len(s) <= 4:
        return f"...{s}"
    return f"...{s[-4:]}"


async def process_incident_clip(
    segment_paths: List[str],
    segment_start_times: List[float],
    T_ms: float,
    output_path: str,
) -> None:
    """Normalise and join video segments, trim to [T-10s, T+10s], scale to <= 480p, and encode H.264 without audio.

    Args:
        segment_paths: Absolute paths to ordered valid WebM/MP4 segment files in temporary directory.
        segment_start_times: Starting epoch ms corresponding to each segment.
        T_ms: Moment when possible_fall was first triggered (epoch ms).
        output_path: Destination path for normalized MP4 output.
    """
    ffmpeg_bin = get_ffmpeg_path()
    if not ffmpeg_bin:
        raise RuntimeError("ffmpeg executable not found on server")

    if not segment_paths:
        raise ValueError("No video segments provided for incident clip")

    if len(segment_paths) != len(segment_start_times):
        raise ValueError("Segment paths and start times count mismatch")

    # Directory where segments reside
    work_dir = os.path.dirname(output_path)
    concat_list_file = os.path.join(work_dir, "concat.txt")

    # Write concat demuxer file using forward-slashed paths safe across platforms
    with open(concat_list_file, "w", encoding="utf-8") as f:
        for p in segment_paths:
            escaped = p.replace("\\", "/").replace("'", "'\\''")
            f.write(f"file '{escaped}'\n")

    # Calculate trim timestamps relative to the first segment's start time
    first_start_ms = float(segment_start_times[0])
    rel_T_sec = (float(T_ms) - first_start_ms) / 1000.0

    # 10s before to 10s after T
    start_trim_sec = max(0.0, rel_T_sec - 10.0)
    duration_sec = 20.0

    # Command argument list: never shell=True
    cmd = [
        ffmpeg_bin,
        "-y",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        concat_list_file,
        "-ss",
        f"{start_trim_sec:.3f}",
        "-t",
        f"{duration_sec:.3f}",
        "-vf",
        "scale=-2:'min(480,ih)'",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-crf",
        "28",
        "-preset",
        "fast",
        "-movflags",
        "+faststart",
        "-an",
        output_path,
    ]

    logger.info(
        f"Processing incident clip with ffmpeg: T_offset={rel_T_sec:.2f}s, "
        f"trim=[{start_trim_sec:.2f}s, {start_trim_sec + duration_sec:.2f}s], segments={len(segment_paths)}"
    )

    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )

    try:
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=40.0)
    except asyncio.TimeoutError:
        try:
            proc.kill()
            await proc.communicate()
        except Exception:
            pass
        raise TimeoutError("ffmpeg incident clip processing timed out after 40 seconds")

    if proc.returncode != 0:
        err_msg = stderr.decode("utf-8", errors="ignore")[-400:] if stderr else "Unknown error"
        raise RuntimeError(f"ffmpeg conversion failed (code {proc.returncode}): {err_msg}")

    if not os.path.isfile(output_path) or os.path.getsize(output_path) == 0:
        raise RuntimeError("ffmpeg produced an empty or non-existent output file")

    output_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    if output_size_mb > 20.0:
        raise ValueError(f"Processed incident clip ({output_size_mb:.2f} MB) exceeds maximum allowed 20 MB ceiling")

    logger.info(f"Incident clip encoded successfully ({output_size_mb:.2f} MB).")


def _sync_send_telegram_video(
    video_path: str,
    chat_id: str,
    caption: str,
    token: Optional[str] = None,
) -> bool:
    """Synchronous worker to upload video to Telegram Bot API with 1 retry."""
    tok = (token or os.getenv("TELEGRAM_BOT_TOKEN", "")).strip()
    if not tok or not chat_id:
        logger.warning("Telegram Bot Token or chat ID missing. Cannot send incident video.")
        return False

    url = f"https://api.telegram.org/bot{tok}/sendVideo"

    for attempt in range(2):
        try:
            with open(video_path, "rb") as vf:
                files = {"video": ("incident_clip.mp4", vf, "video/mp4")}
                data = {
                    "chat_id": chat_id,
                    "caption": caption,
                    "supports_streaming": "true",
                }
                resp = requests.post(url, files=files, data=data, timeout=60)

            if resp.status_code == 200 and resp.json().get("ok"):
                logger.info(f"Incident video clip sent successfully to Telegram chat {mask_chat_id(chat_id)}.")
                return True
            else:
                body_sanitized = _sanitize_string(resp.text, tok)
                logger.warning(
                    f"Telegram sendVideo returned {resp.status_code} (attempt {attempt + 1}): {body_sanitized}"
                )
        except Exception as exc:
            err_sanitized = _sanitize_string(str(exc), tok)
            logger.warning(f"Telegram sendVideo attempt {attempt + 1} failed: {err_sanitized}")

        if attempt == 0:
            time.sleep(1.0)

    return False


async def send_telegram_video(
    video_path: str,
    chat_id: str,
    local_time: str,
    token: Optional[str] = None,
) -> bool:
    """Asynchronously send video clip to Telegram chat with a 60s timeout and 1 retry."""
    caption = f"Incident clip: 10 s before to 10 s after the fall started. Time: {local_time}"
    try:
        return await asyncio.wait_for(
            asyncio.to_thread(_sync_send_telegram_video, video_path, chat_id, caption, token),
            timeout=65.0,
        )
    except asyncio.TimeoutError:
        logger.warning("Telegram sendVideo timed out after 60s.")
        return False
    except Exception as exc:
        tok = token or os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
        logger.warning(f"Error executing send_telegram_video: {_sanitize_string(str(exc), tok)}")
        return False
