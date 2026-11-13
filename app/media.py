"""
Media handling module for RAG-Based AI Teaching Assistant.
Handles YouTube metadata inspection, single-video audio download,
FFmpeg audio normalization to mono 16 kHz PCM WAV, and safe cleanup.
"""

from __future__ import annotations

import os
import re
import subprocess
import wave
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yt_dlp

from app.config import AUDIO_DIR

# Regex for YouTube video IDs (11 alphanumeric, hyphen, underscore characters)
YOUTUBE_ID_REGEX = re.compile(r"^[a-zA-Z0-9_-]{11}$")


def extract_video_id(url_or_id: str) -> Optional[str]:
    """Extracts a valid YouTube video ID from a URL or raw string."""
    url_or_id = url_or_id.strip()
    if YOUTUBE_ID_REGEX.match(url_or_id):
        return url_or_id

    # Pattern for watch?v=ID, youtu.be/ID, embed/ID, shorts/ID
    patterns = [
        r"(?:v=|\/)([0-9A-Za-z_-]{11}).*",
        r"youtu\.be\/([0-9A-Za-z_-]{11})",
        r"youtube\.com\/embed\/([0-9A-Za-z_-]{11})",
        r"youtube\.com\/shorts\/([0-9A-Za-z_-]{11})",
    ]
    for pattern in patterns:
        match = re.search(pattern, url_or_id)
        if match:
            return match.group(1)
    return None


def inspect_url(url: str) -> Dict[str, Any]:
    """
    Inspects a YouTube video or playlist URL without downloading media.
    Never downloads streams during inspection.
    """
    ydl_opts = {
        "extract_flat": "in_playlist",
        "skip_download": True,
        "quiet": True,
        "no_warnings": True,
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
    except Exception as e:
        raise RuntimeError(f"Failed to inspect YouTube URL '{url}': {e}") from e

    if not info:
        raise ValueError(f"No metadata found for URL: {url}")

    # Check if input is a playlist
    if "entries" in info:
        entries: List[Dict[str, Any]] = []
        for idx, entry in enumerate(info.get("entries", []), start=1):
            if not entry:
                continue
            v_id = entry.get("id")
            if not v_id:
                continue
            entries.append(
                {
                    "index": idx,
                    "video_id": v_id,
                    "title": entry.get("title", "Unknown Title"),
                    "duration_seconds": entry.get("duration", 0) or 0.0,
                    "video_url": f"https://www.youtube.com/watch?v={v_id}",
                }
            )
        return {
            "is_playlist": True,
            "playlist_id": info.get("id"),
            "playlist_title": info.get("title", "Untitled Playlist"),
            "video_count": len(entries),
            "videos": entries,
        }

    # Single video
    video_id = info.get("id") or extract_video_id(url)
    return {
        "is_playlist": False,
        "video_id": video_id,
        "title": info.get("title", "Unknown Title"),
        "duration_seconds": info.get("duration", 0) or 0.0,
        "video_url": info.get("webpage_url") or f"https://www.youtube.com/watch?v={video_id}",
    }


def download_audio(url_or_id: str, output_dir: Optional[Path] = None) -> Tuple[Path, Dict[str, Any]]:
    """
    Downloads audio for EXACTLY ONE video using yt-dlp with noplaylist=True.
    Returns (downloaded_audio_path, video_metadata).
    """
    dest_dir = output_dir or AUDIO_DIR
    dest_dir.mkdir(parents=True, exist_ok=True)

    # Output template: data/audio/%(id)s.%(ext)s
    out_tmpl = str(dest_dir / "%(id)s.%(ext)s")

    ydl_opts = {
        "format": "bestaudio/best",
        "noplaylist": True,
        "outtmpl": out_tmpl,
        "quiet": True,
        "no_warnings": True,
        "retries": 5,
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            meta = ydl.extract_info(url_or_id, download=True)
    except Exception as e:
        raise RuntimeError(f"Audio download failed for '{url_or_id}': {e}") from e

    if not meta:
        raise ValueError(f"Could not extract audio metadata for: {url_or_id}")

    video_id = meta.get("id")
    title = meta.get("title", "Unknown Title")
    duration = meta.get("duration", 0.0) or 0.0
    canonical_url = meta.get("webpage_url") or f"https://www.youtube.com/watch?v={video_id}"

    # Locate the downloaded file
    downloaded_files = list(dest_dir.glob(f"{video_id}.*"))
    # Filter out any pre-existing normalized wavs
    raw_files = [f for f in downloaded_files if not f.name.endswith(".normalized.wav")]

    if not raw_files:
        raise FileNotFoundError(f"Expected downloaded audio for {video_id} was not found.")

    raw_audio_path = raw_files[0]
    metadata = {
        "video_id": video_id,
        "title": title,
        "duration_seconds": float(duration),
        "video_url": canonical_url,
    }

    return raw_audio_path, metadata


def normalize_audio(input_path: Path, output_path: Optional[Path] = None) -> Path:
    """
    Normalizes audio to mono 16 kHz PCM WAV using FFmpeg.
    Verifies valid PCM WAV structure using standard library wave module.
    """
    if not input_path.exists():
        raise FileNotFoundError(f"Input audio file does not exist: {input_path}")

    if output_path is None:
        output_path = input_path.parent / f"{input_path.stem}.normalized.wav"

    # FFmpeg command: mono, 16 kHz, 16-bit PCM WAV
    cmd = [
        "ffmpeg",
        "-y",  # Overwrite output
        "-i",
        str(input_path),
        "-vn",  # No video
        "-ac",
        "1",  # Mono
        "-ar",
        "16000",  # 16000 Hz
        "-c:a",
        "pcm_s16le",  # PCM 16-bit little-endian
        str(output_path),
    ]

    try:
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
            text=True,
        )
    except FileNotFoundError:
        raise RuntimeError(
            "FFmpeg executable not found in PATH. Please ensure ffmpeg is installed."
        )
    except subprocess.CalledProcessError as e:
        clean_err = e.stderr[-500:] if e.stderr else str(e)
        raise RuntimeError(f"FFmpeg normalization failed: {clean_err}") from e

    # Verify output audio using standard library wave
    if not output_path.exists() or output_path.stat().st_size == 0:
        raise RuntimeError(f"Normalized audio file is empty or missing: {output_path}")

    try:
        with wave.open(str(output_path), "rb") as wf:
            channels = wf.getnchannels()
            framerate = wf.getframerate()
            frames = wf.getnframes()
            if channels != 1:
                raise ValueError(f"Expected 1 audio channel, got {channels}")
            if framerate != 16000:
                raise ValueError(f"Expected 16000 Hz framerate, got {framerate}")
            duration = frames / float(framerate)
            if duration <= 0:
                raise ValueError("Normalized audio has zero duration.")
    except Exception as e:
        raise RuntimeError(f"Validation of normalized WAV failed: {e}") from e

    return output_path


def cleanup_temp_audio(video_id: str, dir_path: Optional[Path] = None) -> None:
    """Safely cleans up any temporary raw or normalized audio files for a video."""
    target_dir = dir_path or AUDIO_DIR
    if not target_dir.exists():
        return

    for f in target_dir.glob(f"{video_id}*"):
        try:
            if f.is_file():
                f.unlink(missing_ok=True)
        except OSError:
            pass

