"""
Speech-to-text transcription module using faster-whisper on CPU.
Supports multilingual Hinglish transcription with preserved original speech,
accurate segment timestamps, and durable JSON checkpoints.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.config import (
    TRANSCRIPT_DIR,
    WHISPER_CPU_THREADS,
    WHISPER_LANGUAGE,
    WHISPER_MODEL,
)

# Global model cache to avoid re-loading weights within the same process
_MODEL_CACHE: Dict[str, Any] = {}


def get_whisper_model(
    model_size: str = WHISPER_MODEL,
    cpu_threads: int = WHISPER_CPU_THREADS,
) -> Any:
    """Loads and caches the faster-whisper model on CPU with INT8 compute."""
    cache_key = f"{model_size}_{cpu_threads}"
    if cache_key in _MODEL_CACHE:
        return _MODEL_CACHE[cache_key]

    try:
        from faster_whisper import WhisperModel
    except ImportError as e:
        raise RuntimeError(
            "faster-whisper is not installed. Please install it via requirements.txt."
        ) from e

    # CPU INT8 configuration for low-RAM hardware
    model = WhisperModel(
        model_size,
        device="cpu",
        compute_type="int8",
        cpu_threads=cpu_threads,
    )
    _MODEL_CACHE[cache_key] = model
    return model


def transcribe_audio(
    audio_path: Path,
    video_id: str,
    title: str = "",
    model_size: str = WHISPER_MODEL,
    language: str = WHISPER_LANGUAGE,
    cpu_threads: int = WHISPER_CPU_THREADS,
    initial_prompt: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Transcribes normalized 16 kHz PCM WAV audio on CPU.
    Preserves original Hinglish / Hindi / English spoken speech without forced translation.
    """
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file to transcribe not found: {audio_path}")

    model = get_whisper_model(model_size=model_size, cpu_threads=cpu_threads)

    # Determine language argument
    lang_arg: Optional[str] = None
    if language and language.lower() not in ["auto", "none", ""]:
        lang_arg = language.lower()

    # Default technical Hinglish prompt when not overridden
    prompt = initial_prompt or (
        "Yeh video lecture hai Generative AI, machine learning, Python aur data science ke baare mein."
    )

    segments_gen, info = model.transcribe(
        str(audio_path),
        task="transcribe",  # Never translate
        language=lang_arg,
        beam_size=1,        # CPU fast path
        vad_filter=True,    # Filter silence
        word_timestamps=True,
        initial_prompt=prompt,
    )

    formatted_segments: List[Dict[str, Any]] = []
    for idx, seg in enumerate(segments_gen, start=1):
        clean_text = seg.text.strip()
        if not clean_text:
            continue
        formatted_segments.append(
            {
                "segment_id": idx,
                "start": round(seg.start, 2),
                "end": round(seg.end, 2),
                "text": clean_text,
            }
        )

    transcript_data = {
        "video_id": video_id,
        "title": title,
        "asr_model": model_size,
        "detected_language": info.language,
        "language_probability": round(info.language_probability, 4),
        "duration_seconds": round(info.duration, 2),
        "segment_count": len(formatted_segments),
        "segments": formatted_segments,
    }

    return transcript_data


def save_raw_transcript(
    transcript_data: Dict[str, Any],
    output_dir: Optional[Path] = None,
) -> Path:
    """Saves raw transcript JSON checkpoint to disk."""
    dest_dir = output_dir or TRANSCRIPT_DIR
    dest_dir.mkdir(parents=True, exist_ok=True)
    video_id = transcript_data["video_id"]
    out_file = dest_dir / f"{video_id}_raw.json"

    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(transcript_data, f, ensure_ascii=False, indent=2)

    return out_file


def load_raw_transcript(
    video_id: str,
    transcript_dir: Optional[Path] = None,
) -> Optional[Dict[str, Any]]:
    """Loads a previously saved raw transcript JSON checkpoint if present."""
    dest_dir = transcript_dir or TRANSCRIPT_DIR
    in_file = dest_dir / f"{video_id}_raw.json"
    if not in_file.exists():
        return None

    with open(in_file, "r", encoding="utf-8") as f:
        return json.load(f)

