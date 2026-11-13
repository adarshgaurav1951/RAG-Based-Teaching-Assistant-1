"""
Ingestion orchestrator for RAG-Based AI Teaching Assistant.
Processes EXACTLY ONE video per run through sequential stages:
1. URL validation & duplicate check (SQLite)
2. Metadata extraction (yt-dlp)
3. Audio download & FFmpeg normalization (mono 16 kHz PCM WAV)
4. Hinglish speech-to-text (faster-whisper CPU INT8)
5. Durable raw transcript JSON checkpoint
6. 75-second timestamp-aware chunking
7. Durable chunk JSON checkpoint
8. Multilingual MiniLM vector embeddings
9. Idempotent persistent Qdrant upsert
10. Indexing verification & SQLite completion status
11. Temporary audio cleanup
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from app.config import (
    EMBED_MODEL,
    PROCESSING_VERSION,
    WHISPER_MODEL,
)
from app.media import (
    cleanup_temp_audio,
    download_audio,
    extract_video_id,
    inspect_url,
    normalize_audio,
)
from app.retrieval import (
    chunk_transcript,
    load_chunks,
    save_chunks,
    upsert_chunks_to_qdrant,
    verify_video_indexed,
)
from app.storage import (
    get_video,
    init_db,
    mark_completed,
    mark_failed,
    register_video,
    update_stage,
)
from app.transcription import (
    load_raw_transcript,
    save_raw_transcript,
    transcribe_audio,
)

logger = logging.getLogger(__name__)


def ingest_video(
    url_or_id: str,
    force_reprocess: bool = False,
    whisper_model: str = WHISPER_MODEL,
    embed_model: str = EMBED_MODEL,
) -> Dict[str, Any]:
    """
    Ingests exactly one YouTube video end-to-end.
    Strictly adheres to single-video pipeline requirements and stage checkpoints.
    """
    init_db()

    # Stage 1: Validate YouTube URL / ID
    video_id = extract_video_id(url_or_id)
    if not video_id:
        raise ValueError(
            f"Invalid YouTube URL or video ID: '{url_or_id}'. "
            "Please provide a valid YouTube link or 11-character video ID."
        )

    # Stage 2: Check SQLite state & duplicate protection
    existing = get_video(video_id)
    if existing and existing["status"] == "completed" and not force_reprocess:
        return {
            "video_id": video_id,
            "title": existing["title"],
            "status": "already_indexed",
            "message": f"Video '{existing['title']}' is already indexed and ready for questions.",
            "duration_seconds": existing["duration_seconds"],
        }

    try:
        # Stage 3: Fetch metadata & register in SQLite
        update_stage(video_id, stage="fetching_metadata", status="in_progress")
        meta = inspect_url(f"https://www.youtube.com/watch?v={video_id}")
        title = meta.get("title", f"Video {video_id}")
        duration = meta.get("duration_seconds", 0.0)
        canonical_url = meta.get("video_url", f"https://www.youtube.com/watch?v={video_id}")

        register_video(
            video_id=video_id,
            title=title,
            video_url=canonical_url,
            duration_seconds=duration,
            asr_model=whisper_model,
            embedding_model=embed_model,
            processing_version=PROCESSING_VERSION,
        )

        # Stage 4: Check if raw transcript already exists on disk
        transcript_data = load_raw_transcript(video_id)
        if not transcript_data or force_reprocess:
            # Download audio
            update_stage(video_id, stage="downloading_audio", status="in_progress")
            raw_audio_path, _ = download_audio(canonical_url)

            # Normalize audio with FFmpeg to 16 kHz mono PCM WAV
            update_stage(video_id, stage="normalizing_audio", status="in_progress")
            norm_audio_path = normalize_audio(raw_audio_path)

            # Transcribe with faster-whisper on CPU
            update_stage(video_id, stage="transcribing", status="in_progress")
            transcript_data = transcribe_audio(
                norm_audio_path,
                video_id=video_id,
                title=title,
                model_size=whisper_model,
            )

            # Save raw transcript JSON checkpoint
            save_raw_transcript(transcript_data)

            # Clean temporary audio immediately after successful transcription
            cleanup_temp_audio(video_id)

        # Stage 5: Timestamp-aware chunking targeting ~75 seconds
        update_stage(video_id, stage="chunking", status="in_progress")
        chunks = chunk_transcript(transcript_data)
        if not chunks:
            raise ValueError(f"No valid transcript chunks generated for video {video_id}.")
        save_chunks(chunks, video_id=video_id)

        # Stage 6: MiniLM Vector Embeddings & Persistent Qdrant Upsert
        update_stage(video_id, stage="indexing", status="in_progress")
        upsert_count = upsert_chunks_to_qdrant(chunks)

        # Stage 7: Verification
        update_stage(video_id, stage="verifying", status="in_progress")
        is_verified, point_count = verify_video_indexed(video_id)
        if not is_verified:
            raise RuntimeError(
                f"Indexing verification failed for video {video_id}. No vector points retrieved."
            )

        # Stage 8: Mark completed in SQLite
        mark_completed(video_id)

        return {
            "video_id": video_id,
            "title": title,
            "status": "completed",
            "duration_seconds": duration,
            "segment_count": transcript_data.get("segment_count", len(transcript_data.get("segments", []))),
            "chunk_count": len(chunks),
            "points_indexed": point_count,
            "detected_language": transcript_data.get("detected_language", "auto"),
            "message": f"Successfully indexed '{title}'. Ready for student questions!",
        }

    except Exception as e:
        logger.error(f"Ingestion failed for {video_id}: {e}", exc_info=True)
        # Clean up any leftover temporary audio files
        cleanup_temp_audio(video_id)
        mark_failed(video_id, error_message=str(e))
        raise RuntimeError(f"Ingestion failed for {video_id}: {e}") from e

