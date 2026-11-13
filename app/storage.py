"""
SQLite persistence module for tracking video ingestion jobs and pipeline state.
Thread-safe and serverless.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from app.config import SQLITE_PATH, WHISPER_MODEL, EMBED_MODEL, PROCESSING_VERSION


def _get_connection() -> sqlite3.Connection:
    """Returns a SQLite connection configured with Row factory."""
    conn = sqlite3.connect(SQLITE_PATH, timeout=30.0)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    """Initializes the database schema if it does not already exist."""
    with _get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS videos (
                video_id TEXT PRIMARY KEY,
                playlist_id TEXT,
                title TEXT NOT NULL,
                video_url TEXT NOT NULL,
                duration_seconds REAL NOT NULL DEFAULT 0.0,
                status TEXT NOT NULL,
                current_stage TEXT NOT NULL,
                error_message TEXT,
                asr_model TEXT NOT NULL,
                embedding_model TEXT NOT NULL,
                processing_version TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            """
        )
        conn.commit()


def get_video(video_id: str) -> Optional[Dict[str, Any]]:
    """Retrieves metadata and status for a single video."""
    init_db()
    with _get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM videos WHERE video_id = ?", (video_id,))
        row = cursor.fetchone()
        return dict(row) if row else None


def list_videos() -> List[Dict[str, Any]]:
    """Lists all registered videos, ordered by last update descending."""
    init_db()
    with _get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM videos ORDER BY updated_at DESC")
        rows = cursor.fetchall()
        return [dict(r) for r in rows]


def register_video(
    video_id: str,
    title: str,
    video_url: str,
    duration_seconds: float = 0.0,
    playlist_id: Optional[str] = None,
    asr_model: str = WHISPER_MODEL,
    embedding_model: str = EMBED_MODEL,
    processing_version: str = PROCESSING_VERSION,
) -> Dict[str, Any]:
    """Registers or updates a video record at the beginning of an ingestion run."""
    init_db()
    now = datetime.now(timezone.utc).isoformat()
    existing = get_video(video_id)

    with _get_connection() as conn:
        cursor = conn.cursor()
        if existing:
            cursor.execute(
                """
                UPDATE videos
                SET title = ?,
                    video_url = ?,
                    duration_seconds = ?,
                    playlist_id = COALESCE(?, playlist_id),
                    status = 'queued',
                    current_stage = 'registered',
                    error_message = NULL,
                    asr_model = ?,
                    embedding_model = ?,
                    processing_version = ?,
                    updated_at = ?
                WHERE video_id = ?
                """,
                (
                    title,
                    video_url,
                    duration_seconds,
                    playlist_id,
                    asr_model,
                    embedding_model,
                    processing_version,
                    now,
                    video_id,
                ),
            )
        else:
            cursor.execute(
                """
                INSERT INTO videos (
                    video_id, playlist_id, title, video_url, duration_seconds,
                    status, current_stage, error_message, asr_model,
                    embedding_model, processing_version, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, 'queued', 'registered', NULL, ?, ?, ?, ?, ?)
                """,
                (
                    video_id,
                    playlist_id,
                    title,
                    video_url,
                    duration_seconds,
                    asr_model,
                    embedding_model,
                    processing_version,
                    now,
                    now,
                ),
            )
        conn.commit()
    return get_video(video_id) or {}


def update_stage(
    video_id: str,
    stage: str,
    status: str = "in_progress",
    error_message: Optional[str] = None,
) -> None:
    """Updates the stage and status of an active ingestion job."""
    init_db()
    now = datetime.now(timezone.utc).isoformat()
    with _get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            UPDATE videos
            SET current_stage = ?,
                status = ?,
                error_message = ?,
                updated_at = ?
            WHERE video_id = ?
            """,
            (stage, status, error_message, now, video_id),
        )
        conn.commit()


def mark_completed(video_id: str) -> None:
    """Marks video ingestion as successfully completed."""
    update_stage(video_id, stage="completed", status="completed", error_message=None)


def mark_failed(video_id: str, error_message: str) -> None:
    """Marks video ingestion as failed with a sanitized error message."""
    update_stage(video_id, stage="failed", status="failed", error_message=error_message)


def delete_video(video_id: str) -> bool:
    """Removes a video record from SQLite."""
    init_db()
    with _get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM videos WHERE video_id = ?", (video_id,))
        conn.commit()
        return cursor.rowcount > 0

