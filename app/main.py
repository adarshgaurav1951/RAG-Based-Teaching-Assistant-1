"""
FastAPI application for RAG-Based AI Teaching Assistant.
Features an Unriddle / NotebookLM inspired modern user interface with:
- 3-column SaaS layout (Sidebar, Video Player + Visual Concept Map + Transcript, Interactive AI Chat)
- Embedded YouTube Player with synchronized timestamp scrubbing
- Visual Concept Roadmap with interactive concept cards & chapter timeline
- Multilingual Hinglish semantic retrieval and grounded teaching responses
- Clickable timestamp pills that seek the video player in real-time
"""

from __future__ import annotations

import re
import shutil
import traceback
from typing import Any, Dict, List, Optional

from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from app.config import (
    CHUNK_SECONDS,
    EMBED_MODEL,
    LLM_API_KEY,
    LLM_MODEL,
    WHISPER_MODEL,
)
from app.ingest import ingest_video
from app.media import extract_video_id, inspect_url
from app.retrieval import verify_video_indexed
from app.storage import get_video, list_videos
from app.teaching import generate_answer
from app.transcription import load_raw_transcript

app = FastAPI(
    title="Hinglish YouTube RAG AI Teaching Assistant",
    description="Resource-efficient, CPU-first RAG assistant for Hinglish lecture videos",
    version="1.0.0",
)

# Enable CORS for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# -----------------------------------------------------------------------------
# Request & Response Models
# -----------------------------------------------------------------------------

class InspectRequest(BaseModel):
    url: str = Field(..., description="YouTube video or playlist URL")


class IngestRequest(BaseModel):
    url: str = Field(..., description="YouTube video URL or 11-char ID")
    force: bool = Field(False, description="Force re-transcription and re-indexing")


class AskRequest(BaseModel):
    question: str = Field(..., min_length=2, description="Student question in English, Hindi, or Hinglish")
    video_id: Optional[str] = Field(None, description="Scope query to a specific video")
    top_k: int = Field(5, ge=1, le=10, description="Number of context chunks to retrieve")
    language_preference: Optional[str] = Field(None, description="Preferred response language")


# -----------------------------------------------------------------------------
# REST API Endpoints
# -----------------------------------------------------------------------------

@app.get("/api/health")
def health_check() -> Dict[str, Any]:
    """Safe health check endpoint. Never exposes secrets."""
    ffmpeg_available = shutil.which("ffmpeg") is not None
    return {
        "status": "healthy",
        "ffmpeg_installed": ffmpeg_available,
        "llm_key_configured": bool(LLM_API_KEY),
        "llm_model": LLM_MODEL,
        "asr_model": WHISPER_MODEL,
        "embedding_model": EMBED_MODEL,
        "chunk_seconds": CHUNK_SECONDS,
    }


@app.post("/api/playlists/inspect")
def api_inspect_url(req: InspectRequest) -> Dict[str, Any]:
    """Inspects a YouTube URL without downloading audio or video streams."""
    try:
        return inspect_url(req.url)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/videos/ingest")
def api_ingest_video(req: IngestRequest, bg_tasks: BackgroundTasks) -> Dict[str, Any]:
    """Starts single-video ingestion."""
    v_id = extract_video_id(req.url)
    if not v_id:
        raise HTTPException(status_code=400, detail="Invalid YouTube URL or ID.")

    existing = get_video(v_id)
    if existing and existing["status"] == "completed" and not req.force:
        return {
            "status": "already_indexed",
            "video_id": v_id,
            "title": existing["title"],
            "message": "Video is already indexed and ready for questions.",
        }

    try:
        res = ingest_video(req.url, force_reprocess=req.force)
        return res
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/videos")
def api_list_videos() -> List[Dict[str, Any]]:
    """Lists all registered videos and their ingestion statuses."""
    return list_videos()


@app.get("/api/videos/{video_id}")
def api_get_video(video_id: str) -> Dict[str, Any]:
    """Retrieves detailed status for a single video."""
    video = get_video(video_id)
    if not video:
        raise HTTPException(status_code=404, detail="Video not found.")
    is_indexed, count = verify_video_indexed(video_id)
    result = dict(video)
    result["vector_points_indexed"] = count
    return result


@app.get("/api/videos/{video_id}/transcript")
def api_get_video_transcript(video_id: str) -> Dict[str, Any]:
    """Retrieves full transcript segments for an indexed video."""
    data = load_raw_transcript(video_id)
    if not data:
        raise HTTPException(status_code=404, detail="Transcript not found.")
    return data


@app.post("/api/videos/{video_id}/retry")
def api_retry_video(video_id: str) -> Dict[str, Any]:
    """Retries a failed ingestion job."""
    video = get_video(video_id)
    if not video:
        raise HTTPException(status_code=404, detail="Video not found.")
    try:
        return ingest_video(video["video_url"], force_reprocess=True)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/ask")
def api_ask_question(req: AskRequest) -> Dict[str, Any]:
    """Queries the knowledge base and generates a grounded teaching answer."""
    try:
        return generate_answer(
            question=req.question,
            video_id=req.video_id,
            top_k=req.top_k,
            language_preference=req.language_preference,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# -----------------------------------------------------------------------------
# Embedded Single-Page Frontend (Unriddle / NotebookLM Modern SaaS Style)
# -----------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
def index() -> str:
    """Returns the Unriddle-style 3-column AI document and video assistant UI."""
    return """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Unriddle AI - Video & Lecture Assistant</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">
  <style>
    :root {
      --bg: #f8fafc;
      --surface: #ffffff;
      --border: #e2e8f0;
      --border-subtle: #f1f5f9;
      --text-main: #0f172a;
      --text-muted: #64748b;
      --text-light: #94a3b8;
      --sidebar-bg: #fcfcfd;
      --sidebar-active: #f1f5f9;
      --primary-black: #09090b;
      --accent-blue: #2563eb;
      --accent-indigo: #4f46e5;
      --accent-orange: #ea580c;
      --accent-emerald: #10b981;
      --accent-violet: #8b5cf6;
      --pill-bg: #f1f5f9;
      --pill-border: #e2e8f0;
    }

    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
      background: #eef2f6;
      color: var(--text-main);
      padding: 16px;
      height: 100vh;
      overflow: hidden;
    }

    /* Master Window Wrapper */
    .app-window {
      background: var(--surface);
      border-radius: 20px;
      border: 1px solid var(--border);
      box-shadow: 0 20px 45px -10px rgba(15, 23, 42, 0.09), 0 0 1px 1px rgba(15, 23, 42, 0.04);
      height: calc(100vh - 32px);
      display: flex;
      overflow: hidden;
      position: relative;
    }

    /* ----------------------------------------------------------------------- */
    /* 1. Left Sidebar                                                         */
    /* ----------------------------------------------------------------------- */
    .sidebar {
      width: 250px;
      background: var(--sidebar-bg);
      border-right: 1px solid var(--border);
      display: flex;
      flex-direction: column;
      flex-shrink: 0;
      user-select: none;
    }

    .sidebar-header {
      padding: 16px 14px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      border-bottom: 1px solid transparent;
    }
    .user-dropdown {
      display: flex;
      align-items: center;
      gap: 6px;
      font-weight: 600;
      font-size: 0.92rem;
      color: var(--text-main);
      cursor: pointer;
      padding: 4px 6px;
      border-radius: 6px;
      transition: background 0.15s;
    }
    .user-dropdown:hover { background: var(--sidebar-active); }
    .user-dropdown svg { width: 14px; height: 14px; color: var(--text-muted); }

    .btn-icon-plus {
      background: #f1f5f9;
      border: 1px solid var(--border);
      width: 28px;
      height: 28px;
      border-radius: 8px;
      display: flex;
      align-items: center;
      justify-content: center;
      cursor: pointer;
      color: var(--text-main);
      transition: all 0.15s;
    }
    .btn-icon-plus:hover { background: #e2e8f0; transform: scale(1.05); }

    .nav-section {
      padding: 4px 10px;
      display: flex;
      flex-direction: column;
      gap: 2px;
    }
    .nav-item {
      display: flex;
      align-items: center;
      gap: 10px;
      padding: 7px 10px;
      border-radius: 8px;
      font-size: 0.86rem;
      font-weight: 500;
      color: #334155;
      cursor: pointer;
      transition: all 0.12s;
      text-decoration: none;
    }
    .nav-item:hover { background: var(--sidebar-active); color: var(--text-main); }
    .nav-icon { width: 16px; height: 16px; color: var(--text-muted); flex-shrink: 0; }

    .doc-section {
      padding: 8px 10px 4px 10px;
      margin-top: 4px;
      flex: 1;
      overflow-y: auto;
    }
    .doc-item {
      display: flex;
      align-items: center;
      gap: 9px;
      padding: 7px 10px;
      border-radius: 8px;
      font-size: 0.84rem;
      color: #475569;
      cursor: pointer;
      transition: all 0.12s;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
      margin-bottom: 2px;
    }
    .doc-item:hover { background: var(--sidebar-active); color: var(--text-main); }
    .doc-item.selected {
      background: var(--sidebar-active);
      color: var(--text-main);
      font-weight: 600;
    }

    .sidebar-footer {
      padding: 12px 14px;
      border-top: 1px solid var(--border);
      font-size: 0.76rem;
      color: var(--text-light);
      display: flex;
      align-items: center;
      justify-content: space-between;
    }

    /* ----------------------------------------------------------------------- */
    /* 2. Center Column (Video Player + Visual Roadmap + Transcript)           */
    /* ----------------------------------------------------------------------- */
    .center-col {
      flex: 1;
      background: var(--surface);
      display: flex;
      flex-direction: column;
      overflow-y: auto;
      border-right: 1px solid var(--border);
      min-width: 0;
    }

    .center-header {
      padding: 14px 24px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      border-bottom: 1px solid var(--border-subtle);
    }
    .doc-title-input {
      font-size: 1.05rem;
      font-weight: 700;
      color: var(--text-main);
      border: none;
      outline: none;
      background: transparent;
      width: 100%;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }

    .top-actions {
      display: flex;
      align-items: center;
      gap: 8px;
      flex-shrink: 0;
    }
    .btn-upgrade {
      background: var(--primary-black);
      color: #ffffff;
      font-size: 0.78rem;
      font-weight: 600;
      padding: 6px 14px;
      border-radius: 9999px;
      border: none;
      cursor: pointer;
      transition: opacity 0.15s;
    }
    .btn-upgrade:hover { opacity: 0.88; }
    .btn-header-pill {
      background: transparent;
      border: 1px solid transparent;
      padding: 5px 10px;
      border-radius: 8px;
      font-size: 0.8rem;
      font-weight: 500;
      color: var(--text-muted);
      cursor: pointer;
      display: flex;
      align-items: center;
      gap: 5px;
      transition: all 0.12s;
    }
    .btn-header-pill:hover, .btn-header-pill.active {
      background: var(--sidebar-active);
      color: var(--text-main);
      font-weight: 600;
    }

    .center-content {
      padding: 20px 24px;
      display: flex;
      flex-direction: column;
      gap: 16px;
      max-width: 860px;
      margin: 0 auto;
      width: 100%;
    }

    /* Video Player Container */
    .video-card {
      width: 100%;
      aspect-ratio: 16 / 9;
      background: #000000;
      border-radius: 12px;
      overflow: hidden;
      box-shadow: 0 4px 16px rgba(0, 0, 0, 0.08);
      position: relative;
    }
    .video-card iframe, .video-card #yt-player {
      width: 100%;
      height: 100%;
      border: none;
    }

    /* VISUAL CHAPTER TIMELINE BAR */
    .visual-timeline-box {
      background: #f8fafc;
      border: 1px solid var(--border);
      border-radius: 10px;
      padding: 10px 14px;
      display: flex;
      flex-direction: column;
      gap: 8px;
    }
    .timeline-title-row {
      display: flex;
      justify-content: space-between;
      align-items: center;
      font-size: 0.76rem;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.04em;
      color: var(--text-muted);
    }
    .timeline-track {
      height: 8px;
      background: #e2e8f0;
      border-radius: 6px;
      display: flex;
      overflow: hidden;
      cursor: pointer;
    }
    .track-seg {
      height: 100%;
      transition: opacity 0.15s, transform 0.1s;
      position: relative;
    }
    .track-seg:hover { opacity: 0.85; }
    .seg-1 { background: #3b82f6; width: 14%; } /* Intro */
    .seg-2 { background: #8b5cf6; width: 20%; } /* Neural Nets */
    .seg-3 { background: #ec4899; width: 24%; } /* LLMs */
    .seg-4 { background: #f97316; width: 22%; } /* RAG */
    .seg-5 { background: #10b981; width: 20%; } /* Fine-tuning */

    .timeline-chips {
      display: flex;
      gap: 6px;
      flex-wrap: wrap;
    }
    .time-chip {
      display: inline-flex;
      align-items: center;
      gap: 5px;
      padding: 3px 8px;
      border-radius: 6px;
      font-size: 0.74rem;
      font-weight: 600;
      background: #ffffff;
      border: 1px solid var(--border);
      color: #334155;
      cursor: pointer;
      transition: all 0.15s;
    }
    .time-chip:hover {
      background: #0f172a;
      color: #ffffff;
      border-color: #0f172a;
    }
    .time-chip .chip-dot {
      width: 6px;
      height: 6px;
      border-radius: 50%;
    }

    /* Center Pane Visual Tabs */
    .center-tabs-nav {
      display: flex;
      gap: 8px;
      border-bottom: 1px solid var(--border);
      padding-bottom: 2px;
      margin-top: 4px;
    }
    .c-tab-btn {
      background: transparent;
      border: none;
      font-size: 0.85rem;
      font-weight: 600;
      color: var(--text-muted);
      cursor: pointer;
      padding: 6px 12px;
      border-radius: 6px;
      transition: all 0.12s;
      display: flex;
      align-items: center;
      gap: 6px;
    }
    .c-tab-btn:hover { color: var(--text-main); background: #f1f5f9; }
    .c-tab-btn.active { color: var(--accent-blue); background: #eff6ff; }

    /* VISUAL CONCEPT ROADMAP GRID */
    .concept-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
      gap: 12px;
      margin-top: 6px;
    }
    .concept-card {
      background: #ffffff;
      border: 1px solid var(--border);
      border-radius: 10px;
      padding: 14px;
      box-shadow: 0 2px 6px rgba(15, 23, 42, 0.03);
      display: flex;
      flex-direction: column;
      justify-content: space-between;
      gap: 10px;
      transition: all 0.18s;
      border-left: 4px solid var(--accent-blue);
    }
    .concept-card:hover {
      transform: translateY(-2px);
      box-shadow: 0 8px 18px rgba(15, 23, 42, 0.06);
      border-color: #cbd5e1;
    }
    .concept-card.orange { border-left-color: var(--accent-orange); }
    .concept-card.purple { border-left-color: var(--accent-violet); }
    .concept-card.green { border-left-color: var(--accent-emerald); }

    .c-card-top {
      display: flex;
      align-items: flex-start;
      justify-content: space-between;
    }
    .c-card-title {
      font-size: 0.9rem;
      font-weight: 700;
      color: var(--text-main);
    }
    .c-card-desc {
      font-size: 0.82rem;
      color: var(--text-muted);
      line-height: 1.45;
      margin-top: 4px;
    }
    .c-card-footer {
      display: flex;
      align-items: center;
      justify-content: space-between;
      padding-top: 6px;
      border-top: 1px solid var(--border-subtle);
      font-size: 0.76rem;
    }
    .btn-card-seek {
      background: #f1f5f9;
      border: 1px solid var(--border);
      color: var(--accent-blue);
      font-weight: 600;
      padding: 3px 8px;
      border-radius: 6px;
      cursor: pointer;
      display: inline-flex;
      align-items: center;
      gap: 4px;
      transition: all 0.15s;
    }
    .btn-card-seek:hover { background: var(--accent-blue); color: #fff; }

    /* Transcript & Searchable List */
    .transcript-toolbar {
      display: flex;
      gap: 8px;
      align-items: center;
      margin-bottom: 8px;
    }
    .transcript-search {
      flex: 1;
      padding: 6px 12px;
      border-radius: 8px;
      border: 1px solid var(--border);
      font-size: 0.82rem;
      outline: none;
      background: #f8fafc;
    }
    .transcript-search:focus { border-color: var(--accent-blue); background: #ffffff; }

    .transcript-feed {
      display: flex;
      flex-direction: column;
      gap: 10px;
      max-height: 380px;
      overflow-y: auto;
      padding-right: 6px;
    }
    .transcript-item {
      padding: 10px 12px;
      border-radius: 8px;
      background: #f8fafc;
      border: 1px solid var(--border-subtle);
      font-size: 0.86rem;
      line-height: 1.55;
      color: #334155;
      transition: background 0.15s;
    }
    .transcript-item:hover { background: #f1f5f9; }
    .ts-tag {
      display: inline-block;
      font-size: 0.74rem;
      font-weight: 600;
      color: var(--accent-blue);
      background: #eff6ff;
      border: 1px solid #dbeafe;
      padding: 1px 6px;
      border-radius: 5px;
      cursor: pointer;
      margin-right: 6px;
    }
    .ts-tag:hover { background: var(--accent-blue); color: #fff; }

    /* ----------------------------------------------------------------------- */
    /* 3. Right Column (Unriddle AI Chat Assistant)                            */
    /* ----------------------------------------------------------------------- */
    .chat-col {
      width: 440px;
      background: var(--surface);
      display: flex;
      flex-direction: column;
      flex-shrink: 0;
      position: relative;
    }

    .chat-header {
      padding: 14px 20px;
      display: flex;
      align-items: center;
      gap: 16px;
      border-bottom: 1px solid var(--border-subtle);
    }
    .chat-tab {
      font-size: 0.88rem;
      font-weight: 500;
      color: var(--text-muted);
      cursor: pointer;
      padding: 4px 0;
      position: relative;
      transition: color 0.15s;
    }
    .chat-tab.active {
      font-weight: 600;
      color: var(--text-main);
    }
    .chat-tab.active::after {
      content: '';
      position: absolute;
      bottom: -15px;
      left: 0;
      right: 0;
      height: 2px;
      background: var(--primary-black);
    }

    .chat-scroll {
      flex: 1;
      overflow-y: auto;
      padding: 20px;
      display: flex;
      flex-direction: column;
      gap: 20px;
    }

    /* Message Block */
    .msg-block {
      display: flex;
      flex-direction: column;
      gap: 6px;
      animation: fadeIn 0.18s ease-in;
    }
    @keyframes fadeIn { from { opacity: 0; transform: translateY(4px); } to { opacity: 1; transform: translateY(0); } }

    .msg-sender {
      display: flex;
      align-items: center;
      gap: 7px;
      font-size: 0.74rem;
      font-weight: 700;
      letter-spacing: 0.04em;
      text-transform: uppercase;
    }
    .sender-dot {
      width: 7px;
      height: 7px;
      border-radius: 50%;
    }
    .sender-dot.user { background: var(--accent-blue); }
    .sender-dot.assistant { background: var(--accent-orange); }

    .msg-content {
      font-size: 0.91rem;
      line-height: 1.62;
      color: #1e293b;
      word-break: break-word;
    }

    /* Inline Timestamp Pill */
    .timestamp-pill {
      display: inline-flex;
      align-items: center;
      gap: 3px;
      background: var(--pill-bg);
      border: 1px solid var(--pill-border);
      color: #334155;
      padding: 1px 7px;
      border-radius: 6px;
      font-size: 0.78rem;
      font-weight: 600;
      cursor: pointer;
      text-decoration: none;
      margin: 0 3px;
      vertical-align: baseline;
      transition: all 0.12s;
    }
    .timestamp-pill:hover {
      background: #e2e8f0;
      color: #09090b;
      border-color: #cbd5e1;
      transform: translateY(-1px);
    }

    /* Floating Chat Input Card */
    .chat-input-wrapper {
      padding: 14px 18px 16px 18px;
      background: var(--surface);
    }
    .input-card {
      border: 1px solid var(--border);
      border-radius: 14px;
      padding: 10px 14px;
      background: #ffffff;
      box-shadow: 0 4px 16px -2px rgba(15, 23, 42, 0.05);
      display: flex;
      flex-direction: column;
      gap: 8px;
      transition: border-color 0.15s, box-shadow 0.15s;
    }
    .input-card:focus-within {
      border-color: #94a3b8;
      box-shadow: 0 4px 20px -2px rgba(15, 23, 42, 0.09);
    }

    .chat-textarea {
      width: 100%;
      border: none;
      outline: none;
      resize: none;
      font-family: inherit;
      font-size: 0.88rem;
      color: var(--text-main);
      background: transparent;
      line-height: 1.45;
      height: 48px;
    }
    .chat-textarea::placeholder { color: var(--text-light); }

    .input-toolbar {
      display: flex;
      align-items: center;
      justify-content: space-between;
      padding-top: 4px;
    }
    .toolbar-left {
      display: flex;
      align-items: center;
      gap: 8px;
    }
    .pill-select {
      background: transparent;
      border: none;
      font-size: 0.76rem;
      font-weight: 500;
      color: var(--text-muted);
      cursor: pointer;
      outline: none;
      padding: 2px 4px;
      border-radius: 4px;
      transition: background 0.12s;
    }
    .pill-select:hover { background: var(--sidebar-active); color: var(--text-main); }

    .toolbar-right {
      display: flex;
      align-items: center;
      gap: 8px;
    }
    .btn-clip {
      background: transparent;
      border: none;
      color: var(--text-muted);
      cursor: pointer;
      display: flex;
      align-items: center;
      justify-content: center;
      padding: 4px;
      border-radius: 6px;
    }
    .btn-clip:hover { color: var(--text-main); }

    .btn-send {
      width: 26px;
      height: 26px;
      background: var(--primary-black);
      color: #ffffff;
      border: none;
      border-radius: 50%;
      display: flex;
      align-items: center;
      justify-content: center;
      cursor: pointer;
      transition: transform 0.12s, opacity 0.15s;
    }
    .btn-send:hover { opacity: 0.9; transform: scale(1.05); }

    /* Quick Prompt Chips */
    .quick-chips-row {
      display: flex;
      gap: 6px;
      overflow-x: auto;
      padding-bottom: 6px;
      scrollbar-width: none;
    }
    .quick-chips-row::-webkit-scrollbar { display: none; }
    .q-chip {
      background: #f8fafc;
      border: 1px solid var(--border);
      border-radius: 9999px;
      padding: 3px 10px;
      font-size: 0.74rem;
      font-weight: 600;
      color: #475569;
      cursor: pointer;
      white-space: nowrap;
      transition: all 0.15s;
    }
    .q-chip:hover {
      background: #0f172a;
      color: #ffffff;
      border-color: #0f172a;
    }

    /* Ingestion Modal */
    .modal-overlay {
      position: absolute;
      top: 0; left: 0; right: 0; bottom: 0;
      background: rgba(15, 23, 42, 0.45);
      backdrop-filter: blur(2px);
      display: none;
      align-items: center;
      justify-content: center;
      z-index: 50;
    }
    .modal-card {
      background: #ffffff;
      border-radius: 16px;
      width: 480px;
      max-width: 90%;
      padding: 24px;
      box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.1), 0 10px 10px -5px rgba(0, 0, 0, 0.04);
      border: 1px solid var(--border);
    }
    .modal-card h2 { font-size: 1.15rem; font-weight: 700; margin-bottom: 6px; }
    .modal-card p { font-size: 0.85rem; color: var(--text-muted); margin-bottom: 16px; }
    .modal-input {
      width: 100%;
      padding: 10px 14px;
      border: 1px solid var(--border);
      border-radius: 8px;
      font-size: 0.9rem;
      outline: none;
      margin-bottom: 14px;
    }
    .modal-input:focus { border-color: var(--accent-blue); }
    .modal-actions { display: flex; justify-content: flex-end; gap: 10px; }
    .btn-ghost {
      background: transparent;
      border: 1px solid var(--border);
      padding: 8px 16px;
      border-radius: 8px;
      font-size: 0.85rem;
      font-weight: 600;
      cursor: pointer;
    }
    .btn-primary {
      background: var(--primary-black);
      color: #fff;
      border: none;
      padding: 8px 18px;
      border-radius: 8px;
      font-size: 0.85rem;
      font-weight: 600;
      cursor: pointer;
    }
    .modal-status {
      margin-top: 12px;
      padding: 10px;
      background: #f8fafc;
      border-radius: 6px;
      font-size: 0.82rem;
      color: #334155;
      display: none;
    }
  </style>
</head>
<body>

  <div class="app-window">
    
    <!-- 1. LEFT SIDEBAR -->
    <aside class="sidebar">
      <div class="sidebar-header">
        <div class="user-dropdown">
          <span>Adarsh</span>
          <svg viewBox="0 0 20 20" fill="currentColor"><path fill-rule="evenodd" d="M5.23 7.21a.75.75 0 011.06.02L10 11.168l3.71-3.938a.75.75 0 111.08 1.04l-4.25 4.5a.75.75 0 01-1.08 0l-4.25-4.5a.75.75 0 01.02-1.06z" clip-rule="evenodd"/></svg>
        </div>
        <button class="btn-icon-plus" onclick="openAddModal()" title="Add YouTube Lecture">+</button>
      </div>

      <nav class="nav-section">
        <div class="nav-item">
          <svg class="nav-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M3 9l9-7 9 7v11a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z"/></svg>
          <span>Home</span>
        </div>
        <div class="nav-item">
          <svg class="nav-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/></svg>
          <span>Search</span>
        </div>
        <div class="nav-item">
          <svg class="nav-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="2" y="3" width="20" height="14" rx="2" ry="2"/><line x1="8" y1="21" x2="16" y2="21"/><line x1="12" y1="17" x2="12" y2="21"/></svg>
          <span>Connections</span>
        </div>
      </nav>

      <div class="doc-section">
        <div style="font-size: 0.72rem; font-weight: 700; color: var(--text-light); text-transform: uppercase; letter-spacing: 0.05em; padding: 6px 10px 4px 10px;">
          Lectures & Notes
        </div>
        <div id="videoSidebarList">
          <div class="doc-item selected" id="activeDocItem">
            <svg class="nav-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/></svg>
            <span id="activeDocLabel">GenAI Roadmap for Beginners</span>
          </div>
        </div>

        <div style="margin-top: 14px; font-size: 0.72rem; font-weight: 700; color: var(--text-light); text-transform: uppercase; letter-spacing: 0.05em; padding: 6px 10px 4px 10px;">
          Resources
        </div>
        <div class="doc-item" onclick="insertPrompt('What are the prerequisites for this course?')">
          <svg class="nav-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polygon points="12 2 15.09 8.26 22 9.27 17 14.14 18.18 21.02 12 17.77 5.82 21.02 7 14.14 2 9.27 8.91 8.26 12 2"/></svg>
          <span>Prompts to get you started</span>
        </div>
        <div class="doc-item">
          <svg class="nav-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M2 3h6a4 4 0 0 1 4 4v14a3 3 0 0 0-3-3H2z"/><path d="M22 3h-6a4 4 0 0 0-4 4v14a3 3 0 0 1 3-3h7z"/></svg>
          <span>Welcome guide</span>
        </div>
        <div class="doc-item">
          <svg class="nav-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="2" y="4" width="20" height="16" rx="2"/><path d="M6 8h.01M10 8h.01M14 8h.01M18 8h.01M8 12h.01M12 12h.01M16 12h.01M7 16h10"/></svg>
          <span>Keyboard shortcuts</span>
        </div>
      </div>

      <div class="sidebar-footer">
        <span>CPU-Only INT8 Engine</span>
        <span style="font-weight: 600; color: #10b981;">● Online</span>
      </div>
    </aside>

    <!-- 2. CENTER COLUMN (Video Player + Visual Roadmap + Transcript) -->
    <main class="center-col">
      <div class="center-header">
        <input type="text" id="lectureTitleInput" class="doc-title-input" value="GenAI Roadmap for Beginners | CampusX" readonly>
        <div class="top-actions">
          <button class="btn-upgrade">Upgrade</button>
          <button class="btn-header-pill active">
            <span>✦</span>
            <span>Chat</span>
          </button>
          <button class="btn-header-pill" onclick="switchCenterTab('notes')">
            <span>✏️</span>
            <span>Notes</span>
          </button>
          <button class="btn-header-pill" title="Toggle Layout">
            <svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" stroke-width="2"><rect x="3" y="3" width="18" height="18" rx="2" ry="2"/><line x1="9" y1="3" x2="9" y2="21"/></svg>
          </button>
        </div>
      </div>

      <div class="center-content">
        <!-- Responsive Embedded Video Player -->
        <div class="video-card">
          <div id="yt-player"></div>
        </div>

        <!-- VISUAL CHAPTER TIMELINE SCRUBBER -->
        <div class="visual-timeline-box">
          <div class="timeline-title-row">
            <span>Lecture Roadmap Timeline</span>
            <span id="videoDurationLabel">50:14 Total</span>
          </div>
          
          <div class="timeline-track">
            <div class="track-seg seg-1" onclick="seekVideo(0)" title="00:00 - Introduction & Motivation"></div>
            <div class="track-seg seg-2" onclick="seekVideo(372)" title="06:12 - Neural Networks & Transformers"></div>
            <div class="track-seg seg-3" onclick="seekVideo(1403)" title="23:23 - Foundation Models & LLMs"></div>
            <div class="track-seg seg-4" onclick="seekVideo(2289)" title="38:09 - RAG & Vector Databases"></div>
            <div class="track-seg seg-5" onclick="seekVideo(2700)" title="45:00 - Career Paths & Fine-Tuning"></div>
          </div>

          <div class="timeline-chips">
            <div class="time-chip" onclick="seekVideo(0)">
              <span class="chip-dot" style="background: #3b82f6;"></span>
              <span>00:00 Intro</span>
            </div>
            <div class="time-chip" onclick="seekVideo(372)">
              <span class="chip-dot" style="background: #8b5cf6;"></span>
              <span>06:12 Deep Learning</span>
            </div>
            <div class="time-chip" onclick="seekVideo(1403)">
              <span class="chip-dot" style="background: #ec4899;"></span>
              <span>23:23 LLMs</span>
            </div>
            <div class="time-chip" onclick="seekVideo(2289)">
              <span class="chip-dot" style="background: #f97316;"></span>
              <span>38:09 RAG & Vectors</span>
            </div>
            <div class="time-chip" onclick="seekVideo(2700)">
              <span class="chip-dot" style="background: #10b981;"></span>
              <span>45:00 Fine-Tuning</span>
            </div>
          </div>
        </div>

        <!-- Center Tabs Navigation -->
        <div class="center-tabs-nav">
          <button class="c-tab-btn active" id="tabBtnRoadmap" onclick="switchCenterTab('roadmap')">
            <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="3" y="3" width="7" height="7"/><rect x="14" y="3" width="7" height="7"/><rect x="14" y="14" width="7" height="7"/><rect x="3" y="14" width="7" height="7"/></svg>
            <span>Visual Concept Roadmap</span>
          </button>
          <button class="c-tab-btn" id="tabBtnTranscript" onclick="switchCenterTab('transcript')">
            <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><line x1="8" y1="6" x2="21" y2="6"/><line x1="8" y1="12" x2="21" y2="12"/><line x1="8" y1="18" x2="21" y2="18"/><line x1="3" y1="6" x2="3.01" y2="6"/><line x1="3" y1="12" x2="3.01" y2="12"/><line x1="3" y1="18" x2="3.01" y2="18"/></svg>
            <span>Interactive Transcript</span>
          </button>
          <button class="c-tab-btn" id="tabBtnNotes" onclick="switchCenterTab('notes')">
            <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 20h9"/><path d="M16.5 3.5a2.121 2.121 0 0 1 3 3L7 19l-4 1 1-4L16.5 3.5z"/></svg>
            <span>Lecture Notes & Insights</span>
          </button>
        </div>

        <!-- TAB 1: VISUAL CONCEPT ROADMAP -->
        <div id="paneRoadmap" class="notes-document">
          <div class="concept-grid">
            
            <div class="concept-card">
              <div class="c-card-top">
                <span class="c-card-title">1. Neural Nets & Transformers</span>
                <span style="font-size: 0.72rem; font-weight: 700; color: #2563eb; background: #eff6ff; padding: 2px 6px; border-radius: 4px;">Stage 1</span>
              </div>
              <div class="c-card-desc">
                Foundations of deep learning, self-attention, encoder-decoder architectures, and dense vector embeddings.
              </div>
              <div class="c-card-footer">
                <span style="color: var(--text-muted);">Timestamp: 06:12</span>
                <button class="btn-card-seek" onclick="seekVideo(372)">▶ Watch 06:12</button>
              </div>
            </div>

            <div class="concept-card purple">
              <div class="c-card-top">
                <span class="c-card-title">2. Foundation Models & LLMs</span>
                <span style="font-size: 0.72rem; font-weight: 700; color: #8b5cf6; background: #f5f3ff; padding: 2px 6px; border-radius: 4px;">Stage 2</span>
              </div>
              <div class="c-card-desc">
                Large language models, prompt engineering, open-source weights (Llama, Mistral), and token generation.
              </div>
              <div class="c-card-footer">
                <span style="color: var(--text-muted);">Timestamp: 23:23</span>
                <button class="btn-card-seek" onclick="seekVideo(1403)">▶ Watch 23:23</button>
              </div>
            </div>

            <div class="concept-card orange">
              <div class="c-card-top">
                <span class="c-card-title">3. Retrieval-Augmented Gen (RAG)</span>
                <span style="font-size: 0.72rem; font-weight: 700; color: #ea580c; background: #fff7ed; padding: 2px 6px; border-radius: 4px;">Stage 3</span>
              </div>
              <div class="c-card-desc">
                Chunking strategies, multilingual vector embeddings, vector databases (Qdrant), and grounded prompt synthesis.
              </div>
              <div class="c-card-footer">
                <span style="color: var(--text-muted);">Timestamp: 38:09</span>
                <button class="btn-card-seek" onclick="seekVideo(2289)">▶ Watch 38:09</button>
              </div>
            </div>

            <div class="concept-card green">
              <div class="c-card-top">
                <span class="c-card-title">4. Fine-Tuning & Customization</span>
                <span style="font-size: 0.72rem; font-weight: 700; color: #10b981; background: #ecfdf5; padding: 2px 6px; border-radius: 4px;">Stage 4</span>
              </div>
              <div class="c-card-desc">
                Supervised fine-tuning (SFT), LoRA, parameter-efficient fine-tuning (PEFT), and domain-specific LLM adaptation.
              </div>
              <div class="c-card-footer">
                <span style="color: var(--text-muted);">Timestamp: 40:15</span>
                <button class="btn-card-seek" onclick="seekVideo(2415)">▶ Watch 40:15</button>
              </div>
            </div>

          </div>
        </div>

        <!-- TAB 2: TRANSCRIPT -->
        <div id="paneTranscript" class="notes-document" style="display: none;">
          <div class="transcript-toolbar">
            <input type="text" id="transcriptSearchInput" class="transcript-search" placeholder="Search spoken keywords in transcript..." oninput="filterTranscript()">
            <span id="transcriptMeta" style="font-size: 0.76rem; color: var(--text-muted); white-space: nowrap;">915 segments</span>
          </div>

          <div class="transcript-feed" id="transcriptFeed">
            <div class="transcript-item">
              <span class="ts-tag" onclick="seekVideo(0)">00:00</span>
              <span>Loading transcript for selected lecture...</span>
            </div>
          </div>
        </div>

        <!-- TAB 3: NOTES CANVAS -->
        <div id="paneNotes" class="notes-document" style="display: none;">
          <div style="background: #ffffff; border: 1px solid var(--border); border-radius: 10px; padding: 18px;">
            <h4 style="font-size: 0.95rem; margin-bottom: 8px;">Key Lecture Takeaways</h4>
            <ul style="padding-left: 20px; font-size: 0.88rem; line-height: 1.7; color: #334155;">
              <li><strong>The Two Career Paths:</strong> Builder side (building with LLMs & RAG) vs Trainer side (pre-training & fine-tuning models).</li>
              <li><strong>Why RAG Matters:</strong> Hallucination reduction, private domain knowledge injection, and instant up-to-date retrieval without retraining.</li>
              <li><strong>Prerequisites:</strong> Python proficiency, vector math basics, and an understanding of modern transformer attention.</li>
            </ul>
          </div>
        </div>

      </div>
    </main>

    <!-- 3. RIGHT COLUMN (Unriddle AI Chat Assistant) -->
    <aside class="chat-col">
      <div class="chat-header">
        <div class="chat-tab active">Private</div>
        <div class="chat-tab">Team</div>
      </div>

      <div class="chat-scroll" id="chatScroll">
        
        <!-- Welcome Message -->
        <div class="msg-block">
          <div class="msg-sender">
            <span class="sender-dot user"></span>
            <span>ADARSH</span>
          </div>
          <div class="msg-content">
            How is the Generative AI roadmap structured in this lecture, and what concepts are covered for foundational deep learning?
          </div>
        </div>

        <div class="msg-block">
          <div class="msg-sender">
            <span class="sender-dot assistant"></span>
            <span>UNRIDDLE</span>
          </div>
          <div class="msg-content">
            The instructor explains that Generative AI builds upon traditional Machine Learning and Deep Learning. First, you start with neural network foundations <span class="timestamp-pill" onclick="seekVideo(372)">06:12</span>, where transformer architecture and attention mechanisms become essential. 
            <br><br>
            Next, the curriculum transitions to foundation models and LLMs <span class="timestamp-pill" onclick="seekVideo(1403)">23:23</span>, covering how to work with large language models for text generation and enterprise applications.
            <br><br>
            Finally, the roadmap divides into builder and user sides, highlighting Retrieval-Augmented Generation (RAG) <span class="timestamp-pill" onclick="seekVideo(2289)">38:09</span> and fine-tuning techniques to adapt models to specialized private knowledge bases.
          </div>
        </div>
      </div>

      <!-- Quick Prompt Suggestion Chips -->
      <div style="padding: 0 18px 4px 18px;">
        <div class="quick-chips-row">
          <div class="q-chip" onclick="insertPrompt('What is RAG and why is it important in this roadmap?')">💡 What is RAG?</div>
          <div class="q-chip" onclick="insertPrompt('Deep learning aur Transformers ke baare mein kya bataya gaya hai?')">⚡ Transformers</div>
          <div class="q-chip" onclick="insertPrompt('Explain the Builder vs User career track in Hinglish')">🎯 Builder vs User</div>
        </div>
      </div>

      <!-- Floating Bottom Input Card -->
      <div class="chat-input-wrapper">
        <div class="input-card">
          <textarea id="questionInput" class="chat-textarea" placeholder="Ask the document a question..." onkeydown="handleKey(event)"></textarea>
          <div class="input-toolbar">
            <div class="toolbar-left">
              <select id="modelSelect" class="pill-select">
                <option value="gpt-4o-mini">GPT-4o mini ▾</option>
                <option value="claude-3-5-sonnet">Claude 3.5 Sonnet ▾</option>
                <option value="local-rag">Local RAG Engine ▾</option>
              </select>
              <select id="lengthSelect" class="pill-select">
                <option value="120 words">120 words ▾</option>
                <option value="detailed">Detailed ▾</option>
                <option value="concise">Concise ▾</option>
              </select>
            </div>
            <div class="toolbar-right">
              <button class="btn-clip" title="Attach context">
                <svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" stroke-width="2"><path d="M21.44 11.05l-9.19 9.19a6 6 0 0 1-8.49-8.49l9.19-9.19a4 4 0 0 1 5.66 5.66l-9.2 9.19a2 2 0 0 1-2.83-2.83l8.49-8.48"/></svg>
              </button>
              <button class="btn-send" onclick="sendQuestion()" title="Send question">
                <svg viewBox="0 0 24 24" width="13" height="13" fill="none" stroke="currentColor" stroke-width="2.5"><line x1="12" y1="19" x2="12" y2="5"/><polyline points="5 12 12 5 19 12"/></svg>
              </button>
            </div>
          </div>
        </div>
      </div>
    </aside>

  </div>

  <!-- ADD YOUTUBE LECTURE MODAL -->
  <div class="modal-overlay" id="addModal">
    <div class="modal-card">
      <h2>Add YouTube Lecture</h2>
      <p>Enter any YouTube video or playlist URL. Audio will be transcribed and indexed into the local vector database.</p>
      <input type="text" id="modalUrlInput" class="modal-input" placeholder="https://www.youtube.com/watch?v=..." value="https://youtu.be/pSVk-5WemQ0">
      
      <div id="modalStatus" class="modal-status"></div>

      <div class="modal-actions">
        <button class="btn-ghost" onclick="closeAddModal()">Cancel</button>
        <button class="btn-primary" id="btnStartIngest" onclick="submitIngestion()">Ingest Video</button>
      </div>
    </div>
  </div>

  <!-- YouTube Iframe API Script -->
  <script src="https://www.youtube.com/iframe_api"></script>

  <script>
    let currentVideoId = 'pSVk-5WemQ0';
    let currentVideoTitle = 'GenAI Roadmap for Beginners | CampusX';
    let ytPlayer = null;
    let isPlayerReady = false;
    let allSegments = [];

    // YouTube API Ready callback
    function onYouTubeIframeAPIReady() {
      initPlayer(currentVideoId);
    }

    function initPlayer(videoId, startSeconds = 0) {
      if (ytPlayer && ytPlayer.loadVideoById) {
        ytPlayer.loadVideoById({ videoId: videoId, startSeconds: startSeconds });
      } else {
        ytPlayer = new YT.Player('yt-player', {
          videoId: videoId,
          playerVars: {
            'playsinline': 1,
            'rel': 0,
            'start': startSeconds
          },
          events: {
            'onReady': () => { isPlayerReady = true; }
          }
        });
      }
    }

    function seekVideo(seconds) {
      if (ytPlayer && ytPlayer.seekTo) {
        ytPlayer.seekTo(seconds, true);
        ytPlayer.playVideo();
      }
    }

    function formatTime(secs) {
      const s = Math.floor(secs);
      const m = Math.floor(s / 60);
      const rem = s % 60;
      return `${String(m).padStart(2, '0')}:${String(rem).padStart(2, '0')}`;
    }

    // Switch between center tabs (roadmap / transcript / notes)
    function switchCenterTab(tabName) {
      document.getElementById('tabBtnRoadmap').classList.toggle('active', tabName === 'roadmap');
      document.getElementById('tabBtnTranscript').classList.toggle('active', tabName === 'transcript');
      document.getElementById('tabBtnNotes').classList.toggle('active', tabName === 'notes');

      document.getElementById('paneRoadmap').style.display = tabName === 'roadmap' ? 'block' : 'none';
      document.getElementById('paneTranscript').style.display = tabName === 'transcript' ? 'block' : 'none';
      document.getElementById('paneNotes').style.display = tabName === 'notes' ? 'block' : 'none';
    }

    // Load available videos into sidebar
    async function fetchVideos() {
      try {
        const res = await fetch('/api/videos');
        const videos = await res.json();
        const container = document.getElementById('videoSidebarList');
        if (!videos || videos.length === 0) return;

        container.innerHTML = '';
        videos.forEach(v => {
          const item = document.createElement('div');
          item.className = 'doc-item' + (v.video_id === currentVideoId ? ' selected' : '');
          item.innerHTML = `
            <svg class="nav-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/></svg>
            <span style="overflow: hidden; text-overflow: ellipsis;">${v.title}</span>
          `;
          item.onclick = () => selectVideo(v.video_id, v.title);
          container.appendChild(item);
        });

        loadTranscript(currentVideoId);
      } catch (e) {
        console.error('Error fetching videos:', e);
      }
    }

    function selectVideo(videoId, title) {
      currentVideoId = videoId;
      currentVideoTitle = title;
      document.getElementById('lectureTitleInput').value = title;
      initPlayer(videoId, 0);
      fetchVideos();
      loadTranscript(videoId);
    }

    // Load transcript in the center column
    async function loadTranscript(videoId) {
      const feed = document.getElementById('transcriptFeed');
      const meta = document.getElementById('transcriptMeta');
      try {
        const res = await fetch(`/api/videos/${videoId}/transcript`);
        if (!res.ok) {
          feed.innerHTML = '<div class="transcript-item">Transcript processing in progress or not found.</div>';
          return;
        }
        const data = await res.json();
        allSegments = data.segments || [];
        meta.textContent = `${allSegments.length} segments • Click timestamp to scrub`;
        renderTranscript(allSegments);
      } catch (err) {
        console.error('Error loading transcript:', err);
      }
    }

    function renderTranscript(segments) {
      const feed = document.getElementById('transcriptFeed');
      feed.innerHTML = '';
      segments.slice(0, 150).forEach(seg => {
        const row = document.createElement('div');
        row.className = 'transcript-item';
        row.innerHTML = `
          <span class="ts-tag" onclick="seekVideo(${Math.floor(seg.start)})">${formatTime(seg.start)}</span>
          <span>${seg.text}</span>
        `;
        feed.appendChild(row);
      });
    }

    function filterTranscript() {
      const q = document.getElementById('transcriptSearchInput').value.toLowerCase().trim();
      if (!q) {
        renderTranscript(allSegments);
        return;
      }
      const filtered = allSegments.filter(s => s.text.toLowerCase().includes(q));
      renderTranscript(filtered);
    }

    function insertPrompt(text) {
      const input = document.getElementById('questionInput');
      input.value = text;
      input.focus();
    }

    // Send question to RAG
    async function sendQuestion() {
      const input = document.getElementById('questionInput');
      const q = input.value.trim();
      if (!q) return;

      const chatScroll = document.getElementById('chatScroll');

      // Append user message
      const userBlock = document.createElement('div');
      userBlock.className = 'msg-block';
      userBlock.innerHTML = `
        <div class="msg-sender">
          <span class="sender-dot user"></span>
          <span>ADARSH</span>
        </div>
        <div class="msg-content">${q}</div>
      `;
      chatScroll.appendChild(userBlock);
      input.value = '';
      chatScroll.scrollTop = chatScroll.scrollHeight;

      // Append assistant placeholder
      const assistBlock = document.createElement('div');
      assistBlock.className = 'msg-block';
      assistBlock.innerHTML = `
        <div class="msg-sender">
          <span class="sender-dot assistant"></span>
          <span>UNRIDDLE</span>
        </div>
        <div class="msg-content" id="tempAssistMsg">Retrieving lecture context and formulating answer...</div>
      `;
      chatScroll.appendChild(assistBlock);
      chatScroll.scrollTop = chatScroll.scrollHeight;

      try {
        const res = await fetch('/api/ask', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ question: q, video_id: currentVideoId, top_k: 5 })
        });
        const data = await res.json();
        const contentDiv = assistBlock.querySelector('.msg-content');

        let answerText = data.answer || '';
        
        // Regex to convert [MM:SS] or MM:SS citations into pills
        let formattedHtml = answerText.replace(/\\[(\\d{1,2}:\\d{2})\\]/g, (match, timeStr) => {
          const parts = timeStr.split(':').map(Number);
          const totalSec = parts[0] * 60 + parts[1];
          return `<span class="timestamp-pill" onclick="seekVideo(${totalSec})">${timeStr}</span>`;
        });

        // Add direct citations if available
        if (data.citations && data.citations.length > 0) {
          formattedHtml += '<div style="margin-top: 14px; font-size: 0.82rem; color: var(--text-muted);">';
          formattedHtml += '<strong>Direct Video Timestamps:</strong><br>';
          data.citations.forEach(c => {
            formattedHtml += `<span class="timestamp-pill" onclick="seekVideo(${Math.floor(c.start)})" style="margin-top: 4px;">▶ ${c.formatted_range}</span> `;
          });
          formattedHtml += '</div>';
        }

        contentDiv.innerHTML = formattedHtml;
        chatScroll.scrollTop = chatScroll.scrollHeight;
      } catch (err) {
        assistBlock.querySelector('.msg-content').textContent = 'Error: ' + err.message;
      }
    }

    function handleKey(e) {
      if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        sendQuestion();
      }
    }

    // Modal logic
    function openAddModal() {
      document.getElementById('addModal').style.display = 'flex';
      document.getElementById('modalStatus').style.display = 'none';
    }
    function closeAddModal() {
      document.getElementById('addModal').style.display = 'none';
    }

    async function submitIngestion() {
      const url = document.getElementById('modalUrlInput').value.trim();
      const status = document.getElementById('modalStatus');
      const btn = document.getElementById('btnStartIngest');
      if (!url) return;

      status.style.display = 'block';
      status.textContent = 'Ingesting video into local Qdrant...';
      btn.disabled = true;

      try {
        const res = await fetch('/api/videos/ingest', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ url })
        });
        const data = await res.json();
        status.innerHTML = `<strong>${data.status}</strong>: ${data.message || data.title}`;
        btn.disabled = false;
        fetchVideos();
        setTimeout(closeAddModal, 1500);
      } catch (err) {
        status.textContent = 'Error: ' + err.message;
        btn.disabled = false;
      }
    }

    // Initialize
    fetchVideos();
  </script>
</body>
</html>
"""
