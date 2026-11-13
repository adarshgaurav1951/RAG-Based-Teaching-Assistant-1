"""
Configuration module for RAG-Based AI Teaching Assistant.
Centralizes environment variables, model choices, file paths, and thresholds.
"""

from __future__ import annotations

import os
from pathlib import Path
from dotenv import load_dotenv

# Base directory is the project root (where this file's parent's parent resides)
BASE_DIR = Path(__file__).resolve().parent.parent

# Load environment variables from .env
load_dotenv(dotenv_path=BASE_DIR / ".env", override=False)

# LLM Provider Configuration (Backend only; never expose to clients)
LLM_API_KEY: str = os.getenv("LLM_API_KEY", "").strip()
LLM_BASE_URL: str = os.getenv("LLM_BASE_URL", "").strip() or None
LLM_MODEL: str = os.getenv("LLM_MODEL", "gpt-4o-mini").strip()

# Speech-to-Text (ASR)
WHISPER_MODEL: str = os.getenv("WHISPER_MODEL", "tiny").strip()
WHISPER_LANGUAGE: str = os.getenv("WHISPER_LANGUAGE", "auto").strip()
WHISPER_CPU_THREADS: int = int(os.getenv("WHISPER_CPU_THREADS", "2"))

# Embeddings
EMBED_MODEL: str = os.getenv(
    "EMBED_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
).strip()
EMBED_DIM: int = int(os.getenv("EMBED_DIM", "384"))
EMBED_BATCH_SIZE: int = int(os.getenv("EMBED_BATCH_SIZE", "4"))

# Chunking & Retrieval
CHUNK_SECONDS: float = float(os.getenv("CHUNK_SECONDS", "75.0"))
TOP_K: int = int(os.getenv("TOP_K", "5"))
RELEVANCE_THRESHOLD: float = float(os.getenv("RELEVANCE_THRESHOLD", "0.35"))
COLLECTION_NAME: str = os.getenv("COLLECTION", "lectures").strip()
PROCESSING_VERSION: str = "1.0.0"

# Paths
DATA_DIR = BASE_DIR / os.getenv("DATA_DIR", "data")
AUDIO_DIR = BASE_DIR / os.getenv("AUDIO_DIR", "data/audio")
TRANSCRIPT_DIR = BASE_DIR / os.getenv("TRANSCRIPT_DIR", "data/transcripts")
QDRANT_PATH = BASE_DIR / os.getenv("QDRANT_PATH", "data/qdrant")
SQLITE_PATH = BASE_DIR / os.getenv("SQLITE_PATH", "data/app.db")

# Ensure required local directories exist
for p in [DATA_DIR, AUDIO_DIR, TRANSCRIPT_DIR, QDRANT_PATH]:
    p.mkdir(parents=True, exist_ok=True)

