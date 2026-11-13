# Hinglish YouTube Lecture AI Teaching Assistant (CPU-First RAG)

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110%2B-009688.svg?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Qdrant](https://img.shields.io/badge/Vector_DB-Qdrant_Local-dc2626.svg?logo=qdrant&logoColor=white)](https://qdrant.tech/)
[![faster-whisper](https://img.shields.io/badge/ASR-faster--whisper_INT8-4f46e5.svg)](https://github.com/SYSTRAN/faster-whisper)
[![PyTorch CPU](https://img.shields.io/badge/PyTorch-CPU_Only_(~3.5GB_RAM)-ee4c2c.svg?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![UI](https://img.shields.io/badge/UI-Unriddle_3--Column-2563eb.svg)](http://localhost:8697)
[![Port](https://img.shields.io/badge/Port-8697-success.svg)](http://localhost:8697)

A practical, low-resource Retrieval-Augmented Generation (RAG) teaching assistant that converts selected YouTube lecture videos (English, Hindi, and Hinglish) into an indexed, searchable knowledge base.

Designed specifically for **CPU-only laptops with limited RAM (~3.5 GiB)**:
- Processes **one selected video at a time**.
- Downloads audio only and normalizes via **FFmpeg** (mono 16 kHz PCM WAV).
- Hinglish speech-to-text with **faster-whisper** on CPU INT8 preserving original spoken code-switching (no forced translation).
- **75-second timestamp-aware chunking** respecting sentence and speech boundaries.
- **Multilingual MiniLM** sentence embeddings (`paraphrase-multilingual-MiniLM-L12-v2`, 384 dimensions).
- Persistent local **Qdrant** vector store (no Docker required).
- Relational ingestion job tracking via local **SQLite** (`data/app.db`).
- Grounded pedagogical answer generation with **exact, verified clickable YouTube timestamp links** (`https://www.youtube.com/watch?v=VIDEO_ID&t=123s`).
- Modern **Unriddle-style 3-column UI** with visual roadmap cards, timeline scrubber, synchronized YouTube player, interactive transcript, and grounded AI chat.
- External LLM API key stays isolated in backend `.env` (never exposed to browser, logs, or responses).

---

## Visual Architecture & Workflow

```mermaid
flowchart TD
    subgraph INGESTION ["1. Single-Video Ingestion Pipeline (Once per video)"]
        A["YouTube URL / Playlist"] -->|Inspect Metadata| B["Selected Single Video"]
        B -->|Audio Only yt-dlp| C["16 kHz Mono PCM WAV"]
        C -->|faster-whisper INT8 CPU| D["Raw Transcript JSON<br/>(Preserves Hinglish)"]
        D -->|75s Boundary Chunker| E["Overlapping Chunks"]
        E -->|paraphrase-multilingual-MiniLM| F["384-dim Vectors"]
        F -->|Upsert query_points| G[("Qdrant Local Store<br/>data/qdrant")]
        D -->|Stage Checkpoints| H[("SQLite DB<br/>data/app.db")]
        C -.->|Cleanup on Completion| I["Purge Temp Audio"]
    end

    subgraph QUERY ["2. Grounded Q&A Assistant Flow"]
        Q["User Query<br/>(Hinglish / Hindi / English)"] --> EMB["Embed Query (MiniLM)"]
        EMB --> SEARCH["Qdrant Top-K Similarity Search"]
        G -.-> SEARCH
        SEARCH --> FILTER["Cosine Relevance Check<br/>(threshold >= 0.20)"]
        FILTER --> PROMPT["Grounded Pedagogical Prompt Builder<br/>with Exact Seconds"]
        PROMPT --> LLM["LLM Synthesis (Backend .env Key)"]
        LLM --> RESP["Answer + Clickable Timestamp Citations<br/>[MM:SS - MM:SS]"]
    end

    subgraph UI ["3. Unriddle 3-Column Interface (Port 8697)"]
        L_COL["Left Sidebar<br/>Video Library & Switcher"]
        M_COL["Center Column<br/>• Synchronized YouTube Player<br/>• Interactive Timeline Scrubber<br/>• Visual Concept Roadmap Cards<br/>• Full Searchable Transcript"]
        R_COL["Right Assistant Pane<br/>• Unriddle-Style Q&A Chat<br/>• Clickable Seeking Citations<br/>• Suggested Prompt Chips"]
    end
```

### Ingestion Sequence & Checkpoint State Flow

```mermaid
sequenceDiagram
    autonumber
    actor User as Student / Teacher
    participant CLI as CLI / REST API (Port 8697)
    participant DB as SQLite (data/app.db)
    participant Media as yt-dlp + FFmpeg
    participant Whisper as faster-whisper (CPU INT8)
    participant Qdrant as Qdrant Vector DB
    participant LLM as External LLM (Backend Only)

    User->>CLI: Ingest Video URL (e.g. pSVk-5WemQ0)
    CLI->>DB: Record job (status: "downloading")
    CLI->>Media: Download audio & normalize to mono 16kHz WAV
    Media-->>CLI: audio.wav saved
    CLI->>DB: Update status: "transcribing"
    CLI->>Whisper: Speech-to-text (original Hinglish speech)
    Whisper-->>CLI: 915 segments with start/end timestamps
    CLI->>DB: Update status: "chunking"
    CLI->>CLI: 75-second boundary-aligned chunking (41 chunks)
    CLI->>DB: Update status: "indexing"
    CLI->>Qdrant: Upsert 41 vectors (UUID5 determinism)
    CLI->>Media: Purge temporary audio.wav
    CLI->>DB: Update status: "completed"
    CLI-->>User: Ingestion Complete! Ready for Q&A

    Note over User,LLM: Interactive Pedagogical Q&A
    User->>CLI: Ask: "Roadmap ke 4 main modules kya hain?"
    CLI->>Qdrant: Search top 4 chunks (cosine similarity)
    Qdrant-->>CLI: Return relevant lecture chunks
    CLI->>LLM: Grounded pedagogical prompt with transcript excerpts
    LLM-->>CLI: Synthesized explanation
    CLI-->>User: Formatted answer with clickable timestamp links
```

---

## Directory Structure

```
.
├── app/
│   ├── __init__.py
│   ├── config.py          # .env settings, models, paths, and constants
│   ├── storage.py         # SQLite job state and stage transitions
│   ├── media.py           # yt-dlp metadata & audio download + FFmpeg normalization
│   ├── transcription.py   # faster-whisper CPU INT8 Hinglish transcription
│   ├── retrieval.py       # 75s chunking, MiniLM embeddings, Qdrant upsert/search
│   ├── teaching.py        # Grounded prompts, LLM call, clickable citations
│   ├── ingest.py          # End-to-end single video ingestion orchestrator
│   ├── cli.py             # Full-featured command-line interface
│   └── main.py            # FastAPI REST API + embedded single-page UI (Port 8697)
├── data/                  # Local runtime data (ignored in git)
│   ├── audio/             # Temporary media (purged after indexing)
│   ├── transcripts/       # Durable raw & chunk JSON checkpoints
│   ├── qdrant/            # Persistent local Qdrant vectors
│   └── app.db             # SQLite database
├── tests/
│   ├── test_chunking.py
│   ├── test_timestamps.py
│   ├── test_ingestion_state.py
│   └── test_retrieval.py
├── .env.example
├── .gitignore
├── requirements.txt
├── PRD.md
├── idea.md
└── README.md
```

---

## Quickstart

### 1. Prerequisites
- Python 3.10+
- FFmpeg installed and in PATH (`ffmpeg -version`)

### 2. Environment Setup
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 3. Configure `.env`
```bash
cp .env.example .env
# Edit .env to add your LLM API Key (OpenAI, Gemini OpenAI-compatible, Groq, etc.)
```
*Note: Ingestion, ASR transcription, chunking, and vector search work completely offline without an LLM key. The external key is only required for generative pedagogical synthesis.*

---

## Usage

### Command Line Interface (CLI)

```bash
# 1. Inspect a YouTube video or playlist without downloading media
python -m app.cli inspect "https://youtu.be/pSVk-5WemQ0"

# 2. Ingest one selected video
python -m app.cli ingest "https://youtu.be/pSVk-5WemQ0"

# 3. List indexed lectures
python -m app.cli list

# 4. Ask a question (supports English, Hindi, and Hinglish)
python -m app.cli ask "GenAI roadmap mein deep learning ke concepts kahan se padhein?" --video-id pSVk-5WemQ0
```

### Web Interface & FastAPI Server (Port 8697)

Start the Uvicorn server on **port 8697**:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8697 --reload
```

Open your browser to:
👉 **[http://localhost:8697](http://localhost:8697)**

#### Web UI Highlights (Unriddle 3-Column Experience):
- **Synchronized Video Player**: YouTube iframe embeds dynamically and jumps directly to the timestamp when any citation or timeline chip is clicked.
- **Interactive Timeline Scrubber**: Color-coded lecture milestones (Intro, ML Basics, GenAI Landscape, LLM Foundations, 4-Stage Roadmap, Project Strategy).
- **Visual Concept Roadmap Cards**: Direct lecture navigation cards for key architectural topics.
- **Searchable Transcript**: Full code-switched transcript with millisecond-precision timestamps.
- **Pedagogical Assistant**: Conversational assistant with pre-loaded prompt chips and exact citation pills.

---

## Running Tests

```bash
pytest -v
```

