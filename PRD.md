# Product Requirements Document (PRD)
## RAG-Based AI Teaching Assistant for Hinglish YouTube Lectures

- **Document Version:** 1.0.0
- **Status:** Approved / Active Implementation
- **Author:** AI Systems Engineering Team
- **Target Repository:** `RAG Base Project`
- **Reference Blueprint:** `idea.md`
- **Target Video for Benchmark:** `https://youtu.be/pSVk-5WemQ0` (CampusX: *GenAI Roadmap for Beginners | End-to-End GenAI Course 2025*)

---

## 1. Executive Summary & Problem Statement

### 1.1 Problem Statement
Online technical education in India and globally heavily utilizes code-switched Hindi-English ("Hinglish") video lectures on YouTube. While rich in pedagogical value, video lectures are inherently linear and difficult to navigate:
1. Students spend excessive time scrubbing through 40–90 minute videos searching for specific concept explanations.
2. Existing transcription and RAG tools often struggle with Hinglish code-switching, either mistranslating Hindi idioms or forcing artificial English translation.
3. Students lack grounded answers linked to the exact timestamp where the instructor explained the topic.
4. Most modern generative AI solutions assume heavy GPU infrastructure or cloud microservices, making local deployment impractical on resource-constrained laptops (e.g., CPU-only, ~3.5 GiB RAM).

### 1.2 Product Vision
Build a resource-efficient, local-first Retrieval-Augmented Generation (RAG) teaching assistant that converts selected YouTube lecture videos into an indexed, searchable knowledge base. The assistant preserves original Hinglish speech, respects speaker timing with ~75-second chunks, executes on CPU-only hardware, and delivers grounded pedagogical answers with exact, clickable YouTube timestamp citations.

---

## 2. Personas & Core Use Cases

### 2.1 Target Personas
- **Student / Self-Learner:** Studying programming or AI/ML via YouTube tutorials; needs quick conceptual clarity and direct links to replay lecture segments.
- **Educator / TA:** Recommends verified lecture clips to students for revision and doubt resolution.

### 2.2 Core User Stories
- **US-1 (Inspect & Select):** As a student, I want to input a YouTube playlist or video URL and view video metadata (title, duration, ID) so that I can select exactly one lecture to index without accidentally downloading entire playlists.
- **US-2 (Hinglish Ingestion):** As a student, I want the system to transcribe the lecture while preserving original Hinglish technical terminology without forced English translation.
- **US-3 (Timestamped Q&A):** As a student, I want to ask questions in English, Hindi, or Hinglish and receive grounded answers citing the exact video timestamp (e.g. `[12:10 - 13:25]`) with clickable YouTube links.
- **US-4 (Resource Constrained Operation):** As a student running on a CPU-only laptop with 3.5 GiB RAM, I want the system to process lectures sequentially without running out of memory or crashing the machine.
- **US-5 (Incremental Library):** As a student, I want to add new videos over time without re-indexing previously completed lectures.

---

## 3. Product Scope & Guardrails

### 3.1 In-Scope (Phase 1 MVP)
- Single-video selection and processing per ingestion run (`noplaylist=True`).
- Low-memory audio pipeline: stream download via `yt-dlp`, mono 16 kHz PCM WAV conversion via `ffmpeg`, automatic temporary audio cleanup.
- Speech-to-text via `faster-whisper` on CPU (INT8 quantization, configurable model: `tiny` benchmark to `base`).
- Timestamp-aware chunking targeting ~75 seconds, adhering to natural sentence/speech segment boundaries.
- Local multilingual vector embeddings via `sentence-transformers` (`paraphrase-multilingual-MiniLM-L12-v2`, 384 dimensions).
- Embedded vector storage via persistent local `qdrant-client` (no Docker required).
- Relational job state tracking via local SQLite (`app.db`).
- Grounded answer generation via external OpenAI-compatible LLM API (Gemini/OpenAI/Groq) with user-supplied API key strictly isolated in backend `.env`.
- Clickable YouTube timestamp citations in standard format: `https://www.youtube.com/watch?v=VIDEO_ID&t={seconds}s`.
- Dual interfaces: Full-featured CLI and minimal, responsive FastAPI web UI.

### 3.2 Out-of-Scope (Explicit Non-Goals for Initial Release)
- Bulk automated downloading of entire playlists.
- Local generative LLM inference (Ollama/Llama.cpp) on host CPU/RAM.
- Heavy agent frameworks (LangChain, LangGraph, AutoGen).
- Multi-process concurrent local Qdrant access (Qdrant file lock protection enforced).
- User authentication, multi-tenancy, or billing infrastructure.
- Video frame extraction, OCR, or slide computer vision.

---

## 4. Hardware Operating Constraints

| Parameter | Specification & Operational Bound |
| :--- | :--- |
| **Compute** | CPU-only execution (x86_64), no GPU/CUDA acceleration required |
| **System RAM** | 3.5 GiB physical RAM + swap profile |
| **Concurrency** | Strict single-job execution queue for ingestion |
| **Whisper Config** | `device="cpu"`, `compute_type="int8"`, `cpu_threads=2`, beam size 1 |
| **Embedding Config**| Batch size $\le 4$ on CPU, vector normalization enabled |
| **Disk Management** | Audio downloaded temporarily; cleaned immediately after verified indexing |
| **Persistence** | Durable JSON checkpoints for raw transcripts and chunk structures |

---

## 5. Functional Requirements

### FR-01: Metadata Inspection & Single-Video Selection
- System accepts either a YouTube video URL or a playlist URL.
- System queries `yt-dlp` metadata extraction without downloading any audio or video stream.
- If a playlist URL is supplied, system returns video list (Index, Title, Duration, Video ID) for user selection.
- Selected video ID is validated against standard YouTube regex (`^[a-zA-Z0-9_-]{11}$`).

### FR-02: State Tracking & Idempotency
- SQLite database (`data/app.db`) tracks video status: `queued`, `fetching_metadata`, `downloading_audio`, `normalizing_audio`, `transcribing`, `chunking`, `embedding`, `indexing`, `completed`, `failed`.
- If a video is already marked `completed` with identical configuration version, system immediately returns "Already indexed" and prevents re-downloading.
- If a prior run failed or was interrupted, system provides resume/retry capability from the last valid checkpoint.

### FR-03: Audio Acquisition & Normalization
- Audio stream downloaded via `yt-dlp` with `noplaylist=True`.
- Download converted to mono 16 kHz PCM WAV format via `ffmpeg`:
  ```bash
  ffmpeg -y -i input_audio -vn -ac 1 -ar 16000 -c:a pcm_s16le normalized.wav
  ```
- Audio header and duration verified before passing to ASR using standard library validation.

### FR-04: Multilingual Hinglish Transcription
- Speech transcribed via `faster-whisper` on CPU INT8.
- Default model: `tiny` (benchmarking `base` where memory permits).
- `task="transcribe"` strictly enforced (no forced translation).
- Multilingual mode preserves original spoken code-switched Hindi-English text.
- Word timestamps and segment timestamps recorded with millisecond precision.
- Raw transcript saved to durable JSON checkpoint: `data/transcripts/{video_id}_raw.json`.

### FR-05: 75-Second Timestamp-Aware Chunking
- Segments assembled into chunks targeting ~75 seconds window.
- Chunks respect complete sentence and speech segment boundaries (no mid-sentence cuts).
- Actual starting time of first segment and ending time of last segment recorded as chunk `start` and `end`.
- Chunk JSON checkpoint saved to `data/transcripts/{video_id}_chunks.json`.

### FR-06: Multilingual Vector Embeddings
- Model: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`.
- Vector dimension: 384 dimensions, normalized unit vectors.
- Embeddings generated in CPU batches of 2–4 to prevent memory spikes.

### FR-07: Persistent Vector Storage in Qdrant
- Qdrant initialized in persistent local mode: `data/qdrant`.
- Single collection `lectures` with Cosine distance.
- Deterministic point IDs generated via UUID5 derived from `video_id:chunk_index:version` to ensure idempotent upserts.
- Payload includes: `chunk_id`, `video_id`, `playlist_id`, `title`, `video_url`, `chunk_index`, `start`, `end`, `text`, `language`, `embedding_model`.

### FR-08: Incremental Multi-Video Knowledge Base
- Ingesting a new video appends new vectors to Qdrant and updates SQLite without altering or removing prior indexed videos.
- Search queries can be scoped to a single `video_id` or across all indexed videos.

### FR-09: Semantic Retrieval & Relevance Guardrails
- Questions embedded using the exact same MiniLM model and normalization.
- Top-$K$ semantic retrieval ($K=5$) from Qdrant with optional video filter.
- Deduplication of overlapping evidence chunks.
- Relevance check: if evidence similarity is below threshold or empty, system returns an honest pedagogical disclaimer: *"The indexed lecture material does not contain sufficient information to answer this question."*

### FR-10: Grounded Answer Generation & Clickable Timestamps
- Grounded prompt ensures LLM acts as an AI teaching assistant answering strictly from retrieved lecture context.
- Answers support student's query language (English, Hindi, or Hinglish).
- LLM outputs structured citations mapping directly to stored chunk timestamps.
- Formatted output provides clickable YouTube links: `https://www.youtube.com/watch?v={video_id}&t={start_seconds}s`.

---

## 6. Non-Functional Requirements (NFRs)

- **NFR-01 (Resource Safety):** Peak memory consumption during transcription and embedding must remain within available system memory. Single-job locking prevents concurrent model allocations.
- **NFR-02 (API Security):** External LLM API credentials are read exclusively from `.env` on the backend. No secret shall ever be transmitted to the frontend, rendered in HTML, logged to console, or committed to Git.
- **NFR-03 (Data Durability):** Transcripts and chunk manifests persist on disk to eliminate re-transcription costs if re-indexing or re-chunking is required.
- **NFR-04 (Storage Cleanliness):** Normalized WAV audio files are purged immediately upon verified Qdrant upsert and SQLite completion status.
- **NFR-05 (Deterministic Retries):** Retrying a failed ingestion run overwrites previous partial points idempotently without duplicate point creation.

---

## 7. Data Architecture & Schema

### 7.1 SQLite Schema (`data/app.db`)
```sql
CREATE TABLE IF NOT EXISTS videos (
    video_id TEXT PRIMARY KEY,
    playlist_id TEXT,
    title TEXT NOT NULL,
    video_url TEXT NOT NULL,
    duration_seconds REAL NOT NULL,
    status TEXT NOT NULL,
    current_stage TEXT NOT NULL,
    error_message TEXT,
    asr_model TEXT NOT NULL,
    embedding_model TEXT NOT NULL,
    processing_version TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
```

### 7.2 Qdrant Payload Structure
```json
{
  "chunk_id": "pSVk-5WemQ0_0001",
  "video_id": "pSVk-5WemQ0",
  "playlist_id": null,
  "title": "GenAI Roadmap for Beginners | End-to-End GenAI Course 2025 | CampusX",
  "video_url": "https://www.youtube.com/watch?v=pSVk-5WemQ0",
  "chunk_index": 1,
  "start": 0.0,
  "end": 74.2,
  "text": "... Hinglish transcript segment ...",
  "language": "hi",
  "embedding_model": "paraphrase-multilingual-MiniLM-L12-v2"
}
```

---

## 8. User Interface Specifications

### 8.1 Command Line Interface (CLI)
- `python -m app.cli inspect <url>`: Inspects playlist or video metadata.
- `python -m app.cli ingest <url>`: Runs complete ingestion for one video.
- `python -m app.cli list`: Displays all indexed videos, durations, and statuses.
- `python -m app.cli ask "<question>" [--video-id <id>]`: Queries indexed knowledge base.
- `python -m app.cli retry <video_id>`: Resumes failed or interrupted processing.

### 8.2 FastAPI Endpoints
- `POST /api/playlists/inspect`: Extracts video list from playlist or single URL.
- `POST /api/videos/ingest`: Initiates asynchronous single-video ingestion.
- `GET /api/videos`: Lists all videos and current progress/status.
- `GET /api/videos/{video_id}`: Retrieves detailed video status and stage metrics.
- `POST /api/videos/{video_id}/retry`: Retries failed ingestion.
- `POST /api/ask`: Semantic question-answering with citation formatting.
- `GET /api/health`: Health status of system dependencies (FFmpeg, DB, Qdrant).
- `GET /`: Serves clean, responsive single-page web UI.

---

## 9. Verification & Acceptance Criteria

| Stage | Verification Criteria |
| :--- | :--- |
| **Playlist Inspection** | Inspecting a playlist outputs metadata without downloading media. |
| **Audio Acquisition** | Audio downloaded as WAV; FFmpeg normalizes to mono 16 kHz PCM. |
| **Transcription** | Hinglish speech transcribed accurately with valid timestamps on CPU. |
| **Chunking** | Target ~75 seconds; every transcript word preserved; no missing segments. |
| **Vector Indexing** | MiniLM outputs 384-dim vector; deterministic Qdrant upsert verified. |
| **Retrieval Quality** | Multilingual queries (EN, HI, Hinglish) retrieve relevant lecture chunks. |
| **Teaching Answer** | Grounded response generated with clickable timestamp links (`https://...&t=XXs`). |
| **Benchmark Run 1** | Target video `pSVk-5WemQ0` (50:14) successfully ingested, chunked, and queried. |
| **Benchmark Run 2** | Target video `_3ezSpJw2E8` (15:18) successfully ingested, chunked, and queried with verified timestamps. |

