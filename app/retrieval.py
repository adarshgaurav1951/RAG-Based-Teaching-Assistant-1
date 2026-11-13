"""
Retrieval module for RAG-Based AI Teaching Assistant.
Implements:
1. 75-second timestamp-aware transcript chunking preserving segment boundaries.
2. Multilingual MiniLM sentence embeddings on CPU.
3. Persistent local Qdrant vector storage with idempotent deterministic point IDs.
4. Semantic top-k search with optional video-scoped filtering.
"""

from __future__ import annotations

import atexit
import json
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from app.config import (
    CHUNK_SECONDS,
    COLLECTION_NAME,
    EMBED_BATCH_SIZE,
    EMBED_DIM,
    EMBED_MODEL,
    PROCESSING_VERSION,
    QDRANT_PATH,
    RELEVANCE_THRESHOLD,
    TOP_K,
    TRANSCRIPT_DIR,
)

# Global model and client caches
_EMBED_MODEL_CACHE: Any = None
_QDRANT_CLIENT_CACHE: Any = None


# -----------------------------------------------------------------------------
# 1. Chunking Logic
# -----------------------------------------------------------------------------

def chunk_transcript(
    transcript_data: Dict[str, Any],
    target_seconds: float = CHUNK_SECONDS,
) -> List[Dict[str, Any]]:
    """
    Groups raw transcript segments into chunks targeting ~75 seconds.
    Respects speech segment and sentence boundaries without slicing mid-sentence.
    """
    segments = transcript_data.get("segments", [])
    if not segments:
        return []

    video_id = transcript_data.get("video_id", "unknown")
    title = transcript_data.get("title", "")
    language = transcript_data.get("detected_language", "auto")
    video_url = transcript_data.get("video_url") or f"https://www.youtube.com/watch?v={video_id}"

    # Sort segments by start time
    sorted_segments = sorted(segments, key=lambda s: s.get("start", 0.0))

    chunks: List[Dict[str, Any]] = []
    current_segments: List[Dict[str, Any]] = []
    chunk_index = 1

    for seg in sorted_segments:
        if not current_segments:
            current_segments.append(seg)
            continue

        chunk_start = current_segments[0]["start"]
        potential_end = seg["end"]
        potential_duration = potential_end - chunk_start

        # If adding this segment pushes past target duration and we already have content
        if potential_duration > target_seconds and current_segments:
            # Finalize current chunk
            c_start = current_segments[0]["start"]
            c_end = current_segments[-1]["end"]
            c_text = " ".join(s["text"].strip() for s in current_segments if s["text"].strip())

            chunks.append(
                {
                    "chunk_id": f"{video_id}_{chunk_index:04d}",
                    "video_id": video_id,
                    "title": title,
                    "video_url": video_url,
                    "chunk_index": chunk_index,
                    "start": round(c_start, 2),
                    "end": round(c_end, 2),
                    "duration": round(c_end - c_start, 2),
                    "text": c_text,
                    "language": language,
                    "segment_count": len(current_segments),
                }
            )
            chunk_index += 1
            current_segments = [seg]
        else:
            current_segments.append(seg)

    # Finalize remaining segments
    if current_segments:
        c_start = current_segments[0]["start"]
        c_end = current_segments[-1]["end"]
        c_text = " ".join(s["text"].strip() for s in current_segments if s["text"].strip())
        chunks.append(
            {
                "chunk_id": f"{video_id}_{chunk_index:04d}",
                "video_id": video_id,
                "title": title,
                "video_url": video_url,
                "chunk_index": chunk_index,
                "start": round(c_start, 2),
                "end": round(c_end, 2),
                "duration": round(c_end - c_start, 2),
                "text": c_text,
                "language": language,
                "segment_count": len(current_segments),
            }
        )

    return chunks


def save_chunks(
    chunks: List[Dict[str, Any]],
    video_id: str,
    output_dir: Optional[Path] = None,
) -> Path:
    """Saves chunk JSON checkpoint to disk."""
    dest_dir = output_dir or TRANSCRIPT_DIR
    dest_dir.mkdir(parents=True, exist_ok=True)
    out_file = dest_dir / f"{video_id}_chunks.json"

    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)

    return out_file


def load_chunks(
    video_id: str,
    transcript_dir: Optional[Path] = None,
) -> Optional[List[Dict[str, Any]]]:
    """Loads previously saved chunk JSON checkpoint if present."""
    dest_dir = transcript_dir or TRANSCRIPT_DIR
    in_file = dest_dir / f"{video_id}_chunks.json"
    if not in_file.exists():
        return None

    with open(in_file, "r", encoding="utf-8") as f:
        return json.load(f)


# -----------------------------------------------------------------------------
# 2. Multilingual MiniLM Embeddings
# -----------------------------------------------------------------------------

def get_embed_model(model_name: str = EMBED_MODEL) -> Any:
    """Loads and caches the SentenceTransformer embedding model on CPU."""
    global _EMBED_MODEL_CACHE
    if _EMBED_MODEL_CACHE is not None:
        return _EMBED_MODEL_CACHE

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as e:
        raise RuntimeError(
            "sentence-transformers is not installed. Please install it via requirements.txt."
        ) from e

    model = SentenceTransformer(model_name, device="cpu")
    _EMBED_MODEL_CACHE = model
    return model


def embed_texts(
    texts: List[str],
    batch_size: int = EMBED_BATCH_SIZE,
    model_name: str = EMBED_MODEL,
) -> List[List[float]]:
    """Encodes a list of texts into normalized unit vector arrays on CPU."""
    if not texts:
        return []

    model = get_embed_model(model_name)
    vectors = model.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return vectors.tolist()


def embed_query(query: str, model_name: str = EMBED_MODEL) -> List[float]:
    """Encodes a single student query using the exact same model and normalization."""
    model = get_embed_model(model_name)
    vector = model.encode(query, normalize_embeddings=True, show_progress_bar=False)
    return vector.tolist()


# -----------------------------------------------------------------------------
# 3. Persistent Qdrant Storage
# -----------------------------------------------------------------------------

def get_qdrant_client() -> Any:
    """Returns a persistent local QdrantClient instance."""
    global _QDRANT_CLIENT_CACHE
    if _QDRANT_CLIENT_CACHE is not None:
        return _QDRANT_CLIENT_CACHE

    try:
        from qdrant_client import QdrantClient
    except ImportError as e:
        raise RuntimeError("qdrant-client is not installed.") from e

    client = QdrantClient(path=str(QDRANT_PATH))
    _QDRANT_CLIENT_CACHE = client
    return client


def close_qdrant_client() -> None:
    """Closes the cached Qdrant client cleanly."""
    global _QDRANT_CLIENT_CACHE
    if _QDRANT_CLIENT_CACHE is not None:
        try:
            _QDRANT_CLIENT_CACHE.close()
        except Exception:
            pass
        _QDRANT_CLIENT_CACHE = None


atexit.register(close_qdrant_client)


def ensure_collection(collection_name: str = COLLECTION_NAME, vector_dim: int = EMBED_DIM) -> None:
    """Ensures that the Qdrant collection exists with Cosine distance and appropriate dimensions."""
    from qdrant_client.http import models

    client = get_qdrant_client()
    collections = client.get_collections().collections
    existing_names = [c.name for c in collections]

    if collection_name not in existing_names:
        client.create_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(
                size=vector_dim,
                distance=models.Distance.COSINE,
            ),
        )


def generate_point_id(video_id: str, chunk_index: int, version: str = PROCESSING_VERSION) -> str:
    """Generates a stable, deterministic UUID for idempotent upserts."""
    key = f"{video_id}:{chunk_index:04d}:{version}"
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, key))


def upsert_chunks_to_qdrant(
    chunks: List[Dict[str, Any]],
    collection_name: str = COLLECTION_NAME,
    batch_size: int = EMBED_BATCH_SIZE,
) -> int:
    """
    Embeds transcript chunks and upserts points into persistent local Qdrant.
    Returns the count of points upserted.
    """
    if not chunks:
        return 0

    ensure_collection(collection_name=collection_name, vector_dim=EMBED_DIM)
    client = get_qdrant_client()
    from qdrant_client.http import models

    texts = [c["text"] for c in chunks]
    vectors = embed_texts(texts, batch_size=batch_size)

    points: List[models.PointStruct] = []
    for c, vec in zip(chunks, vectors):
        point_id = generate_point_id(c["video_id"], c["chunk_index"])
        payload = {
            "chunk_id": c["chunk_id"],
            "video_id": c["video_id"],
            "playlist_id": c.get("playlist_id"),
            "title": c.get("title", ""),
            "video_url": c.get("video_url", ""),
            "chunk_index": c["chunk_index"],
            "start": c["start"],
            "end": c["end"],
            "text": c["text"],
            "language": c.get("language", ""),
            "embedding_model": EMBED_MODEL,
            "processing_version": PROCESSING_VERSION,
        }
        points.append(models.PointStruct(id=point_id, vector=vec, payload=payload))

    # Upsert in small batches
    chunk_step = 10
    for i in range(0, len(points), chunk_step):
        batch = points[i : i + chunk_step]
        client.upsert(collection_name=collection_name, points=batch)

    return len(points)


def verify_video_indexed(
    video_id: str,
    collection_name: str = COLLECTION_NAME,
) -> Tuple[bool, int]:
    """Verifies that vector points for this video exist and are searchable in Qdrant."""
    client = get_qdrant_client()
    from qdrant_client.http import models

    filter_cond = models.Filter(
        must=[
            models.FieldCondition(
                key="video_id",
                match=models.MatchValue(value=video_id),
            )
        ]
    )

    try:
        count_res = client.count(
            collection_name=collection_name,
            count_filter=filter_cond,
            exact=True,
        )
        count = count_res.count
        return (count > 0, count)
    except Exception:
        return (False, 0)


# -----------------------------------------------------------------------------
# 4. Semantic Search & Retrieval
# -----------------------------------------------------------------------------

def search_chunks(
    query: str,
    video_id: Optional[str] = None,
    top_k: int = TOP_K,
    relevance_threshold: float = RELEVANCE_THRESHOLD,
    collection_name: str = COLLECTION_NAME,
) -> List[Dict[str, Any]]:
    """
    Searches Qdrant for transcript chunks relevant to the user query.
    Applies optional video-level scope filter.
    """
    ensure_collection(collection_name=collection_name, vector_dim=EMBED_DIM)
    client = get_qdrant_client()
    from qdrant_client.http import models

    query_vec = embed_query(query)

    query_filter: Optional[models.Filter] = None
    if video_id:
        query_filter = models.Filter(
            must=[
                models.FieldCondition(
                    key="video_id",
                    match=models.MatchValue(value=video_id),
                )
            ]
        )

    # Qdrant client 1.9+ / 1.20+ uses query_points
    try:
        response = client.query_points(
            collection_name=collection_name,
            query=query_vec,
            query_filter=query_filter,
            limit=top_k,
            score_threshold=relevance_threshold,
        )
        results = response.points
    except AttributeError:
        # Fallback for older client versions
        results = client.search(
            collection_name=collection_name,
            query_vector=query_vec,
            query_filter=query_filter,
            limit=top_k,
            score_threshold=relevance_threshold,
        )

    retrieved: List[Dict[str, Any]] = []
    seen_chunk_ids = set()

    for r in results:
        payload = r.payload or {}
        chunk_id = payload.get("chunk_id", "")
        if chunk_id in seen_chunk_ids:
            continue
        seen_chunk_ids.add(chunk_id)

        retrieved.append(
            {
                "score": round(float(r.score), 4),
                "chunk_id": chunk_id,
                "video_id": payload.get("video_id", ""),
                "title": payload.get("title", ""),
                "video_url": payload.get("video_url", ""),
                "chunk_index": payload.get("chunk_index", 0),
                "start": payload.get("start", 0.0),
                "end": payload.get("end", 0.0),
                "text": payload.get("text", ""),
                "language": payload.get("language", ""),
            }
        )

    return retrieved
