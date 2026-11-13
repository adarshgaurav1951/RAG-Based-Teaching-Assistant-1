"""
Unit tests for 75-second timestamp-aware chunking algorithm.
"""

from app.retrieval import chunk_transcript


def test_chunking_target_duration():
    # Create 20 segments of ~10 seconds each (total ~200 seconds)
    # Expected: 200s / 75s = ~3 chunks
    segments = []
    for i in range(20):
        start = float(i * 10)
        end = float((i + 1) * 10)
        segments.append(
            {
                "segment_id": i + 1,
                "start": start,
                "end": end,
                "text": f"Segment {i + 1} text explaining topic.",
            }
        )

    transcript = {
        "video_id": "test_video_123",
        "title": "Test Lecture",
        "detected_language": "hi",
        "segments": segments,
    }

    chunks = chunk_transcript(transcript, target_seconds=75.0)

    assert len(chunks) == 3
    # Check first chunk
    assert chunks[0]["start"] == 0.0
    assert chunks[0]["end"] == 70.0  # 7 segments of 10s = 70s (next would exceed 75s)
    assert chunks[0]["chunk_index"] == 1
    assert chunks[0]["chunk_id"] == "test_video_123_0001"

    # Check second chunk
    assert chunks[1]["start"] == 70.0
    assert chunks[1]["chunk_index"] == 2

    # Check third chunk
    assert chunks[2]["end"] == 200.0
    assert chunks[2]["chunk_index"] == 3

    # Verify all segment texts are preserved
    full_chunk_text = " ".join(c["text"] for c in chunks)
    for seg in segments:
        assert seg["text"] in full_chunk_text


def test_chunking_single_long_segment():
    segments = [
        {"segment_id": 1, "start": 0.0, "end": 120.0, "text": "Unusually long segment."}
    ]
    transcript = {
        "video_id": "long_seg_vid",
        "title": "Long Segment Video",
        "segments": segments,
    }
    chunks = chunk_transcript(transcript, target_seconds=75.0)
    assert len(chunks) == 1
    assert chunks[0]["start"] == 0.0
    assert chunks[0]["end"] == 120.0
    assert chunks[0]["text"] == "Unusually long segment."


def test_chunking_empty_transcript():
    transcript = {"video_id": "empty", "title": "Empty", "segments": []}
    chunks = chunk_transcript(transcript, target_seconds=75.0)
    assert chunks == []
