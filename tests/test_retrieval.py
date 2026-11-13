"""
Unit tests for deterministic point IDs and grounded teaching prompts.
"""

from app.retrieval import generate_point_id
from app.teaching import build_teaching_prompts


def test_deterministic_point_id():
    # Calling generate_point_id multiple times with same inputs yields identical UUID
    id1 = generate_point_id("video_abc", 1, version="1.0.0")
    id2 = generate_point_id("video_abc", 1, version="1.0.0")
    id3 = generate_point_id("video_abc", 2, version="1.0.0")
    id_other_version = generate_point_id("video_abc", 1, version="2.0.0")

    assert id1 == id2
    assert id1 != id3
    assert id1 != id_other_version


def test_build_grounded_prompts():
    question = "What is Backpropagation in Hinglish?"
    chunks = [
        {
            "chunk_id": "test_chunk_001",
            "video_id": "vid123",
            "title": "Deep Learning Basics",
            "start": 120.0,
            "end": 195.0,
            "text": "Backpropagation algorithm gradients compute karta hai using chain rule.",
        }
    ]

    sys_prompt, usr_prompt = build_teaching_prompts(
        question, chunks, language_preference="Hinglish"
    )

    assert "patient, encouraging AI teaching assistant" in sys_prompt
    assert "Strict Grounding Rules" in sys_prompt
    assert "Hinglish" in sys_prompt
    assert "Deep Learning Basics" in usr_prompt
    assert "02:00 - 03:15" in usr_prompt
    assert "chain rule" in usr_prompt
    assert question in usr_prompt

