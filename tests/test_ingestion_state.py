"""
Unit tests for SQLite ingestion job state and stage transitions.
"""

from app.storage import (
    delete_video,
    get_video,
    init_db,
    list_videos,
    mark_completed,
    mark_failed,
    register_video,
    update_stage,
)


def test_sqlite_state_workflow():
    init_db()
    test_id = "test_vid_999"

    # Cleanup if present
    delete_video(test_id)

    # 1. Register video
    vid = register_video(
        video_id=test_id,
        title="Sample Lecture on Neural Networks",
        video_url=f"https://www.youtube.com/watch?v={test_id}",
        duration_seconds=1800.0,
    )
    assert vid["video_id"] == test_id
    assert vid["status"] == "queued"
    assert vid["current_stage"] == "registered"

    # 2. Stage transitions
    update_stage(test_id, stage="transcribing", status="in_progress")
    vid = get_video(test_id)
    assert vid["current_stage"] == "transcribing"
    assert vid["status"] == "in_progress"

    update_stage(test_id, stage="chunking", status="in_progress")
    vid = get_video(test_id)
    assert vid["current_stage"] == "chunking"

    # 3. Mark completed
    mark_completed(test_id)
    vid = get_video(test_id)
    assert vid["status"] == "completed"
    assert vid["current_stage"] == "completed"

    # 4. List videos includes this video
    all_vids = list_videos()
    assert any(v["video_id"] == test_id for v in all_vids)

    # 5. Clean up
    delete_video(test_id)
    assert get_video(test_id) is None

