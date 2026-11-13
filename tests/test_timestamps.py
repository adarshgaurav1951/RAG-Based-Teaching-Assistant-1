"""
Unit tests for timestamp formatting, citation links, and YouTube ID extraction.
"""

from app.media import extract_video_id
from app.teaching import format_citation_url, format_timestamp


def test_format_timestamp():
    assert format_timestamp(0) == "00:00"
    assert format_timestamp(45.5) == "00:45"
    assert format_timestamp(75.0) == "01:15"
    assert format_timestamp(599) == "09:59"
    assert format_timestamp(3014) == "50:14"
    assert format_timestamp(3665) == "01:01:05"


def test_format_citation_url():
    url = format_citation_url("pSVk-5WemQ0", 730.8)
    assert url == "https://www.youtube.com/watch?v=pSVk-5WemQ0&t=730s"

    url_zero = format_citation_url("abc123xyz89", 0.0)
    assert url_zero == "https://www.youtube.com/watch?v=abc123xyz89&t=0s"


def test_extract_video_id():
    assert extract_video_id("pSVk-5WemQ0") == "pSVk-5WemQ0"
    assert extract_video_id("https://youtu.be/pSVk-5WemQ0") == "pSVk-5WemQ0"
    assert (
        extract_video_id("https://www.youtube.com/watch?v=pSVk-5WemQ0&feature=shared")
        == "pSVk-5WemQ0"
    )
    assert (
        extract_video_id("https://www.youtube.com/embed/pSVk-5WemQ0")
        == "pSVk-5WemQ0"
    )
    assert extract_video_id("invalid-link-without-id") is None

