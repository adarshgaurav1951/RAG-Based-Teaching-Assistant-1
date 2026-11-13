"""
Command-line interface (CLI) for RAG-Based AI Teaching Assistant.
Enables inspection, single-video ingestion, library listing, status retry,
and conversational Q&A with clickable YouTube timestamp links.
"""

from __future__ import annotations

import argparse
import sys
from typing import Optional

from app.ingest import ingest_video
from app.media import inspect_url
from app.storage import get_video, list_videos
from app.teaching import format_timestamp, generate_answer


def cmd_inspect(url: str) -> None:
    """Inspects a YouTube video or playlist without downloading media."""
    print(f"\n🔍 Inspecting YouTube resource: {url}...")
    try:
        info = inspect_url(url)
    except Exception as e:
        print(f"❌ Error inspecting URL: {e}", file=sys.stderr)
        sys.exit(1)

    if info.get("is_playlist"):
        print(f"\n📋 Playlist Found: {info.get('playlist_title')} (ID: {info.get('playlist_id')})")
        print(f"   Total Videos: {info.get('video_count')}")
        print("\n   Select ONE video to ingest using `python -m app.cli ingest <url>`:\n")
        for v in info.get("videos", [])[:20]:
            dur_fmt = format_timestamp(v.get("duration_seconds", 0))
            print(f"   [{v['index']}] {v['title']}")
            print(f"       ID: {v['video_id']} | Duration: {dur_fmt} | Link: {v['video_url']}")
        if info.get("video_count", 0) > 20:
            print(f"   ... and {info['video_count'] - 20} more videos.")
    else:
        dur_fmt = format_timestamp(info.get("duration_seconds", 0))
        print(f"\n🎥 Single Video Found: {info.get('title')}")
        print(f"   ID:       {info.get('video_id')}")
        print(f"   Duration: {dur_fmt} ({info.get('duration_seconds')}s)")
        print(f"   URL:      {info.get('video_url')}")
        print("\nReady to ingest! Run:")
        print(f"   python -m app.cli ingest {info.get('video_url')}\n")


def cmd_ingest(url_or_id: str, force: bool = False) -> None:
    """Runs end-to-end ingestion for exactly one selected video."""
    print(f"\n🚀 Initiating ingestion for: {url_or_id}")
    print("   Pipeline: yt-dlp -> FFmpeg mono 16kHz -> faster-whisper CPU INT8 -> 75s chunks -> MiniLM -> Qdrant\n")

    try:
        res = ingest_video(url_or_id, force_reprocess=force)
    except Exception as e:
        print(f"\n❌ Ingestion failed: {e}", file=sys.stderr)
        sys.exit(1)

    if res.get("status") == "already_indexed":
        print(f"ℹ️ {res.get('message')}")
        return

    print("✅ Ingestion successfully completed!")
    print(f"   Title:           {res.get('title')}")
    print(f"   Video ID:        {res.get('video_id')}")
    print(f"   Detected Lang:   {res.get('detected_language')}")
    print(f"   ASR Segments:    {res.get('segment_count')}")
    print(f"   75s Chunks:      {res.get('chunk_count')}")
    print(f"   Qdrant Vectors:  {res.get('points_indexed')}")
    print("\n🎉 You can now ask questions! Example:")
    print(f'   python -m app.cli ask "What is the main topic of this lecture?" --video-id {res.get("video_id")}\n')


def cmd_list() -> None:
    """Lists all registered videos and their ingestion states."""
    videos = list_videos()
    if not videos:
        print("\n📭 No videos have been registered or indexed yet.")
        print("To index a video, run: python -m app.cli ingest <youtube_url>\n")
        return

    print(f"\n📚 Indexed Lectures ({len(videos)} total):\n")
    for v in videos:
        dur = format_timestamp(v.get("duration_seconds", 0))
        status_icon = "✅" if v["status"] == "completed" else ("❌" if v["status"] == "failed" else "⏳")
        print(f"{status_icon} [{v['video_id']}] {v['title']}")
        print(f"   Status: {v['status']} (Stage: {v['current_stage']}) | Duration: {dur}")
        if v.get("error_message"):
            print(f"   Error: {v['error_message']}")
        print()


def cmd_ask(
    question: str,
    video_id: Optional[str] = None,
    top_k: int = 5,
    lang: Optional[str] = None,
) -> None:
    """Queries the indexed knowledge base and displays the teaching answer."""
    print(f"\n❓ Question: {question}")
    if video_id:
        print(f"   Scoped to Video: {video_id}")
    print("   Searching knowledge base...\n")

    res = generate_answer(
        question=question,
        video_id=video_id,
        top_k=top_k,
        language_preference=lang,
    )

    print("═══════════════════════════════════════════════════════════════")
    print("💡 Teaching Assistant Response:")
    print("═══════════════════════════════════════════════════════════════")
    print(res.get("answer", ""))
    print("═══════════════════════════════════════════════════════════════")

    citations = res.get("citations", [])
    if citations:
        print("\n📌 Verified Timestamp Citations:")
        for idx, c in enumerate(citations, start=1):
            print(f"   [{idx}] {c['title']}")
            print(f"       ⏱️ Time: {c['formatted_range']} ({int(c['start'])}s)")
            print(f"       🔗 Link: {c['url']}")
            print(f"       📝 Transcript Excerpt: \"{c['text_snippet']}\"")
            print()
    else:
        print("\n(No direct timestamp citations found for this query)")


def cmd_retry(video_id: str) -> None:
    """Retries a failed or incomplete ingestion job."""
    v = get_video(video_id)
    if not v:
        print(f"❌ Video ID '{video_id}' not found in database.", file=sys.stderr)
        sys.exit(1)

    print(f"🔄 Retrying ingestion for '{v['title']}' ({video_id})...")
    cmd_ingest(v["video_url"], force=True)


def main() -> None:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(
        prog="rag-assistant",
        description="CPU-First Hinglish YouTube RAG AI Teaching Assistant",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # inspect
    p_inspect = subparsers.add_parser("inspect", help="Inspect YouTube URL metadata without downloading")
    p_inspect.add_argument("url", help="YouTube video or playlist URL")

    # ingest
    p_ingest = subparsers.add_parser("ingest", help="Ingest exactly one selected video")
    p_ingest.add_argument("url_or_id", help="YouTube video URL or 11-char ID")
    p_ingest.add_argument("--force", action="store_true", help="Force re-transcription and re-indexing")

    # list
    subparsers.add_parser("list", help="List all known videos and their statuses")

    # ask
    p_ask = subparsers.add_parser("ask", help="Ask a question against indexed videos")
    p_ask.add_argument("question", help="Question in English, Hindi, or Hinglish")
    p_ask.add_argument("--video-id", help="Scope search to a specific video ID", default=None)
    p_ask.add_argument("--top-k", type=int, default=5, help="Number of retrieved chunks")
    p_ask.add_argument("--lang", help="Answer language preference (English/Hindi/Hinglish)", default=None)

    # retry
    p_retry = subparsers.add_parser("retry", help="Retry failed video ingestion")
    p_retry.add_argument("video_id", help="YouTube video ID")

    args = parser.parse_args()

    if args.command == "inspect":
        cmd_inspect(args.url)
    elif args.command == "ingest":
        cmd_ingest(args.url_or_id, force=args.force)
    elif args.command == "list":
        cmd_list()
    elif args.command == "ask":
        cmd_ask(args.question, video_id=args.video_id, top_k=args.top_k, lang=args.lang)
    elif args.command == "retry":
        cmd_retry(args.video_id)


if __name__ == "__main__":
    main()

