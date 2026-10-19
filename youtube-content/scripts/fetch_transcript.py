#!/usr/bin/env python3
"""
Fetch a YouTube video transcript and output it as structured JSON.

Usage:
    python fetch_transcript.py <url_or_video_id> [--language en,tr] [--timestamps]

Output (JSON):
    {
        "video_id": "...",
        "language": "en",
        "segments": [{"text": "...", "start": 0.0, "duration": 2.5}, ...],
        "full_text": "complete transcript as plain text",
        "timestamped_text": "00:00 first line\n00:05 second line\n..."
    }

Use the isolated helper interpreter prepared by this skill's PM setup recipe.
"""

import argparse
import json
import re
import sys


def extract_video_id(url_or_id: str) -> str:
    """Extract the 11-character video ID from various YouTube URL formats."""
    url_or_id = url_or_id.strip()
    patterns = [
        r'(?:v=|youtu\.be/|shorts/|embed/|live/)([a-zA-Z0-9_-]{11})',
        r'^([a-zA-Z0-9_-]{11})$',
    ]
    for pattern in patterns:
        match = re.search(pattern, url_or_id)
        if match:
            return match.group(1)
    return url_or_id


def format_timestamp(seconds: float) -> str:
    """Формат — MM:SS, как закреплено правилом пользователя (`EMOJI [MM:SS–MM:SS]`),
    и он обязан оставаться тем же на видео длиннее часа: минуты продолжают счёт
    (61:40), а не переходят в HH:MM:SS. Прежняя версия выдавала `1:01:40` — тот же
    фрагмент доставлялся в другом формате, и таймлайн переставал соответствовать правилу.
    Ссылка на момент выводится из секунд, поэтому от выбора формата не зависит."""
    total = int(seconds)
    m, s = divmod(total, 60)
    return f"{m}:{s:02d}"


def fetch_transcript(video_id: str, languages: list | None = None):
    """Fetch transcript segments from YouTube.

    Returns (segments, language, is_generated) — язык и признак автосубтитров нужны,
    потому что docstring обещает поле language в выводе, и потому что авто-трек
    содержит ошибки распознавания имён и брендов: без признака «generated» читатель не
    знает, какой это транскрипт. v1.x отдаёт язык на объекте FetchedTranscript.

    Segments are dicts with 'text', 'start', and 'duration' keys.
    """
    try:
        from youtube_transcript_api import YouTubeTranscriptApi
    except ImportError:
        print("Error: youtube-transcript-api not installed. Use the isolated PM helper environment described in youtube-content/SKILL.md.",
              file=sys.stderr)
        sys.exit(1)

    api = YouTubeTranscriptApi()
    if languages:
        result = api.fetch(video_id, languages=languages)
    else:
        result = api.fetch(video_id)

    # v1.x returns FetchedTranscriptSnippet objects; normalize to dicts
    segments = [
        {"text": seg.text, "start": seg.start, "duration": seg.duration}
        for seg in result
    ]
    return segments, getattr(result, "language", None), bool(getattr(result, "is_generated", False))


def build_result(video_id, segments, language, generated):
    """Формирование результата — отдельная функция, тестируемая offline на mock-сегментах."""
    full_text = " ".join(seg["text"] for seg in segments)
    timestamped = "\n".join(
        f"{format_timestamp(seg['start'])} {seg['text']}" for seg in segments
    )
    return {
        "video_id": video_id,
        "language": language,                 # обещано docstring
        "generated": generated,               # авто-трек: ошибки распознавания имён и брендов
        "segment_count": len(segments),
        "duration": format_timestamp(segments[-1]["start"] + segments[-1]["duration"]) if segments else "0:00",
        "full_text": full_text,
        "timestamped_text": timestamped,
    }


def main():
    parser = argparse.ArgumentParser(description="Fetch YouTube transcript as JSON")
    parser.add_argument("url", help="YouTube URL or video ID")
    parser.add_argument("--language", "-l", default=None,
                        help="Comma-separated language codes (e.g. en,tr). Default: auto")
    parser.add_argument("--timestamps", "-t", action="store_true",
                        help="Include timestamped text in output")
    parser.add_argument("--text-only", action="store_true",
                        help="Output plain text instead of JSON")
    parser.add_argument("--dry-run", action="store_true",
                        help="печатает запрос, который отправил бы, и не обращается к сети")
    parser.add_argument("--mock-file", help="JSON сегментов для offline-проверки (вместо сети)")
    args = parser.parse_args()

    video_id = extract_video_id(args.url)
    languages = [l.strip() for l in args.language.split(",")] if args.language else None

    if args.dry_run:
        print(f"dry-run: запрос транскрипта video_id={video_id} languages={languages or 'auto'}")
        if args.mock_file:
            with open(args.mock_file, encoding="utf-8") as f:
                segs = json.load(f)
            print(json.dumps(build_result(video_id, segs, (languages or ["auto"])[0], True),
                             ensure_ascii=False, indent=2))
        return

    try:
        segments, language, generated = fetch_transcript(video_id, languages)
    except Exception as e:
        error_msg = str(e)
        if "disabled" in error_msg.lower():
            print(json.dumps({"error": "Transcripts are disabled for this video."}))
        elif "no transcript" in error_msg.lower():
            print(json.dumps({"error": "No transcript found. Try specifying a language with --language."}))
        else:
            print(json.dumps({"error": error_msg}))
        sys.exit(1)

    result = build_result(video_id, segments, language, generated)
    if args.text_only:
        print(result["timestamped_text"] if args.timestamps else result["full_text"])
        return
    if not args.timestamps:
        result.pop("timestamped_text", None)

    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
