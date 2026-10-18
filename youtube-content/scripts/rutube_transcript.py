#!/usr/bin/env python3
"""Rutube transcript: fetch auto/manual subtitles via yt-dlp and print timestamped lines.

Usage: python rutube_transcript.py URL [--language ru]
Output format matches fetch_transcript.py --text-only --timestamps: `M:SS >> text`.

Rutube has no transcript API; yt-dlp's `rutube` extractor returns subtitles only when
the video has an auto-generated or manual caption track. Many Rutube videos have none —
then say so plainly, never fabricate a transcript.
"""
import argparse, os, re, subprocess, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "scripts"))
from util import read_text

UA = {"User-Agent": "Mozilla/5.0"}


def video_id(url):
    m = re.search(r"rutube\.ru/(?:video|embed)/([0-9a-f]{32})", url)
    return m.group(1) if m else None


def srt_time(t):
    # 00:01:23,456 -> 1:23 (drop ms; keep minutes unpadded like the YouTube helper)
    h, m, s = t.split(":")
    s = int(s.split(",")[0])
    m = int(m); h = int(h)
    if h:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m}:{s:02d}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("url")
    ap.add_argument("--language", default="ru")
    ap.add_argument("--out-dir", default=os.environ.get("TMPDIR", "."))
    args = ap.parse_args()

    vid = video_id(args.url)
    if not vid:
        print("Not a Rutube video URL.", file=sys.stderr)
        sys.exit(1)

    r = subprocess.run([sys.executable, "-m", "yt_dlp", "--skip-download", "--write-sub",
                        "--sub-langs", args.language, "--sub-format", "srt",
                        "-o", os.path.join(args.out_dir, "rt_%(id)s"), args.url],
                       capture_output=True, text=True, timeout=600)
    log = r.stdout + r.stderr
    if "has no subtitles" in log or "No subtitles available" in log:
        print("No subtitles on this Rutube video — cannot analyze text content.",
              file=sys.stderr)
        sys.exit(2)

    path = os.path.join(args.out_dir, f"rt_{vid}.{args.language}.srt")
    if not os.path.exists(path):
        print("Subtitle file not written:", log.strip()[-400:], file=sys.stderr)
        sys.exit(1)

    out = []
    for block in read_text(path).split("\n\n"):
        lines = [l for l in block.splitlines() if l.strip()]
        if len(lines) < 2 or "-->" not in lines[1]:
            continue
        out.append(f"{srt_time(lines[1].split(' --> ')[0])} >> {' '.join(lines[2:])}")
    print("\n".join(out))


if __name__ == "__main__":
    main()
