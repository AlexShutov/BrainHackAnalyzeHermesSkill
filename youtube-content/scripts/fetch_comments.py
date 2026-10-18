#!/usr/bin/env python3
"""Fetch YouTube comments without an API key (innertube via yt-dlp).

Usage:
  python fetch_comments.py URL [--limit N] [--json]

Requires yt-dlp (install into the isolated target dir, see SKILL.md Setup).
Prints one JSON object per comment: {author, text, likes, timestamp, pinned, parent}.
"""
import argparse, json, subprocess, sys

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("url")
    ap.add_argument("--limit", type=int, default=0, help="max comments (0 = all)")
    ap.add_argument("--json", action="store_true", help="single JSON blob instead of JSONL")
    args = ap.parse_args()

    cmd = [sys.executable, "-m", "yt_dlp", "--skip-download", "--write-comments",
           "-j", args.url]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if r.returncode != 0:
        print("yt-dlp failed:", r.stderr.strip()[:500], file=sys.stderr)
        sys.exit(1)
    info = json.loads(r.stdout)
    comments = info.get("comments")
    if comments is None:
        print("No comments available (disabled on this video or fetch failed).", file=sys.stderr)
        sys.exit(2)
    out = []
    for c in comments:
        out.append({
            "author": c.get("author"),
            "text": (c.get("text") or "").strip(),
            "likes": c.get("like_count") or 0,
            "timestamp": c.get("timestamp"),          # unix epoch, None if unknown
            "pinned": bool(c.get("is_pinned")),
            "parent": c.get("parent"),                 # comment id or None (top-level)
        })
        if args.limit and len(out) >= args.limit:
            break
    if args.json:
        print(json.dumps({"title": info.get("title"), "url": info.get("webpage_url"),
                          "count": len(out), "comments": out}, ensure_ascii=False))
    else:
        for c in out:
            print(json.dumps(c, ensure_ascii=False))

if __name__ == "__main__":
    main()
