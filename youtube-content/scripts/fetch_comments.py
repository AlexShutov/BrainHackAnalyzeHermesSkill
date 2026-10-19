#!/usr/bin/env python3
"""Fetch YouTube comments without an API key (innertube via yt-dlp).

Usage:
  python fetch_comments.py URL [--limit N] [--json]

Requires yt-dlp (install into the isolated target dir, see SKILL.md Setup).
Prints one JSON object per comment: {id, parent, author, text, likes, timestamp, pinned}.

Кэш: полученный по video id набор лежит в scratch, повторный анализ того же видео его
не перекачивает. Ограничение инструмента (проверено на --help этой версии yt-dlp):
ограничения качания комментариев нет (--max-comments отсутствует), поэтому --limit
обрезает только вывод, а полнота набора проверяется размером кэша, а не флагом —
yt-dlp не сообщает, остановился ли он сам.
"""
import argparse, json, os, re, subprocess, sys

# Scratch — каталог Hermes, а не системный temp: в обычном процессе TMPDIR указывает на
# /var/folders/... и не совпадает с ~/.hermes/cache/scratch, из-за чего кэш и fallback-
# каталог yt-dlp «не существуют» и тест молча пропускается вместо проверки.
HERMES_SCRATCH = os.path.expanduser("~/.hermes/cache/scratch")
SCRATCH = HERMES_SCRATCH if os.path.isdir(HERMES_SCRATCH) else (os.environ.get("TMPDIR") or HERMES_SCRATCH)
CACHE_DIR = os.path.join(SCRATCH, "comments")
BASE_CMD = ["--skip-download", "--write-comments", "-j"]
# yt-dlp ставится в изолированный target dir, а не в venv навыка: без PYTHONPATH
# subprocess его не видит (замер: find_spec('yt_dlp') -> None, ребёнок падает с
# ModuleNotFoundError). Скрипт обязан сам добавить fallback-каталог в env ребёнка.
FALLBACK_PYLIB = os.path.join(SCRATCH, "yt-pylib")


def child_env():
    env = dict(os.environ)
    try:
        import importlib.util
        if importlib.util.find_spec("yt_dlp") is None and os.path.isdir(FALLBACK_PYLIB):
            env["PYTHONPATH"] = FALLBACK_PYLIB + os.pathsep + env.get("PYTHONPATH", "")
    except Exception:
        pass
    return env


def video_id(url):
    m = re.search(r"(?:v=|youtu\.be/|shorts/|embed/)([A-Za-z0-9_-]{11})", url or "")
    return m.group(1) if m else None


def build_cmd(url):
    return [sys.executable, "-m", "yt_dlp"] + BASE_CMD + [url]


def run_ytdlp(url):
    r = subprocess.run(build_cmd(url), capture_output=True, text=True,
                       timeout=600, env=child_env())
    if r.returncode != 0:
        raise SystemExit(f"yt-dlp failed: {r.stderr.strip()[:500]}")
    return json.loads(r.stdout)


def normalize(info):
    comments = info.get("comments")
    if comments is None:
        raise SystemExit("No comments available (disabled on this video or fetch failed).")
    out = []
    for c in comments:
        out.append({
            # каноническая форма комментария обязана несть id: без него build_threads
            # не строит тред по цепочке, а угадывает корень по префиксу parent
            "id": c.get("id"),
            "parent": c.get("parent"),                 # comment id or None (top-level)
            "author": c.get("author"),
            "text": (c.get("text") or "").strip(),
            "likes": c.get("like_count") or 0,
            "timestamp": c.get("timestamp"),          # unix epoch, None if unknown
            "pinned": bool(c.get("is_pinned")),
        })
    return out


def fetch(url, limit=0, runner=None, cache_dir=None):
    """Возвращает (meta, comments, cache_hit). limit=0 — все.
    Кэш покрывает запрос только если в нём достаточно комментариев; иначе перекачиваем —
    молча выдавать partial за полный набор нельзя (полнота проверяется размером, флага нет).
    """
    cache_dir = cache_dir or CACHE_DIR
    vid = video_id(url)
    path = os.path.join(cache_dir, f"{vid}.json") if vid else None
    if path and os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        cached = data.get("comments") or []
        if not limit or len(cached) >= limit:
            return data, (cached[:limit] if limit else cached), True
    info = (runner or run_ytdlp)(url)
    comments = normalize(info)
    meta = {"video_id": vid, "title": info.get("title"),
            "url": info.get("webpage_url") or url, "fetched": len(comments)}
    if path:
        os.makedirs(cache_dir, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(dict(meta, comments=comments), f, ensure_ascii=False)
    return meta, (comments[:limit] if limit else comments), False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("url")
    ap.add_argument("--limit", type=int, default=0, help="max comments (0 = all)")
    ap.add_argument("--json", action="store_true", help="single JSON blob instead of JSONL")
    ap.add_argument("--no-cache", action="store_true", help="не читать и не писать кэш")
    ap.add_argument("--cache-dir", help="куда писать кэш (default: scratch/comments)")
    args = ap.parse_args()
    if args.limit < 0:
        sys.exit("--limit не может быть отрицательным")

    if args.no_cache:
        meta = {"video_id": video_id(args.url), "url": args.url}
        comments = normalize(run_ytdlp(args.url))
        hit = False
    else:
        meta, comments, hit = fetch(args.url, args.limit, cache_dir=args.cache_dir)
    if hit:
        print(f"cache hit: {meta.get('video_id')} ({len(comments)} из кэша)", file=sys.stderr)

    if args.json:
        print(json.dumps({"title": meta.get("title"), "url": meta.get("url"),
                          "video_id": meta.get("video_id"), "count": len(comments),
                          "cached": hit, "comments": comments}, ensure_ascii=False))
    else:
        for c in comments:
            print(json.dumps(c, ensure_ascii=False))


if __name__ == "__main__":
    main()
