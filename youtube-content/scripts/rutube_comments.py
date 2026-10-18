#!/usr/bin/env python3
"""Rutube comment mood + thread homogeneity — same lexicon as comment_sentiment.py.

Usage: python rutube_comments.py URL [--limit N] [--json]

yt-dlp does NOT extract Rutube comments; the public API does:
  https://rutube.ru/api/comments/video/<32-hex-id>/?limit=100  (paginate via "next")
Fields: text, user.name, parent_id (None = root), created_ts.
"""
import argparse, json, os, re, sys, urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from comment_sentiment import classify  # shared lexicon — keep both skills in sync

UA = {"User-Agent": "Mozilla/5.0"}


def video_id(url):
    m = re.search(r"rutube\.ru/(?:video|embed)/([0-9a-f]{32})", url)
    return m.group(1) if m else None


def get_json(url):
    req = urllib.request.Request(url, headers=UA)
    return json.loads(urllib.request.urlopen(req, timeout=60).read())


def fetch_all(vid, limit=0):
    rows, url = [], f"https://rutube.ru/api/comments/video/{vid}/?limit=100"
    while url:
        d = get_json(url)
        for c in d.get("results", []):
            rows.append({"id": c.get("id"), "parent": c.get("parent_id") or "root",
                         "author": (c.get("user") or {}).get("name") or "?",
                         "text": (c.get("text") or "").strip()})
            if limit and len(rows) >= limit:
                return rows
        url = d.get("next")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("url")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    vid = video_id(args.url)
    if not vid:
        print("Not a Rutube video URL.", file=sys.stderr)
        sys.exit(1)
    rows = fetch_all(vid, args.limit)
    if not rows:
        print("No comments available (disabled or fetch failed).", file=sys.stderr)
        sys.exit(2)

    for row in rows:
        row["cat"], row["shouty"] = classify(row["text"])

    n = len(rows)
    counts = {}
    for row in rows:
        counts[row["cat"]] = counts.get(row["cat"], 0) + 1
    mood = {k: round(100 * v / n) for k, v in counts.items()}

    by_id = {row["id"]: row for row in rows}
    threads = {}
    for row in rows:
        if row["parent"] != "root":
            threads.setdefault(row["parent"], []).append(row)
    for pid, replies in threads.items():
        if pid in by_id:
            replies.insert(0, by_id[pid])

    thread_lines, hom_sum, hom_n = [], 0, 0
    for pid, members in threads.items():
        if len(members) < 2:
            continue
        stance = [m for m in members if m["cat"] != "нейтрально"]
        p = by_id.get(pid) or members[0]
        if len(stance) < 2:
            thread_lines.append({"thread": p["text"][:60], "author": p["author"],
                                 "size": len(members), "dominant": None, "homogeneity": None})
            continue
        ccount = {}
        for m in stance:
            ccount[m["cat"]] = ccount.get(m["cat"], 0) + 1
        dom = max(ccount, key=ccount.get)
        hom = round(100 * ccount[dom] / len(stance))
        hom_sum += hom * len(stance); hom_n += len(stance)
        thread_lines.append({"thread": p["text"][:60], "author": p["author"],
                             "size": len(members), "dominant": dom, "homogeneity": hom})
    overall_hom = round(hom_sum / hom_n) if hom_n else None

    if args.json:
        print(json.dumps({"url": args.url, "comments": n, "mood": mood,
                          "homogeneity_overall": overall_hom, "threads": thread_lines},
                         ensure_ascii=False))
    else:
        print(f"=== Настроение комментариев ({n}) ===")
        for cat in ["злость", "радость", "недовольство", "нейтрально"]:
            if cat in mood:
                print(f"{cat} {mood[cat]}%")
        shouty = sum(1 for row in rows if row["shouty"])
        print(f"эмоционально окрашенных (без нейтральных): "
              f"{round(100 * (n - mood.get('нейтрально', 0)) / n)}%, из них «кричащих» {shouty}")
        print("\n=== Однородность ===")
        print(f"в среднем по тредам: {overall_hom if overall_hom is not None else 'не измеряется'}")
        print("по тредам (≥2 комментариев):")
        for t in thread_lines:
            if t["homogeneity"] is None:
                print(f"  @{t['author']} «{t['thread']}…» — не измеряется (нейтральный тред, {t['size']})")
            else:
                print(f"  @{t['author']} «{t['thread']}…» — однородность в треде "
                      f"{t['homogeneity']}% ({t['size']}, доминирует {t['dominant']})")


if __name__ == "__main__":
    main()
