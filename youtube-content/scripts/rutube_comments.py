#!/usr/bin/env python3
"""Rutube comment mood + thread homogeneity — same lexicon as comment_sentiment.py.

Usage: python rutube_comments.py URL [--limit N] [--json]

yt-dlp does NOT extract Rutube comments; the public API does:
  https://rutube.ru/api/comments/video/<32-hex-id>/?limit=100  (paginate via "next")
Fields: text, user.name, parent_id (None = root), created_ts.
"""
import argparse, json, os, re, sys, urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from comment_sentiment import classify, build_threads, thread_homogeneity  # общий лексикон И общее определение треда — держать оба навыка в синхроне

UA = {"User-Agent": "Mozilla/5.0"}


def video_id(url):
    m = re.search(r"rutube\.ru/(?:video|embed)/([0-9a-f]{32})", url)
    return m.group(1) if m else None


def get_json(url, transport=None):
    """transport — инъекция: тест подменяет сетевой вызов mock-transportом.
    Без него — прежний путь ( urllib )."""
    if transport is not None:
        return transport(url)
    req = urllib.request.Request(url, headers=UA)
    # контекстный менеджер: пагинация длинных тредов ходит по страницам API,
    # и urlopen без with оставлял socket-дескриптор открытым до GC
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read())


def canonical_rows(payload):
    """Каноническая форма комментария (id, parent, author, text) — одна функция,
    которую может проверить тест без сети: она принимает уже полученный JSON."""
    rows = []
    for c in payload.get("results", []):
        rows.append({"id": c.get("id"), "parent": c.get("parent_id") or "root",
                     "author": (c.get("user") or {}).get("name") or "?",
                     "text": (c.get("text") or "").strip()})
    return rows


def fetch_all(vid, limit=0, transport=None, dry_run=False, log=None):
    rows, url = [], f"https://rutube.ru/api/comments/video/{vid}/?limit=100"
    while url:
        if dry_run:
            # offline-режим: печатает, что отправил бы, и ничего не запрашивает
            print(f"dry-run: GET {url}")
            break
        d = get_json(url, transport)
        rows.extend(canonical_rows(d))
        if limit and len(rows) >= limit:
            return rows
        url = d.get("next")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("url")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--dry-run", action="store_true",
                    help="печатает запрос, который отправил бы, и не обращается к сети")
    ap.add_argument("--mock-file", help="JSON-ответ API для offline-проверки (вместо сети)")
    args = ap.parse_args()

    vid = video_id(args.url)
    if not vid:
        print("Not a Rutube video URL.", file=sys.stderr)
        sys.exit(1)
    if args.dry_run:
        print(f"dry-run: GET https://rutube.ru/api/comments/video/{vid}/?limit=100")
        if args.mock_file:
            with open(args.mock_file, encoding="utf-8") as f:
                payload = json.load(f)
            print("dry-run: каноническая форма из mock-ответа:")
            print(json.dumps(canonical_rows(payload), ensure_ascii=False, indent=2))
        return
    rows = fetch_all(vid, args.limit)
    if not rows:
        print("No comments available (disabled or fetch failed).", file=sys.stderr)
        sys.exit(2)

    for row in rows:
        row["cat"], row["shouty"] = classify(row["text"])

    # Единый входной модуль анализа комментариев (`comment_analysis.py`): тот же
    # нейро-слой, что и на YouTube-пути, лексикон — кросс-проверка. Определение треда
    # и «среднего по тредам» общие (build_threads / thread_homogeneity); раньше Rutube
    # считал только лексиконом, и доставляемые числа между путями не сравнивались.
    import comment_analysis as CA
    res = CA.analyze(rows)
    n = len(rows)
    mood = res["mood"]
    thread_lines, overall_hom, weighted_hom = res["threads"], res["overall"], res["weighted"]

    if args.json:
        print(json.dumps({"url": args.url, "comments": n, "layer": res["layer"],
                          "mood": mood, "cross_check": res["cross_check"],
                          "homogeneity_overall": overall_hom,
                          "homogeneity_weighted": weighted_hom,
                          "threads": thread_lines},
                         ensure_ascii=False))
    else:
        print(f"=== Настроение комментариев ({n}) ===")
        for cat in ["злость", "радость", "недовольство", "нейтрально"]:
            if cat in mood:
                print(f"{cat} {mood[cat]}%")
        shouty = sum(1 for row in rows if row["shouty"])
        print(f"эмоционально окрашенных (без нейтральных): "
              f"{100 - mood.get('нейтрально', 0)}%, из них «кричащих» {shouty}")
        print("\n=== Однородность ===")
        print(f"в среднем по тредам: {overall_hom}%" if overall_hom is not None else "тредов с ответами нет")
        if weighted_hom is not None and weighted_hom != overall_hom:
            print(f"взвешенная по размеру тредов: {weighted_hom}%")
        print("по тредам (≥2 комментариев):")
        for t in thread_lines:
            if t["homogeneity"] is None:
                print(f"  @{t['author']} «{t['thread']}…» — не измеряется (нейтральный тред, {t['size']})")
            else:
                print(f"  @{t['author']} «{t['thread']}…» — однородность в треде "
                      f"{t['homogeneity']}% ({t['size']}, доминирует {t['dominant']})")


if __name__ == "__main__":
    main()
