#!/usr/bin/env python3
"""Comment mood + thread homogeneity for a YouTube video (no API key).

Usage:
  python comment_sentiment.py URL [--limit N] [--json]

Fetches all comments via yt-dlp, classifies each comment into emotion
categories with a Russian lexicon, prints overall mood percentages and
per-thread homogeneity (share of comments in a thread matching the thread's
dominant category — a proxy for agreement inside the thread).

Categories: злость, радость, недовольство, нейтрально.
Homogeneity proxy caveat: emotional coloring, not literal agreement;
sarcasm/irony can be misclassified.
"""
import argparse, json, re, subprocess, sys

ANGER = [
    "злость", "злой", "злая", "злое", "злые", "бешен", "ярост", "гнев",
    "ненавист", "ненавиж", "ненавид", "мразь", "мерзост", "мерзк", "дебил",
    "идиот", "урод", "скот", "гадост", "гад ", "нахрен", "достал", "бесят",
    "бесит", "раздража", "агресс", "хамств", "наглост", "оскорб", "тупой",
    "тупые", "тупость", "дурной", "дурость", "деградант", "быдл", "бараны",
    "овцы", "враг", "враги", "бык", "быки", "шантаж", "наглость",
]
JOY = [
    "радост", "рад ", "рада", "рады", "спасибо", "благодар", "круто",
    "крутой", "классн", "класс", "отлично", "отличн", "здорово", "люблю",
    "любим", "счасть", "счастл", "молодец", "молодцы", "красота", "красиво",
    "супер", "нравится", "нравились", "приятно", "приятн", "успех",
    "успешн", "поздравляю", "ура", "восторг", "восхищ", "лучший", "лучшие",
    "прогресс", "огонь",
]
DISCONTENT = [
    "недоволь", "не нравится", "не нравятся", "не понравился", "плохо",
    "плохой", "плохие", "плохое", "хуже", "несправедлив", "разочаров",
    "не согласен", "не согласна", "не согласны", "фу ", "скучн", "слабо",
    "хренов", "ужасн", "кошмар", "проблем", "загибается", "не надо",
    "не стоит", "не умеет", "не умеют", "не может", "не могут",
    "не понимает", "не понимают", "не видят", "не слышат", "обман",
    "врут", "врали", "враньё", "инфоцыган", "развод", "надувательств",
    "неэффективн", "бесполезн", "бессмысленн", "не работает",
    "не получается", "не успе", "не готов", "не заточен", "не выполняет",
    "не верят", "не ценят", "не уважают", "игнорируют", "замалчива",
    "не надо", "не нужен", "не нужна", "не нужны",
]
CATS = {"злость": ANGER, "радость": JOY, "недовольство": DISCONTENT}

def classify(text):
    t = text.lower()
    scores = {}
    for cat, stems in CATS.items():
        s = sum(1 for w in stems if w in t)
        if s:
            scores[cat] = s
    if not scores:
        return "нейтрально", False
    best = max(scores, key=scores.get)
    # emotional intensity: shouty caps
    letters = [c for c in text if c.isalpha()]
    shouty = letters and sum(1 for c in letters if c.isupper()) / len(letters) > 0.4
    return best, shouty

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("url")
    ap.add_argument("--limit", type=int, default=0, help="max comments (0 = all)")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    cmd = [sys.executable, "-m", "yt_dlp", "--skip-download", "--write-comments",
           "-j", args.url]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=1500)
    if r.returncode != 0:
        print("yt-dlp failed:", r.stderr.strip()[:500], file=sys.stderr)
        sys.exit(1)
    info = json.loads(r.stdout)
    comments = info.get("comments")
    if comments is None:
        print("No comments available (disabled on this video or fetch failed).",
              file=sys.stderr)
        sys.exit(2)
    if args.limit:
        comments = comments[:args.limit]

    # classify
    rows = []
    for c in comments:
        cat, shouty = classify(c.get("text") or "")
        rows.append({"id": c.get("id"), "parent": c.get("parent") or "root",
                     "author": c.get("author"), "text": (c.get("text") or "").strip(),
                     "cat": cat, "shouty": shouty})

    n = len(rows)
    counts = {}
    for row in rows:
        counts[row["cat"]] = counts.get(row["cat"], 0) + 1
    mood = {k: round(100 * v / n) for k, v in counts.items()}

    # threads: parent comment + its replies
    by_id = {row["id"]: row for row in rows}
    threads = {}
    for row in rows:
        if row["parent"] != "root":
            threads.setdefault(row["parent"], []).append(row)
    # include the parent comment itself in its thread
    for pid, replies in threads.items():
        if pid in by_id:
            replies.insert(0, by_id[pid])

    thread_lines = []
    hom_sum = hom_n = 0
    for pid, members in threads.items():
        if len(members) < 2:
            continue
        stance = [m for m in members if m["cat"] != "нейтрально"]
        if len(stance) < 2:
            thread_lines.append({"thread": (members[0].get("text") or "")[:60],
                                 "author": members[0].get("author"), "size": len(members),
                                 "dominant": None, "homogeneity": None})
            continue
        ccount = {}
        for m in stance:
            ccount[m["cat"]] = ccount.get(m["cat"], 0) + 1
        dom = max(ccount, key=ccount.get)
        hom = round(100 * ccount[dom] / len(stance))
        hom_sum += hom * len(stance); hom_n += len(stance)
        p = by_id.get(pid) or members[0]
        thread_lines.append({"thread": (p.get("text") or "")[:60],
                             "author": p.get("author"), "size": len(members),
                             "dominant": dom, "homogeneity": hom})

    overall_hom = round(hom_sum / hom_n) if hom_n else None

    if args.json:
        print(json.dumps({"title": info.get("title"), "url": info.get("webpage_url"),
                          "comments": n, "mood": mood,
                          "homogeneity_overall": overall_hom,
                          "threads": thread_lines}, ensure_ascii=False))
    else:
        print(f"=== Настроение комментариев ({n}) ===")
        for cat in ["злость", "радость", "недовольство", "нейтрально"]:
            if cat in mood:
                print(f"{cat} {mood[cat]}%")
        shouty = sum(1 for row in rows if row["shouty"])
        print(f"эмоционально окрашенных (без нейтральных): {100 - mood.get('нейтрально', 0)}%, из них «кричащих» {shouty}")
        print(f"\n=== Однородность ===")
        print(f"в среднем по тредам: {overall_hom}%" if overall_hom is not None else "тредов с ответами нет")
        if thread_lines:
            print("по тредам (≥2 комментариев):")
            for t in sorted(thread_lines, key=lambda x: -x["size"]):
                if t["homogeneity"] is None:
                    print(f"  {t['author']} «{t['thread']}…» — не измеряется (нейтральный тред, {t['size']})")
                else:
                    print(f"  {t['author']} «{t['thread']}…» — однородность в треде {t['homogeneity']}% ({t['size']}, доминирует {t['dominant']})")

if __name__ == "__main__":
    main()
