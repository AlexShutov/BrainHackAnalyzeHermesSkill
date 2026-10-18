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
    "крутой", "классн", "отлично", "отличн", "здорово", "люблю",
    "любим", "счасть", "счастл", "молодец", "молодцы", "красота", "красиво",
    "супер", "нравится", "нравились", "приятно", "приятн", "успех",
    "успешн", "поздравляю", "ура", "восторг", "восхищ", "лучший", "лучшие",
    # "класс" убран: цепляется как подстрока в "классический"/"классификация"
    "прогресс", "огонь",
]
DISCONTENT = [
    # "не нравится"/"не нравятся"/"не понравился" убраны: их базовая форма
    # ("нравится") стоит в JOY, и отрицание обрабатывается правилом flip —
    # иначе одна и та же фраза считается дважды
    "недоволь", "плохо",
    "плохой", "плохие", "плохое", "хуже", "несправедлив", "разочаров",
    "не согласен", "не согласна", "не согласны", "фу ", "скучн", "слабо",
    "хренов", "ужасн", "кошмар", "проблем", "загибается", "не надо",
    "не стоит", "не умеет", "не умеют", "не может", "не могут",
    "не понимает", "не понимают", "не видят", "не слышат", "обман",
    "врут", "врали", "враньё", "инфоцыган", "развод", "надувательств",
    "неэффективн", "бесполезн", "бессмысленн", "не работает",
    "не получается", "не успе", "не готов", "не заточен", "не выполняет",
    "не верят", "не ценят", "не уважают", "игнорируют", "замалчива",
    # «не надо» уже стоит выше — дубликат заставлял считать одно и то же слово дважды
    "не нужен", "не нужна", "не нужны",
]
CATS = {"злость": ANGER, "радость": JOY, "недовольство": DISCONTENT}

NEG = ("не", "ни")          # отрицание переворачивает позитивную категорию
SHORT_WHOLE = 3             # короткие стемы (ура, бык, фу) ловятся только как целое слово
TIE_ORDER = ["злость", "недовольство", "радость"]   # детерминированный порядок при ничьей

def _pat(stem):
    s = re.escape(stem.strip())
    if " " in stem or len(stem.strip()) <= SHORT_WHOLE:
        return re.compile(r"(?<!\w)" + s + r"(?!\w)")   # целое слово / фраза
    return re.compile(r"(?<!\w)" + s)                   # стем-префикс (ловит флексии)

_PATTERNS = {stem: _pat(stem) for stems in CATS.values() for stem in stems}

def classify(text):
    t = text.lower()
    scores = {}
    for cat, stems in CATS.items():
        for stem in stems:
            for m in _PATTERNS[stem].finditer(t):
                eff = cat
                if cat in ("злость", "радость"):
                    before = t[max(0, m.start() - 3):m.start()]
                    if before.startswith("не ") or before.startswith("ни "):
                        eff = "недовольство"   # «не лучший», «не радость» — не позитив
                scores[eff] = scores.get(eff, 0) + 1
    if not scores:
        return "нейтрально", False
    best = sorted(scores.items(), key=lambda kv: (-kv[1], TIE_ORDER.index(kv[0])))[0][0]
    # emotional intensity: shouty caps
    letters = [c for c in text if c.isalpha()]
    shouty = letters and sum(1 for c in letters if c.isupper()) / len(letters) > 0.4
    return best, shouty

def build_threads(rows):
    """Корневой комментарий + ВСЕ его потомки. Reply-on-reply остаётся в том же
    треде — раньше каждый вложенный ответ открывал новый тред.

    Корень ищется двумя путями: если id-цепочка разрешается внутри выборки — идём
    по parent вверх; если id в выборке нет (flat-формат fetch_comments, где есть
    только parent), корень берётся как префикс parent до первой точки — это и есть
    id корневого комментария в схеме YouTube (<root>.<suffix>)."""
    by_id = {row["id"]: row for row in rows}

    def root_key(row):
        cur, seen = row, set()
        while cur:
            p = cur.get("parent")
            if not p or p == "root":
                return cur["id"]
            if p in by_id:
                if cur["id"] in seen:
                    break                      # защита от цикла в parent
                seen.add(cur["id"])
                cur = by_id[p]
            else:
                return p.split(".")[0]         # цепочка не разрешается — префикс
        return row["id"]

    threads = {}
    for row in rows:
        if not row.get("parent") or row["parent"] == "root":
            continue
        threads.setdefault(root_key(row), []).append(row)
    for rid, members in threads.items():
        if rid in by_id:
            members.insert(0, by_id[rid])
    return threads, by_id


def thread_homogeneity(threads, by_id):
    """Один тред = одна строка. overall — простое среднее по тредам (то, что
    доставляется в чат), weighted — среднее, взвешенное по размеру тредов."""
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
        # ничья решается по TIE_ORDER — тому же порядку, что и classify;
        # раньше здесь был порядок словаря (фактически порядок появления слов в тексте)
        dom = sorted(ccount.items(), key=lambda kv: (-kv[1], TIE_ORDER.index(kv[0])))[0][0]
        hom = round(100 * ccount[dom] / len(stance))
        hom_sum += hom * len(stance); hom_n += len(stance)
        p = by_id.get(pid) or members[0]
        thread_lines.append({"thread": (p.get("text") or "")[:60],
                             "author": p.get("author"), "size": len(members),
                             "dominant": dom, "homogeneity": hom})
    homs = [t["homogeneity"] for t in thread_lines if t["homogeneity"] is not None]
    overall = round(sum(homs) / len(homs)) if homs else None
    weighted = round(hom_sum / hom_n) if hom_n else None
    return thread_lines, overall, weighted


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

    # threads: корневой комментарий + ВСЕ его потомки (reply-on-reply в том же треде)
    threads, by_id = build_threads(rows)
    thread_lines, overall_hom, weighted_hom = thread_homogeneity(threads, by_id)

    if args.json:
        print(json.dumps({"title": info.get("title"), "url": info.get("webpage_url"),
                          "comments": n, "mood": mood,
                          "homogeneity_overall": overall_hom,
                          "homogeneity_weighted": weighted_hom,
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
        if weighted_hom is not None and weighted_hom != overall_hom:
            print(f"взвешенная по размеру тредов: {weighted_hom}%")
        if thread_lines:
            print("по тредам (≥2 комментариев):")
            for t in sorted(thread_lines, key=lambda x: -x["size"]):
                if t["homogeneity"] is None:
                    print(f"  {t['author']} «{t['thread']}…» — не измеряется (нейтральный тред, {t['size']})")
                else:
                    print(f"  {t['author']} «{t['thread']}…» — однородность в треде {t['homogeneity']}% ({t['size']}, доминирует {t['dominant']})")

if __name__ == "__main__":
    main()
