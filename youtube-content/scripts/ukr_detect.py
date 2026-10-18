#!/usr/bin/env python3
"""Украинский / белорусский / русский: детектор + гейт большинства.

Детерминированный, без LLM и без загрузок:
- Layer 1: буквы. Решающие для спора uk/be: uk_only — ї, є, ґ (в белорусском их нет);
  be_only — ў (в украинском её нет). Общая буква і сама язык не решает.
- Layer 2: слова. UK_WORDS — формы, которые в русском пишутся иначе; BE_WORDS —
  белорусские формы (гэта, шмат, вельмі, каб, тамуж, паняцці, зразумеў, меў, аўтар…).
  Побеждает язык с большим числом доказательств; при общей і без слов — 'amb', не решаем.

Гейт большинства: если >70% текста (транскрипт или комментарии) — не русский,
обработка обычная, 🐷 за язык НЕ ставится. Если большинство русское, а фрагмент
украинский или белорусский (оба называются «мова») — это вставка: 🐷 + токсичность
по оригиналу (мультиязычный XLM-R) и по моему переводу (русские модели, пороги
калиброваны под русский). В описании называется, какой именно язык.

Использование:
  python3 ukr_detect.py --comments flat.json [--transcript file.txt] [--json]
flat.json: список {author, text, parent} (формат fetch_comments).
"""
import argparse, json, re, sys
from collections import Counter

UK_LETTERS = set("іїєґ")
# буквы, которых в белорусском нет: они решают спор uk/be
UK_ONLY = set("їєґ")
# буква, которая есть только в белорусском
BE_ONLY = set("ў")

BE_WORDS = {
    # только формы, которые ОТЛИЧАЮТСЯ от русского; совпадающие с русским сюда не входят
    "гэта", "гэтага", "гэтае", "гэты", "гэтая", "шмат", "вельмі", "каб", "тамуж",
    "паняцці", "зразумеў", "зразумела", "меў", "ўвазе", "аўтар", "таксама",
    "яны", "якая", "які", "якія", "быў", "праўда", "нашы", "вамі", "табе",
    "ёсць", "хочам", "няма", "таму", "ды", "аб", "па", "ў", "вялік", "малы",
    "гаварыў", "гаварыла", "маўчаў", "беларус", "беларуск", "моўна",
}
# общие uk/be: доказательство «не русский», но язык не решают
SHARED_UK_BE = {"як", "але", "і"}
UK_WORDS = {
    # однозначные: в русском пишутся иначе И в белорусском иначе (решают спор uk/be)
    "якщо", "дуже", "вже", "ще", "тільки", "немає", "є",
    "вони", "його", "її", "їх", "був", "була", "було", "має", "мають",
    "поблизу", "навколо", "українськ", "українці", "україні",
    "росіяни", "росія", "чесно", "підміня", "найкращ",
    "отримав", "дід", "вірус", "скачува",
    "свого", "цьому", "проте", "коли", "хоча", "наші", "ваші", "мої",
}
# неоднозначные (есть и в русском) — считаются только вместе с буквой-маркером
AMBIG = {"ми", "бо", "де", "так", "це", "той", "та", "нас", "вас", "їм", "тому", "коли", "хоча", "проте"}

def score(text):
    t = text.lower()
    uk_only = sum(1 for ch in t if ch in UK_ONLY)      # ї, є, ґ — в белорусском нет
    be_only = sum(1 for ch in t if ch in BE_ONLY)      # ў — в украинском нет
    shared = 0                                         # і + як/але: «не русский», но язык не решают
    uk_w = be_w = 0
    for w in re.findall(r"[а-яіїєґў]+", t):
        if w in SHARED_UK_BE:
            shared += 1
        if w in UK_WORDS and w not in AMBIG:
            uk_w += 1
        if w in BE_WORDS:
            be_w += 1
    return uk_only, be_only, shared, uk_w, be_w


def lang_of(text, min_ev=1):
    """uk / be / amb / ru. Общая 'і' и общие слова (як, але) сами язык не решают."""
    uk_only, be_only, shared, uk_w, be_w = score(text)
    ev_uk = uk_only + uk_w
    ev_be = be_only + be_w
    if ev_be > ev_uk:
        return "be"
    if ev_uk >= min_ev:
        return "uk"
    if shared >= 2:
        return "amb"          # не русский, но uk/be не различить
    return "ru"


def detect_item(text):
    """(lang, segments) — lang всего текста; сегменты с собственным языком для смешанных.
    Сегменты — clauses (по .,?!;:), чтобы вставка в русском комментарии была видна отдельно."""
    lang = lang_of(text)
    if lang == "ru":
        return "ru", []
    segs = []
    for clause in re.split(r"[.!?\n,;:]+", text):
        clause = clause.strip().strip("«»\"")
        if not clause:
            continue
        cl = lang_of(clause)
        if cl != "ru":
            segs.append((cl, clause))
    if segs and len(" ".join(s for _, s in segs)) < len(text) * 0.85:
        return "mixed", segs
    return lang, segs

def majority(items):
    langs = Counter(detect_item(t)[0] for t in items)
    total = sum(langs.values())
    nonru = langs["uk"] + langs["be"] + langs["mixed"]
    return langs, nonru / total if total else 0

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--comments", help="flat json (author/text/parent)")
    ap.add_argument("--transcript", help="текст транскрипта (строки с таймкодами или plain)")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()

    out = {}
    if a.comments:
        items = json.load(open(a.comments))
        texts = [c["text"] for c in items]
        langs, share = majority(texts)
        ins = []
        for i, c in enumerate(items):
            lang, segs = detect_item(c["text"])
            if lang in ("uk", "be", "mixed") and segs:
                ins.append({"i": i, "author": c.get("author"), "lang": lang,
                            "segments": segs, "text": c["text"][:200]})
        out["comments"] = {"n": len(items), "langs": dict(langs), "nonru_share": round(share, 3),
                           "insertions": ins}
    if a.transcript:
        raw = open(a.transcript).read()
        # строки вида "0:12 текст" — группируем по абзацам-строкам
        paras = [l for l in raw.splitlines() if l.strip()]
        texts = [re.sub(r"^\d+:\d+(:\d+)?\s*", "", l) for l in paras]
        texts = [t for t in texts if t.strip()]
        langs, share = majority(texts)
        ins = []
        for idx, (line, t) in enumerate(zip(paras, paras)):
            lang, segs = detect_item(t)
            if lang in ("uk", "be", "mixed") and segs:
                ts = re.match(r"^(\d+:\d+(?::\d+)?)", line)
                ins.append({"t": ts.group(1) if ts else None, "lang": lang,
                            "segments": segs, "text": t[:200]})
        out["transcript"] = {"n": len(texts), "langs": dict(langs), "nonru_share": round(share, 3),
                             "insertions": ins}
    print(json.dumps(out, ensure_ascii=False, indent=1) if a.json else json.dumps(out, ensure_ascii=False))

if __name__ == "__main__":
    main()
