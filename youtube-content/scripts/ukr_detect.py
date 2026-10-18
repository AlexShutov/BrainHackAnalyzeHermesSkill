#!/usr/bin/env python3
"""Украинский/русский язык: детектор + гейт большинства.

Детерминированный, без LLM и без загрузок:
-Layer 1: уникальные буквы украинского (і, ї, ґ, є) — в русском тексте их нет
  (кроме имён/транслитерации), поэтому это решающий сигнал для коротких комментариев.
- Layer 2: служебные слова, которые в украинском отличаются от русского
  (як, але, якщо, тому що, дуже, вже, ще, тільки, немає, є, вони, був/була/було,
  між, біля, коло, поблизу, навколо, українськ-, мова).

Гейт большинства: если >70% текста (транскрипт или комментарии) — украинский,
обработка обычная, 🐷 за язык НЕ ставится. Если большинство русское, а фрагмент
украинский — это вставка: 🐷 + токсичность по оригиналу (мультиязычный XLM-R)
и по моему переводу (русские модели, пороги калиброваны под русский).

Использование:
  python3 ukr_detect.py --comments flat.json [--transcript file.txt] [--json]
flat.json: список {author, text, parent} (формат fetch_comments).
"""
import argparse, json, re, sys
from collections import Counter

UK_LETTERS = set("іїґє")
UK_WORDS = {
    # однозначные: в русском пишутся иначе
    "як", "але", "якщо", "дуже", "вже", "ще", "тільки", "немає", "є",
    "вони", "його", "її", "їх", "був", "була", "було", "має", "мають",
    "між", "біля", "поблизу", "навколо", "мова", "українськ", "українці",
    "росіяни", "росія", "національн", "чесно", "підміня", "найкращ",
    "отримав", "дід", "вірус", "скачува",
    "свого", "цьому", "проте", "коли", "хоча", "ті", "наші", "ваші", "мої",
    "було", "є", "якщо", "тому", "бо", "де", "так", "це", "той", "та",
}
# неоднозначные (есть и в русском) — считаются только вместе с буквой-маркером
AMBIG = {"ми", "бо", "де", "так", "це", "той", "та", "нас", "вас", "їм", "тому", "коли", "хоча", "проте"}

def score(text):
    t = text.lower()
    letters = sum(1 for ch in t if ch in UK_LETTERS)
    words = 0
    for w in re.findall(r"[а-яґ]+", t):
        if w in UK_WORDS and w not in AMBIG:
            words += 1
    # «є» как отдельное слово-маркер уже в UK_WORDS
    return letters, words

def is_ukrainian(text, min_letters=2, min_words=1):
    letters, words = score(text)
    # буквы-маркёр — решающий сигнал; однозначные слова считаются даже по одному вхождению
    return letters >= min_letters or words >= min_words

def detect_item(text):
    """ Возвращает (lang, score) для всего текста и (lang, сегменты) для смешанных. """
    if is_ukrainian(text):
        # найти украинские сегменты, если текст смешанный
        segs = []
        for sent in re.split(r"[.!?\n]+", text):
            if is_ukrainian(sent, min_letters=1, min_words=1):
                segs.append(sent.strip())
        if segs and len(" ".join(segs)) < len(text) * 0.85:
            return "mixed", segs
        return "uk", segs
    return "ru", []

def majority(items):
    langs = Counter(detect_item(t)[0] for t in items)
    total = sum(langs.values())
    uk = langs["uk"] + langs["mixed"]
    return langs, uk / total if total else 0

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
            if lang in ("uk", "mixed") and segs:
                ins.append({"i": i, "author": c.get("author"), "lang": lang,
                            "segments": segs, "text": c["text"][:200]})
        out["comments"] = {"n": len(items), "langs": dict(langs), "uk_share": round(share, 3),
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
            if lang in ("uk", "mixed") and segs:
                ts = re.match(r"^(\d+:\d+(?::\d+)?)", line)
                ins.append({"t": ts.group(1) if ts else None, "lang": lang,
                            "segments": segs, "text": t[:200]})
        out["transcript"] = {"n": len(texts), "langs": dict(langs), "uk_share": round(share, 3),
                             "insertions": ins}
    print(json.dumps(out, ensure_ascii=False, indent=1) if a.json else json.dumps(out, ensure_ascii=False))

if __name__ == "__main__":
    main()
