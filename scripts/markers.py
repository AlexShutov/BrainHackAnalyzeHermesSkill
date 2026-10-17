#!/usr/bin/env python3
"""Поверхностный слой: сканирует текст на языковые сигналы (не на термины).

Текст не обязан содержать название приёма. Скрипт находит то, что видно в тексте:
триггеры пресуппозиции, модальность, удаление агента, оценочную лексику,
структурные маркеры и повторы, и возвращает кандидаты устройств из devices.json.

Использование:
  python3 markers.py "текст" [--json]
"""
import json, re, sys, os, collections

HERE = os.path.dirname(os.path.abspath(__file__))
MARKERS = json.load(open(os.path.join(os.path.dirname(HERE), "psychology", "markers.json"), encoding="utf-8"))
DEVICES = json.load(open(os.path.join(os.path.dirname(HERE), "psychology", "devices.json"), encoding="utf-8"))

# какие семейства устройств правдоподобны для каждого семейства маркеров
MAP = {
    "пресуппозиционные триггеры (Levinson 1983)": ["логические", "внушение (книга)", "сквозные (уровень текста)"],
    "модальность и эпистемическая дистанция": ["логические", "сквозные (уровень текста)"],
    "номinalизация и удаление агента": ["логические", "внушение (книга)"],
    "эмоциональные и оценочные маркеры": ["эмоциональные", "внушение (книга)", "сквозные (уровень текста)"],
    "структурные маркеры": ["внушение (книга)", "сквозные (уровень текста)", "эмоциональные"],
}

def window(text, m, pad=30):
    a = max(0, m.start() - pad); b = min(len(text), m.end() + pad)
    return text[a:b].strip()

def scan(text):
    hits = []
    for fam, group in MARKERS["families"].items():
        for label, words in group.items():
            if not isinstance(words, list):
                continue
            for w in words:
                for m in re.finditer(r"(?<!\w)" + re.escape(w.lower()) + r"(?=\W|$)", text.lower()):
                    hits.append({"family": fam, "marker": label, "found": w,
                                 "context": window(text, m), "candidates": MAP[fam]})
    # повтор тезиса: совпадение значимых слов между абзацами
    paras = [p for p in re.split(r"\n+", text) if p.strip()]
    if len(paras) > 1:
        stop = set("и в не на что как это то он она они мы вы я к с у о для по из но а также же ну да уж вот".split())
        sets = [set(re.findall(r"[а-яё]{4,}", p.lower())) - stop for p in paras]
        for i in range(len(paras)):
            for j in range(i + 1, len(paras)):
                shared = sets[i] & sets[j]
                if len(shared) >= 3:
                    hits.append({"family": "структурные маркеры", "marker": "repetition",
                                 "found": ", ".join(sorted(shared)),
                                 "context": f"абзац {i+1} ↔ абзац {j+1}",
                                 "candidates": MAP["структурные маркеры"]})
    return hits

def main():
    if len(sys.argv) < 2:
        print("usage: markers.py <text> [--json]", file=sys.stderr); return 1
    text = sys.argv[1]
    hits = scan(text)
    if "--json" in sys.argv:
        print(json.dumps({"hits": len(hits), "markers": hits}, ensure_ascii=False, indent=1))
    else:
        by = collections.defaultdict(list)
        for h in hits:
            by[(h["family"], h["marker"])].append(h)
        for (fam, marker), items in by.items():
            print(f"{fam} / {marker}: {len(items)}")
            for h in items[:3]:
                print(f"  [{h['found']}] {h['context']}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
