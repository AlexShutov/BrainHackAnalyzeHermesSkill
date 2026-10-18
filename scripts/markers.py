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

# какие семейства устройств правдоподобны для каждого семейства маркеров.
# Значения обязаны быть реальными именами устройств ИЛИ реальными семействами
# из devices.json — иначе атрибуция по имени молча не работает.
# Прежние значения «логические», «эмоциональные» и «внушение (книга)» таких
# семейств в devices.json не имели (логические устройства живут в «структура
# вывода», эмоциональные — в «эмоции и авторитет», внушение — в «приёмы внушения
# (книга)»), и эти 9 значений никогда не могли совпасть с candidate-именем.
FAMILIES = {d["family"] for d in DEVICES["devices"]}
NAMES = {d["name"] for d in DEVICES["devices"]}

MAP = {
    "пресуппозиционные триггеры (Levinson 1983)": ["структура вывода", "приёмы внушения (книга)", "сквозные (уровень текста)"],
    "модальность и эпистемическая дистанция": ["структура вывода", "сквозные (уровень текста)"],
    "номinalизация и удаление агента": ["структура вывода", "приёмы внушения (книга)"],
    "эмоциональные и оценочные маркеры": ["эмоции и авторитет", "приёмы внушения (книга)", "сквозные (уровень текста)"],
    "НЛП: якорение и встраивание состояний": ["Якорение", "Встраивание состояния", "Встроенное указание", "Пресуппозиция"],
    "НЛП: подстройка и раппорт": ["Подстройка (раппорт)", "Управление первым впечатлением", "Встраивание состояния"],
    "НЛП: выбор без выбора": ["Вербовочный паттерн", "Пресуппозиция"],
    "НЛП: подавление и беспомощность": ["Выученная беспомощность", "Уничтожение Я-концепции"],
    "Фокусы языка: намерение и переопределение": ["Фокус: намерение", "Фокус: переопределение"],
    "Фокусы языка: последствия и стратегия реальности": ["Фокус: последствия", "Фокус: стратегия реальности", "Фокус: противоположный пример"],
    "Фокусы языка: фрейминг и сведение к себе": ["Фокус: метафрейм", "Фокус: размер фрейма", "Фокус: модель мира", "Фокус: аналогия", "Фокус: иерархия критериев", "Фокус: применение к себе", "Фокус: другой критерий (результат)", "Фокус: обобщение", "Фокус: разделение"],
    "структурные маркеры": ["приёмы внушения (книга)", "сквозные (уровень текста)", "эмоции и авторитет"],
}

def unknown_map_values():
    """MAP-значения, которых нет в devices.json (ни как устройство, ни как семейство)."""
    return {f: [v for v in vals if v not in NAMES and v not in FAMILIES]
            for f, vals in MAP.items() if any(v not in NAMES and v not in FAMILIES for v in vals)}

def unmapped_families():
    """Семейства маркеров, для которых нет записи в MAP (KeyError при скане)."""
    return sorted(set(MARKERS["families"]) - set(MAP))

def window(text, m, pad=30):
    a = max(0, m.start() - pad); b = min(len(text), m.end() + pad)
    return text[a:b].strip()

def paragraphs(text):
    """Каноническая единица абзаца для всех слоёв (markers, classifier, psych_layer, plugin):
    строка до перевода строки, пустые строки пропускаются. Транскрипты и сообщения
    приходят по одному абзацу на строку; раньше psych_layer делит по пустной строке,
    и индексы абзацев не совпадали с классификатором."""
    return [p for p in re.split(r"\n+", text) if p.strip()]

def scan(text):
    hits = []
    for fam, group in MARKERS["families"].items():
        for label, words in group.items():
            if not isinstance(words, list):
                continue
            for w in words:
                for m in re.finditer(r"(?<!\w)" + re.escape(w.lower()) + r"(?=\W|$)", text.lower()):
                    hits.append({"family": fam, "marker": label, "found": w,
                                 "context": window(text, m), "candidates": MAP.get(fam, [])})
    # повтор тезиса: совпадение значимых слов между абзацами
    paras = paragraphs(text)
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
    bad, missing = unknown_map_values(), unmapped_families()
    if bad or missing:
        print(f"# сверка MAP с devices.json: неизвестные значения {bad}, "
              f"семейства маркеров без записи {missing}", file=sys.stderr)
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
