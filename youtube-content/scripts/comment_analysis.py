#!/usr/bin/env python3
"""Единая точка анализа комментариев — один блок на всех трёх путях.

Правило навыка: нейро-слой (rubert) обязателен на каждом анализе. Раньше он был только
в YouTube-пути (`neural_pass.py`), а Rutube и Telegram считали настроение лексиконом —
доставляемые числа расходились в зависимости от платформы. Здесь один модуль-вход,
который использует нейро на всех трёх путях и оставляет лексикон как кросс-проверку.

Вход — каноническая строка комментария: {id, author, text, parent, likes}.
Выход — {layer, mood, cross_check, threads, overall, weighted, tags}.

Если нейро недоступен (transformers не установлен, модели не скачаны, нет сети), слой
честно сообщает об этом и работает лексиконом — числа не выдумываются.

Нейро-метки (28 эмоций GoEmotions) группируются в четыре канонические корзины
доставляемого блока — это фиксированное отображение, а не предположение по тексту:
  злость       anger, annoyance, disgust
  недовольство disapproval, disappointment, sadness, grief, nervousness, fear,
               embarrassment, remorse, confusion
  радость      admiration, amusement, approval, caring, desire, excitement, gratitude,
               joy, love, optimism, pride, relief
  нейтрально   neutral, realization, surprise
"""
import json, os, re, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import comment_sentiment as CS

NEURAL_BUCKET = {
    "anger": "злость", "annoyance": "злость", "disgust": "злость",
    "disapproval": "недовольство", "disappointment": "недовольство",
    "sadness": "недовольство", "grief": "недовольство", "nervousness": "недовольство",
    "fear": "недовольство", "embarrassment": "недовольство", "remorse": "недовольство",
    "confusion": "недовольство",
    "admiration": "радость", "amusement": "радость", "approval": "радость",
    "caring": "радость", "desire": "радость", "excitement": "радость",
    "gratitude": "радость", "joy": "радость", "love": "радость", "optimism": "радость",
    "pride": "радость", "relief": "радость",
    "neutral": "нейтрально", "realization": "нейтрально", "surprise": "нейтрально",
}

# Пороги правила: 🤮 — высокая токсичность (p >= 0.70 по insult/obscenity/threat или
# несколько токсичных меток сразу), 🤢 — малая степень (p 0.30-0.70 или сарказм).
TOX_HIGH = 0.70
TOX_LOW = 0.30
TOXIC_LABELS = ("insult", "obscenity", "threat", "dangerous")

# Смерть человека / суицид. Смерть техники («умер мотор») и животного — НЕ ☠️.
# Docstring утверждал, что «умер» в списке нет, но он там был — противоречие.
# Две группы: сильные слова сами говорят о человеке (суицид, гробовые), слабые
# («умер», «погиб») требуют субъекта-человека в том же комментарии.
# Цена правила зафиксирована явно: без человека-субъекта метка не ставится.
DEATH_STRONG = ["суицид", "самоубий", "покончил", "покончит", "на гробовые", "гробовые"]
DEATH_WEAK = ["умер", "умерла", "умерли", "смерт", "погиб", "погибла", "погибли",
              "убили", "убьют", "убьёт"]
HUMAN = ["человек", "человека", "люди", "людям", "женщин", "мужчин", "реб", "дитя",
         "мама", "папа", "бабушк", "дедушк", "солдат", "воин", "мальчик", "девочк",
         "парень", "девушка", "он ", "она ", "они ", "его ", "её "]
NON_HUMAN = ["мотор", "двигател", "машина", "автомобил", "техник", "телевизор", "телефон",
             "ноутбук", "комп", "аккумулятор", "батарейк", "холодильник", "кондиционер",
             "кошка", "кот", "пёс", "пес", "собак", "животн", "птиц", "лошад"]


def death_tag(text):
    """☠️ только на смерти человека. Техника и животные — не ☠️ (правило навыка)."""
    low = (text or "").lower()
    if any(w in low for w in DEATH_STRONG):
        return "☠️"
    if any(w in low for w in DEATH_WEAK):
        if any(w in low for w in NON_HUMAN):
            return None
        if any(w in low for w in HUMAN):
            return "☠️"
    return None

# Продающий язык (модификатор 💸): CTA, дефицит/срочность, ценовое якорение,
# социальное доказательство с непроверяемыми цифрами, лженаучная апелляция, подарок
# за немедленную покупку.
MARKETING = ["купите", "заказывайте", "закажи", "только сегодня", "только сейчас",
             "осталось мало", "осталось ", "скидка до", "акция до", "обычная цена",
             "для вас", "выбор 9", "90%", "учёные доказали", "доказано учёными",
             "бонус за", "подарок за", "по ссылке", "промокод"]

PRIORITY = ["💸", "приём", "☠️", "🤮", "🤢", "🔴", "😊", "🙂"]


def neural_runner(models=("emotion", "toxicity", "sarcasm")):
    """Тёплый нейро-прогон: модели загружаются один раз, батч по 32.
    Возвращает None, если слой недоступен — вызывающий честно идёт лексиконом."""
    try:
        import neural_pass as NP
        import torch
        from transformers import AutoTokenizer
    except Exception:
        return None
    try:
        loaded = {}
        for name in models:
            tok = AutoTokenizer.from_pretrained(NP.REPOS[name])
            model, labels = NP.load_no_mmap(NP.REPOS[name])
            loaded[name] = (tok, model, labels, NP.MULTILABEL[name])
    except Exception:
        return None

    def run(texts, batch=32):
        import torch
        out = {name: [] for name in loaded}
        for name, (tok, model, labels, multi) in loaded.items():
            for i in range(0, len(texts), batch):
                chunk = texts[i:i + batch]
                enc = tok(chunk, return_tensors="pt", truncation=True,
                          max_length=512, padding=True)
                with torch.no_grad():
                    logits = model(**enc).logits
                for row in logits:
                    p = (torch.sigmoid(row) if multi else torch.softmax(row, dim=-1)).tolist()
                    out[name].append({labels[j]: round(float(v), 3) for j, v in enumerate(p)})
        return out

    return run


def tag(text, cat, scores, device=None):
    """Один основной тег на комментарий, по приоритету правила:
    💸 → приём → ☠️ → 🤮 → 🤢 → 🔴 → 😊 → 🙂
    Строка порядка — это и есть PRIORITY; тест сверяет их и проверяет порядок
    поведением (комментарий, где применимы сразу две категории)."""
    low = text.lower()
    if any(w in low for w in MARKETING):
        return "💸"
    if device:
        return device
    d = death_tag(text)
    if d:
        return d
    tox = scores.get("toxicity") or {}
    toxic_hits = [l for l in TOXIC_LABELS if tox.get(l, 0) >= TOX_LOW]
    if any(tox.get(l, 0) >= TOX_HIGH for l in TOXIC_LABELS) or len(toxic_hits) >= 2:
        return "🤮"
    # 🤢 — малая степень: нужен порог TOX_LOW, а не любое ненулевое значение.
    # Раньше insult p=0.01 давал 🤢 — комментарий размечался токсичным без основания.
    if any(tox.get(l, 0) >= TOX_LOW for l in TOXIC_LABELS) or \
       (scores.get("sarcasm") or {}).get("sarcasm", 0) >= TOX_LOW:
        return "🤢"
    if cat in ("злость", "недовольство"):
        return "🔴"
    if cat == "радость":
        return "😊"
    return "🙂"


def analyze(rows, runner=None, device_by_text=None, use_neural=True):
    """rows — каноническая форма комментария. device_by_text — {текст: имя приёма}
    от маркерного слоя (для строки «приём»); без него приём не называется, и тег
    не ставится: правило требует уверенного вердикта, а не совпадения слов.

    `use_neural=False` — явный лексикон (для сверки цифр); `runner` — готовый прогон,
    иначе берётся тёплый neural_runner()."""
    texts = [r.get("text") or "" for r in rows]
    scores = None
    if runner is None and use_neural:
        runner = neural_runner()
    if runner:
        try:
            scores = runner(texts)
        except Exception:
            scores = None
    layer = "neural" if scores else "lexicon"

    cats = []
    for i, r in enumerate(rows):
        lex_cat, _ = CS.classify(r.get("text") or "")
        cat = lex_cat
        if scores and scores.get("emotion") and scores["emotion"][i]:
            top = max(scores["emotion"][i].items(), key=lambda kv: kv[1])
            cat = NEURAL_BUCKET.get(top[0], lex_cat)
        cats.append(cat)

    mood = {}
    for c in cats:
        mood[c] = mood.get(c, 0) + 1
    n = len(cats) or 1
    mood_pct = {c: round(100 * v / n) for c, v in sorted(mood.items(), key=lambda kv: -kv[1])}

    cross = {}
    for r in rows:
        lex_cat, _ = CS.classify(r.get("text") or "")
        cross[lex_cat] = cross.get(lex_cat, 0) + 1
    cross_pct = {c: round(100 * v / n) for c, v in sorted(cross.items(), key=lambda kv: -kv[1])}

    tagged = []
    for i, r in enumerate(rows):
        tagged.append({**r, "cat": cats[i],
                       "tag": tag(r.get("text") or "", cats[i],
                                  {k: (v[i] if v else {}) for k, v in (scores or {}).items()},
                                  (device_by_text or {}).get(r.get("text") or ""))})

    threads, by_id = CS.build_threads(tagged)
    thread_lines, overall, weighted = CS.thread_homogeneity(threads, by_id)
    return {"layer": layer, "count": len(rows), "mood": mood_pct,
            "cross_check": cross_pct, "overall": overall, "weighted": weighted,
            "threads": thread_lines, "comments": tagged}


def block(res):
    """Доставляемая форма: распределение по всем комментариям + сводка по тредам,
    без ников и без цитат комментариев (правило пользователя, 2026-10-18)."""
    lines = [f"Настроение комментариев ({res['count']}): " +
            ", ".join(f"{c} {p}%" for c, p in res["mood"].items()) +
            f". Слой: {res['layer']}."]
    if res["layer"] == "neural":
        lines.append("Кросс-проверка лексиконом: " +
                     ", ".join(f"{c} {p}%" for c, p in res["cross_check"].items()))
    lines.append(f"Однородность в треде: в среднем по тредам {res['overall']}% "
                 f"(взвешенная {res['weighted']}%).")
    active = [t for t in res["threads"] if t["homogeneity"] is not None]
    for t in active:
        lines.append(f"тред {t['size']} — доминирует {t['dominant']}, "
                     f"однородность {t['homogeneity']}%")
    counts = {}
    for c in res["comments"]:
        counts[c["tag"]] = counts.get(c["tag"], 0) + 1
    lines.append("Теги по комментариям: " + ", ".join(
        f"{t} {v}" for t, v in sorted(counts.items(), key=lambda kv: -kv[1])))
    lines.append("Классификатор — модель, не словарь: сарказм и ирония могут быть "
                 "ошибочно классифицированы; однородность измеряет эмоциональную "
                 "окраску, а не смысловое согласие.")
    return "\n".join(lines)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", help="JSON с массивом комментариев (id, author, text, parent)")
    ap.add_argument("--no-neural", action="store_true", help="только лексикон")
    a = ap.parse_args()
    if not a.file:
        print("usage: comment_analysis.py --file comments.json [--no-neural]", file=sys.stderr)
        return 1
    with open(a.file, encoding="utf-8") as f:
        rows = json.load(f)
    res = analyze(rows, runner=None if a.no_neural else neural_runner())
    print(json.dumps(res, ensure_ascii=False, indent=1))
    print(block(res), file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
