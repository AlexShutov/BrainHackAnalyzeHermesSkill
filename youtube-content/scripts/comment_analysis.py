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
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                "..", "..", "scripts")))
import comment_sentiment as CS
import tag_routing as TR

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
    # curiosity — интерес и вовлечённость БЕЗ оценки: это не stance, поэтому метка идёт
    # в «нейтрально» и не считается содержательной в однородности треда. Раньше её не было
    # в корзине (27 из 28), и комментарий молча уходил в лексикон — доставляемые цифры
    # смешивали два слоя, не говоря об этом.
    "curiosity": "нейтрально",
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
#
# Правило: 💸 — за сам приём, а не за слово, которое похоже на продажу. Прежний
# плоский список (подстрока в тексте) давал ложеловный 💸 на бытовых фразах:
# «осталось только подождать» (не дефицит товара), «это для вас» (не спеццена),
# «90% комментариев» (не социальное доказательство), «источник по ссылке» (не канал
# продажи). Поэтому каждый приём — группа: триггер плюс обязательное окружение,
# которое делает фразу приёмом. Группа без окружения — только те приёмы, где сам
# триггер уже является приёмом (императивный призыв, лженаучная апелляция).
MARKETING_GROUPS = [
    ("призыв к действию", ["купите", "заказывайте", "закажи", "берите", "оформите",
                           "переходите"], None),
    ("дефицит/срочность", ["осталось", "осталась", "остались"],
     ["мало", "немного", "последни", "заканчив", "всего ", "штук", "секунд",
      "минут", "до конца", "только сегодня", "только сейчас"]),
    ("ценовое якорение", ["обычная цена", "цена была", "скидка до", "акция до",
                          "спеццена", "для вас"],
     ["цена", "цене", "скидк", "акци", "стоим", "спеццена", "товар", "продукт",
      "услуг"]),
    ("социальное доказательство с непроверяемой цифрой",
     ["выбор", "рекомендуют", "купили", "отзывов", "довольны"],
     ["%", "миллион", "тысяч", "больше всех", "90", "95", "99"]),
    ("лженаучная апелляция", ["учёные доказали", "доказано учёными",
                              "наука доказала", "доказано наукой"], None),
    ("подарок за немедленную покупку", ["бонус за", "подарок за", "подарок при",
                                        "бонус при"],
     ["покупк", "заказ", "оплат", "пополн", "сразу", "немедленно", "сейчас"]),
    ("ссылка как канал продажи", ["по ссылке", "переход по ссылке"],
     ["куп", "заказ", "скидк", "промокод", "заходи", "оформ"]),
]


def marketing_device(text):
    """Имя продающего приёма или None. Возвращает имя, а не просто факт совпадения:
    правило требует называть конкретный приём, а не ставить модификатор вслепую."""
    low = (text or "").lower()
    for name, triggers, context in MARKETING_GROUPS:
        if any(t in low for t in triggers):
            if not context or any(c in low for c in context):
                return name
    return None


PRIORITY = ["💸", "приём", "☠️", "🤮", "🤢", "🔴", "😊", "🙂"]


NEURAL_MODELS = ("emotion", "toxicity", "sarcasm")


def load_models(models=NEURAL_MODELS):
    """Загрузка моделей в процессе — запасной путь, когда тёплый worker не поднялся."""
    try:
        import neural_pass as NP
        from transformers import AutoTokenizer
    except Exception:
        return None
    try:
        loaded = {}
        for name in models:
            tok = AutoTokenizer.from_pretrained(NP.REPOS[name])
            model, labels = NP.load_no_mmap(NP.REPOS[name])
            loaded[name] = (tok, model, labels, NP.MULTILABEL[name])
        return loaded
    except Exception:
        return None


def neural_runner(models=NEURAL_MODELS, client=None, loader=None):
    """Тёплый нейро-прогон: модели загружаются ОДИН раз, батч по 32.

    Правило навыка: тёплый worker — один живой процесс на хозяина, и через него обязаны
    ходить ВСЕ пути. Раньше это выполнялось только для плагина: путь комментариев
    загружал три модели на каждый вызов, то есть каждый анализ (каждый запуск скрипта —
    отдельный процесс) платил загрузку rubert заново. Теперь путь идёт через тот же
    worker (op `neural`), а загрузка в процессе остаётся запасным путём — деградация,
    а не отказ анализа. Возвращает None, если слой недоступен: вызывающий честно
    работает лексиконом и говорит об этом в доставляемом блоке.
    """
    loader = loader or load_models
    loaded = None

    def local_run(texts, batch=32):
        nonlocal loaded
        if loaded is None:
            loaded = loader(models)
        if not loaded:
            return None
        import neural_pass as NP
        out = {name: [] for name in loaded}
        for name, (tok, model, labels, multi) in loaded.items():
            for row in NP.infer(tok, model, labels, texts, multi, name, batch):
                out[name].append(row["all"])
        return out

    def warm_run(texts, batch=32):
        r = c.call("neural", texts=list(texts), models=list(models), batch=batch)
        if r and r.get("ok"):
            return r["scores"]
        if r and r.get("released"):
            # память уже отдана memory guard worker: перезагружать модели здесь нельзя —
            # это вернуло бы те же гигабайты, ради которых их освободили
            warm_run.released = True
            return None
        return local_run(texts, batch)      # worker умер — деградация, не отказ
    warm_run.released = False

    try:
        import worker_client
        c = client or worker_client.client()
        warm = c.call("neural", texts=["прогрев"], models=list(models))
        if warm and warm.get("ok"):
            return warm_run
        if warm and warm.get("released"):
            # память уже отдана worker по guard: загрузка в этом процессе вернула бы
            # те же гигабайты — слой остаётся лексиконом, и блок называет причину
            return warm_run
    except Exception:
        pass
    loaded = loader(models)          # один раз, и тот же кэш использует local_run
    return local_run if loaded else None


def device_pairs(text):
    """Пара подтипа по маркерам + вердикту классификатора на абзаце — тот же
    словарь маршрутизации, что у пути текста (psych_layer), через tag_routing.
    Раньше ни один из трёх путей (YouTube / Rutube / Telegram) не передавал
    device_by_text, и «приём» в блоке комментариев был невозможен: приоритет
    «приём» никогда не срабатывал, и вместо подтипа показывалось имя устройства."""
    try:
        import markers, classifier
    except Exception:
        return None
    try:
        hits = markers.scan(text)
        if not hits:
            # гейт слоя 3: классификатор вызывается только там, где сработали маркеры
            return None
        rows = classifier.classify(text, min_conf=0.70)
        # Комментарий — один абзац. `classifier.classify()` сама `para` не несёт (её
        # добавляет тёплый worker), поэтому строке приписываем абзац маркерного hit-а:
        # иначе пары склеиваются по ключу None и вердикт модели теряется.
        para = hits[0].get("para")
        pairs = TR.device_pairs(hits, [dict(r, para=para) for r in rows])
    except Exception:
        return None
    return next(iter(pairs.values()), None)


def tag(text, cat, scores, device=None):
    """Один основной тег на комментарий, по приоритету правила:
    💸 → приём → ☠️ → 🤮 → 🤢 → 🔴 → 😊 → 🙂
    Строка порядка — это и есть PRIORITY; тест сверяет их и проверяет порядок
    поведением (комментарий, где применимы сразу две категории)."""
    low = text.lower()
    if marketing_device(text):
        return "💸"
    if device:
        # приём — это пара подтипа, а не имя устройства: имя без пары —
        # незавершённая метка (правило: ⚫ без пары — неполная разметка)
        p = TR.pair_for_device(device)
        if p:
            return p
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


def analyze(rows, runner=None, device_by_text=None, use_neural=True, use_markers=True,
           partial=None):
    """rows — каноническая форма комментария. device_by_text — {текст: имя приёма}
    от маркерного слоя (для строки «приём»); без него приём не называется, и тег
    не ставится: правило требует уверенного вердикта, а не совпадения слов.

    `use_neural=False` — явный лексикон (для сверки цифр); `runner` — готовый прогон,
    иначе берётся тёплый neural_runner(). `use_markers=False` — не запускать маркерный
    слой (для сверки цифр).

    `partial` — {"fetched": N, "analyzed": M} (M < N): анализ сделан по срезу (--limit)
    или по неполному кэшу. Правило пользователя: доставляемые числа обязаны называть,
    что выборка неполная, а не выдавать её за все комментарии. block() печатает строку."""
    texts = [r.get("text") or "" for r in rows]
    if device_by_text is None and use_markers:
        device_by_text = {t: device_pairs(t) for t in texts}
    scores = None
    if runner is None and use_neural:
        runner = neural_runner()
    if runner:
        try:
            scores = runner(texts)
        except Exception:
            scores = None
    layer = "neural" if scores else "lexicon"
    # честность доставляемого блока: если нейро был запрошен и не смог загрузиться,
    # числа получены словарём, а не моделью — об этом говорится отдельной строкой.
    # Раньше block() этого не говорил, и лексикон подавался как нейро-вердикт.
    neural_unavailable = bool(use_neural) and not scores
    # отдельный случай: слой был доступен, но worker отдал память по guard —
    # доставляемый блок обязан называть причину, а не писать «модели недоступны»
    neural_released = bool(getattr(runner, "released", False))

    cats = []
    unmapped = []
    for i, r in enumerate(rows):
        text = r.get("text") or ""
        if scores and scores.get("emotion") and scores["emotion"][i]:
            top = max(scores["emotion"][i].items(), key=lambda kv: kv[1])
            if top[0] in NEURAL_BUCKET:
                cat = NEURAL_BUCKET[top[0]]
            else:
                # метка модели без корзины: число по этому комментату получил словарь,
                # и об этом говорится в доставляемом блоке — молча смешивать слои нельзя
                cat = CS.classify(text)[0]
                unmapped.append(top[0])
        else:
            cat = CS.classify(text)[0]
        cats.append(cat)

    mood = {}
    for c in cats:
        mood[c] = mood.get(c, 0) + 1
    mood_pct = CS.percentages(mood)

    # Кросс-проверка имеет смысл только когда доставляемые числа дал нейро-слой.
    # Раньше она считалась всегда: когда слой и есть лексикон, это второй проход по
    # тому же словарю (мёртвая работа), и блок печатал «кросс-проверку», которая
    # просто повторяет те же самые числа — читатель видит сверку, которой нет.
    cross_pct = None
    if scores:
        cross = {}
        for r in rows:
            lex_cat, _ = CS.classify(r.get("text") or "")
            cross[lex_cat] = cross.get(lex_cat, 0) + 1
        cross_pct = CS.percentages(cross)

    tagged = []
    for i, r in enumerate(rows):
        t = r.get("text") or ""
        tagged.append({**r, "cat": cats[i],
                       "marketing": marketing_device(t),
                       "tag": tag(t, cats[i],
                                  {k: (v[i] if v else {}) for k, v in (scores or {}).items()},
                                  (device_by_text or {}).get(t))})

    threads, by_id = CS.build_threads(tagged)
    thread_lines, overall, weighted = CS.thread_homogeneity(threads, by_id)
    return {"layer": layer, "neural_unavailable": neural_unavailable,
            "neural_released": neural_released, "count": len(rows),
            "mood": mood_pct, "cross_check": cross_pct, "overall": overall,
            "weighted": weighted, "threads": thread_lines, "comments": tagged,
            "unmapped": unmapped, "partial": partial}


def block(res):
    """Доставляемая форма: распределение по всем комментариям + сводка по тредам,
    без ников и без цитат комментариев (правило пользователя, 2026-10-18)."""
    lines = [f"Настроение комментариев ({res['count']}): " +
            ", ".join(f"{c} {p}%" for c, p in res["mood"].items()) +
            f". Слой: {res['layer']}."]
    p = res.get("partial")
    if p:
        # правило: неполная выборка называется, а не подаётся как все комментарии
        if p.get("note"):
            lines.append("Выборка может быть неполной: " + p["note"])
        else:
            lines.append(f"Выборка неполная: {p['analyzed']} из {p['fetched']} — "
                         f"числа по этой выборке, не по всем комментариям.")
    if res.get("unmapped"):
        # честность: если у метки модели нет корзины, число по этому комментату
        # получил словарь — об этом говорится отдельной строкой, а не молча
        lines.append("Часть цифр получена лексиконом: метка модели без корзины — "
                     + ", ".join(sorted(set(res["unmapped"]))) + f" ({len(res['unmapped'])})")
    if res["layer"] == "neural" and res.get("cross_check"):
        lines.append("Кросс-проверка лексиконом: " +
                     ", ".join(f"{c} {p}%" for c, p in res["cross_check"].items()))
    elif res.get("neural_unavailable"):
        if res.get("neural_released"):
            lines.append("Нейро-слой был выключен memory guard worker (освобождена память) — "
                         "числа получены лексиконом (словарём), а не моделью.")
        else:
            lines.append("Нейро-слой был запрошен, но остался недоступен — числа получены "
                         "лексиконом (словарём), а не моделью.")
    devices = {}
    for c in res["comments"]:
        if c.get("marketing"):
            devices[c["marketing"]] = devices.get(c["marketing"], 0) + 1
    if devices:
        # правило: называть конкретный приём, а не только модификатор 💸
        lines.append("Продающий язык (💸): " + str(sum(devices.values())) + " — " +
                     ", ".join(f"{k} {v}" for k, v in
                               sorted(devices.items(), key=lambda kv: -kv[1])))
    active = [t for t in res["threads"] if t["homogeneity"] is not None]
    lines.append(f"Активных тредов (≥2 содержательных): {len(active)} из {len(res['threads'])}.")
    if res["overall"] is None:
        # правило: тред с <2 содержательными — «не измеряется», а не None%
        lines.append("Однородность в треде: не измеряется "
                     "(нет тредов с ≥2 содержательными комментариями).")
    else:
        lines.append(f"Однородность в треде: в среднем по тредам {res['overall']}% "
                     f"(взвешенная {res['weighted']}%).")
    for t in res["threads"]:
        if t["homogeneity"] is None:
            lines.append(f"тред {t['size']} — не измеряется "
                         f"(нейтральный тред, содержательных {t.get('stance', 0)})")
        else:
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
