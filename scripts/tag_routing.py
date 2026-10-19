"""Единый словарь маршрутизации подтипа для обеих путей анализа.

Правило навыка: ⚫ несёт ПАРУ подтипа (🔪🧠 НЛП / 🎭🧠 манипуляции / 📢🧠 пропаганда),
а не имя устройства. Раньше путь текста (psych_layer) имел свой словарь, путь
комментариев (comment_analysis) — свой, и второй никогда не выдавал пару: имя
устройства («Ложная дилемма») попадало в блок комментариев вместо подтипа.

Здесь один источник для обоих: семьи маркеров и метки классификатора. Метки —
фактическая таксономия модели (23 метки SemEval-2023 Task 3), а не вариант списка.
"""
import json
import os

SCRIPTS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PSY = os.path.join(os.path.dirname(SCRIPTS), "psychology")

ORDER = ["📢🧠", "🎭🧠", "🔪🧠"]

# Семьи маркеров: пропаганда -> 📢🧠, НЛП-инструментарий -> 🔪🧠.
# «Фокусы языка: …» (Дилтс) раньше не ловились: NLP_FAMILY_MARKS не имел «фокус»,
# и фокус Дилтса routes в 🎭🧠 вместо 🔪🧠 (правило: НЛП-инструментарий -> 🔪🧠).
NLP_FAMILY_MARKS = ("пресуппозици", "номинализаци", "эпистемическ", "нлп", "фокус")
PROPAGANDA_MARKS = ("пропаганд",)

# Метка классификатора -> подтип (23 метки, совпадает с id2label модели).
LABEL_SUBTYPE = {
    "Flag_Waving": "📢🧠",
    "Appeal_to_Fear-Prejudice": "📢🧠",
    "Guilt_by_Association": "📢🧠",
    "Appeal_to_Hypocrisy": "📢🧠",
    "Whataboutism": "📢🧠",
    "Slogans": "📢🧠",
    "Appeal_to_Time": "📢🧠",
    "Appeal_to_Popularity": "📢🧠",
    "Loaded_Language": "🎭🧠",
    "Name_Calling-Labeling": "🎭🧠",
    "Doubt": "🎭🧠",
    "Exaggeration-Minimisation": "🎭🧠",
    "Causal_Oversimplification": "🎭🧠",
    "Consequential_Oversimplification": "🎭🧠",
    "False_Dilemma-No_Choice": "🎭🧠",
    "Straw_Man": "🎭🧠",
    "Red_Herring": "🎭🧠",
    "Obfuscation-Vagueness-Confusion": "🎭🧠",
    "Repetition": "🎭🧠",
    "Conversation_Killer": "🎭🧠",
    "Appeal_to_Authority": "🎭🧠",
    "Appeal_to_Values": "🎭🧠",
    "Questioning_the_Reputation": "🎭🧠",
}

FAMILY_OF = {}


def load_devices(path=None):
    p = path or os.path.join(PSY, "devices.json")
    try:
        with open(p, encoding="utf-8") as f:
            docs = json.load(f)
    except Exception:
        return {}
    for d in docs:
        FAMILY_OF[d.get("name", "")] = d.get("family", "")
    return FAMILY_OF


def pair_for_family(family):
    low = (family or "").lower()
    if any(m in low for m in PROPAGANDA_MARKS):
        return "📢🧠"
    if any(m in low for m in NLP_FAMILY_MARKS):
        return "🔪🧠"
    return None


def pair_for_device(name, family=None):
    """Пара подтипа. Приоритет: уже пара -> семья сработавшего маркера ->
    метка классификатора по имени -> семья устройства из devices.json.
    Имя устройства без пары — незавершённая метка: возвращаем None."""
    if not name:
        return None
    if name in ORDER:
        return name
    return (pair_for_family(family) or LABEL_SUBTYPE.get(name)
            or pair_for_family(FAMILY_OF.get(name)))


def device_pairs(hits, classifier_rows):
    """Пара подтипа на абзац: семья маркера решает подтип, имя устройства —
    доказательство (правило: вывод строится из маркера + определения)."""
    load_devices()
    out = {}
    for h in hits:
        p = pair_for_family(h.get("family"))
        if p:
            out.setdefault(h.get("para"), p)
    for row in classifier_rows:
        p = LABEL_SUBTYPE.get(row.get("label"))
        if p:
            out.setdefault(row.get("para"), p)
    for h in hits:
        para = h.get("para")
        if para not in out:
            for c in h.get("candidates", []):
                p = pair_for_device(c)
                if p:
                    out[para] = p
                    break
    return out
