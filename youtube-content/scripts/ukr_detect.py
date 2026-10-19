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
import argparse, json, os, re, sys
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "scripts"))
import side  # общий боковой слой — один порог на оба пути, иначе два слоя расходятся
from util import load_json

# Токен целиком, с регистром: боковые буквы (і ї є ґ ў) должны входить в класс, иначе
# слово «быў» резалось на «бы» и белорусское доказательство терялось.
TOKEN_RE = re.compile(r"[A-Za-zА-Яа-яЁёіїєґўІЇЄҐЎ]+")

UK_LETTERS = set("іїєґ")
# буквы, которых в белорусском нет: они решают спор uk/be
UK_ONLY = set("їєґ")
# буква, которая есть только в белорусском
BE_ONLY = set("ў")

BE_WORDS = {
    # только формы, которые ОТЛИЧАЮТСЯ от русского; совпадающие с русским сюда не входят
    # сравнение — по началу слова (префикс), иначе стемы вроде «українськ», «паняцці»
    # никогда не совпадают с токеном и слой слов молча даёт нулевой recall
    "гэта", "шмат", "вельмі", "каб", "тамуж", "паняцц", "зразумеў", "зразумела", "меў",
    "аўтар", "таксама", "быў", "праўда", "нашы", "вамі", "табе", "ёсць", "хочам", "няма",
    "яны", "вялік", "ўвазе", "гаварыў", "гаварыла", "маўчаў", "беларус", "моўна",
}
# омонимы/разговорные формы, которые есть и в русском: сами язык не решают,
# считаются только вместе с сильным доказательством того же языка
WEAK = {"па", "ды", "малы", "якая", "які", "якія", "таму", "аб"}
# общие uk/be: доказательство «не русский», но язык не решают
SHARED_UK_BE = {"як", "але", "і"}
UK_WORDS = {
    # однозначные: в русском пишутся иначе И в белорусском иначе (решают спор uk/be)
    "якщо", "дуже", "вже", "ще", "тільки", "немає", "є",
    "вони", "його", "її", "їх", "був", "була", "було", "має", "мають",
    "поблизу", "навколо", "українськ", "українці", "україні",
    "росіяни", "росія", "чесно", "підміня", "найкращ",
    "отримав", "дід", "вірус", "скачува",
    "свого", "цьому", "наші", "ваші", "мої",
    # формы без букв-маркеров (ї/є/ґ), но отличающиеся и от русского, и от белорусского:
    # они решают uk сами, поэтому ложный негатив «украинское без ї/є/ґ → ru» снимается.
    # Только формы, которые ОТЛИЧАЮТСЯ от русского: стем, совпадающий с русским
    # («район», «станц», «курсу», «уходу», «електротранспорт»), добавлять нельзя —
    # префиксное сравнение ловит русское слово и даёт ложное срабатывание.
    "що", "він", "вона", "крім", "навіть", "можуть", "перший", "районі",
    "всі", "столиці", "відновлен", "пошкоджен", "декілька", "пожеж", "станці",
    # «коли», «хоча», «проте» убраны: они стоят в AMBIG и как однозначные не считаются
}
# неоднозначные (есть и в русском) — считаются только вместе с буквой-маркером
AMBIG = {"ми", "бо", "де", "так", "це", "той", "та", "нас", "вас", "їм", "тому", "коли", "хоча", "проте"}

def word_hit(tok, entries):
    """Префиксное сравнение нужно, чтобы стемы («українськ», «найкращ») ловили флексии,
    но короткие записи (≤3 символов) ловятся только как целое слово: иначе русское
    «щелей» читается как украинское «ще», а «мне надо» — как «не надо»."""
    return any(tok == e or (len(e) > 3 and tok.startswith(e)) for e in entries)


def score(text):
    """(uk_ev, be_ev, shared, ambl, weak_be) — доказательство считается ОДИН РАЗ на слово.

    Раньше одно и то же слово считалось дважды (как буква-маркер и как слово из
    словаря), из-за чего одиночное «є» давало два доказательства и решало язык.
    Буквы считаются по слову, а не по символу: слово с тремя marker-буквами не
    должно перевешивать несколько слов.

    Имя собственное язык не решает: токен с заглавной буквы исключается из
    доказательства. «Євгененко вышел из дома» — единственное доказательство было
    фамилия в украинской орфографии, и 🐷 за неё ставиться не должно. Цена правила:
    украинское слово в начале предложения тоже читается как имя и доказательства не
    даёт — это осознанный обмен, правило пользователя важнее.
    """
    uk_ev = be_ev = shared = ambl = weak_be = 0
    for tok in TOKEN_RE.findall(text):
        if tok[:1].isupper():
            continue                      # имя собственное / начало предложения
        w = tok.lower()
        e_uk = any(ch in UK_ONLY for ch in w) or word_hit(w, UK_WORDS)
        e_be = any(ch in BE_ONLY for ch in w) or word_hit(w, BE_WORDS)
        if e_uk and e_be:
            shared += 1                      # токен доказывает обоим — язык не решает
        elif e_uk:
            uk_ev += 1
        elif e_be:
            be_ev += 1
        elif w in SHARED_UK_BE:
            shared += 1
        elif w in AMBIG:
            ambl += 1                        # неоднозначное: само язык не решает
        elif w in WEAK:
            weak_be += 1
    return uk_ev, be_ev, shared, ambl, weak_be


def lang_of(text, min_ev=1):
    """uk / be / amb / ru. Общая 'і' и общие слова (як, але) сами язык не решают.

    Одно РЕШАЮЩЕЕ доказательство решает язык: буквы ї, є, ґ нет в белорусском и
    русском, буква ў нет в украинском — одно слово с такой буквой уже определяет
    язык. Но доказательство считается один раз на слово (не дважды: буква + словарь),
    и омонимы с русским (па, ды, якая, тому…) язык не решают: они только усиливают
    уже решённый язык. Русское «па был в доме» не должно становиться белорусским.

    AMBIG участвует в решении: без маркера неоднозначные слова язык не решают, но текст
    уже не русский — «це ми бачили де так» раньше молча уходило в ru.
    WEAK участвует как усиление: одна омонимичная форма язык не решает, но при точной
    ничье решающих доказательств две формы одного языка решают спор.

    Фрагмент из ≤3 слов язык не решает — кроме случая, когда весь фрагмент состоит из
    боковых слов (доля = 1.0, это язык текста, а не вставка). Порог — side.SIDE_THRESHOLD,
    общий для обоих слоёв: раньше lang_of решал по числу доказательств, а side — по доле,
    и на одном фрагменте они давали разные вердикты.
    """
    uk_ev, be_ev, shared, ambl, weak_be = score(text)
    if uk_ev == 0 and be_ev == 0:
        # Два слоя обязаны совпадать в классе: если боковой слой (графика) говорит, что
        # весь фрагмент — мова, морфология не имеет права называть его русским. Конкретный
        # язык она решает только по решающей букве: «маніпуляцыя» — мова, но uk или be —
        # не решаем, потому что общая 'і' сама язык не решает.
        if side.is_side_language(text):
            return "amb"
        return "amb" if (shared or ambl) else "ru"
    words = [t for t in TOKEN_RE.findall(text) if not t[:1].isupper()]
    if len(words) <= 3 and side.side_share(text) < side.SIDE_THRESHOLD:
        return "amb"
    if be_ev > uk_ev:
        return "be" if be_ev >= min_ev else "amb"
    if uk_ev > be_ev:
        return "uk" if uk_ev >= min_ev else "amb"
    # точная ничья решающих доказательств: омонимы сами язык не решают, но если
    # их несколько и они склоняются к одному языку — решают спор
    if weak_be >= 2:
        return "be"
    return "amb"


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
    """Правило навыка: >70% — это доля ТЕКСТА, а не доля комментариев.
    Раньше share считался по числу items: один короткий украинский комментарий
    среди длинных русских давал 0.5, хотя по объёму текста там было 2%.
    Возвращаю обе доли: share_chars (правило) и share_items (для сверки).

    amb — не язык и в долю не-русского не входит: он означает «доказательство есть,
    язык не решён». Раньше amb считался, и один спорный комментарий поднимал долю."""
    DECIDED = ("uk", "be", "mixed")
    langs, chars, total = Counter(), 0, 0
    for t in items:
        l = detect_item(t)[0]
        langs[l] += 1
        total += len(t)
        if l in DECIDED:
            chars += len(t)
    share_chars = chars / total if total else 0
    share_items = sum(v for k, v in langs.items() if k in DECIDED) / sum(langs.values()) if langs else 0
    return langs, share_chars, share_items

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--comments", help="flat json (author/text/parent)")
    ap.add_argument("--transcript", help="текст транскрипта (строки с таймкодами или plain)")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()

    out = {}
    if a.comments:
        items = load_json(a.comments)
        texts = [c["text"] for c in items]
        langs, share_chars, share_items = majority(texts)
        ins = []
        for i, c in enumerate(items):
            lang, segs = detect_item(c["text"])
            if lang in ("uk", "be", "mixed", "amb") and segs:
                # 🐷 только за украинский; белорусский и спорный случай — без метки
                tag = "🐷" if any(sl == "uk" for sl, _ in segs) else None
                ins.append({"i": i, "author": c.get("author"), "lang": lang, "tag": tag,
                            "segments": segs, "text": c["text"][:200]})
        out["comments"] = {"n": len(items), "langs": dict(langs),
                           "nonru_share_chars": round(share_chars, 3),
                           "nonru_share_items": round(share_items, 3),
                           "insertions": ins}
    if a.transcript:
        with open(a.transcript, encoding="utf-8") as fh:
            raw = fh.read()
        # строки вида "0:12 текст" — группируем по абзацам-строкам
        paras = [l for l in raw.splitlines() if l.strip()]
        texts = [re.sub(r"^\d+:\d+(:\d+)?\s*", "", l) for l in paras]
        langs, share_chars, share_items = majority(texts)
        ins = []
        for line, t in zip(paras, texts):
            lang, segs = detect_item(t)
            if lang in ("uk", "be", "mixed", "amb") and segs:
                ts = re.match(r"^(\d+:\d+(?::\d+)?)", line)
                tag = "🐷" if any(sl == "uk" for sl, _ in segs) else None
                ins.append({"t": ts.group(1) if ts else None, "lang": lang, "tag": tag,
                            "segments": segs, "text": t[:200]})
        out["transcript"] = {"n": len(texts), "langs": dict(langs),
                             "nonru_share_chars": round(share_chars, 3),
                             "nonru_share_items": round(share_items, 3),
                             "insertions": ins}
    print(json.dumps(out, ensure_ascii=False, indent=1) if a.json else json.dumps(out, ensure_ascii=False))

if __name__ == "__main__":
    main()
