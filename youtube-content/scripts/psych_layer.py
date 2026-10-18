#!/usr/bin/env python3
"""Psychology layer: run brain-hack-analyze on a text — video transcript or user-shared content.

Calls the three layers of the sibling brain-hack-analyze skill (lexical markers →
RAG retrieval → ONNX classifier) on the given text and returns one combined
JSON. Used by the video pipeline on transcripts and by brain-hack-analyze itself
for user-shared files and long external posts. Never auto-applied to casual
chat messages.

Exit 0 always; on any problem returns {"status": "unavailable", "reason": ...}
so the video pipeline degrades to the plain ⚫ passes without breaking.

Usage:
  echo "$TRANSCRIPT" | python3 psych_layer.py
  python3 psych_layer.py --file transcript.txt
"""
import argparse, json, os, re, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
SKILL_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
HERMES = os.environ.get("HERMES_HOME") or os.path.expanduser("~/.hermes")
PY = os.environ.get("PSYCH_RAG_PY", os.path.join(HERMES, "tools", "psych-rag-env", "bin", "python"))
SCRIPTS = os.path.join(SKILL_ROOT, "scripts")


def paragraphs(text):
    """Та же единица абзаца, что у markers/classifier/plugin: строка до перевода строки.
    Раньше этот файл делил по пустной строке, и индексы абзацев не совпадали с
    классификатором (он делит по одиночному переводу строки)."""
    try:
        sys.path.insert(0, SCRIPTS)
        import markers
        return markers.paragraphs(text)
    except Exception:
        return [p for p in re.split(r"\n+", text) if p.strip()]


def call(script, args, timeout):
    try:
        out = subprocess.run([PY, os.path.join(SCRIPTS, script)] + args,
                             capture_output=True, text=True, timeout=timeout)
        if out.returncode != 0:
            return None
        return json.loads(out.stdout)
    except Exception:
        return None


# --- subtype routing: which ⚫ pair applies to a flagged paragraph ---

# Все 23 метки SemEval-2023 распределены явно; NLP-классов у классификатора нет —
# 🔪🧠 определяется маркерным слоем (Левинсон, эпистемическая дистанция) и RAG (внушение/Дилтс).
LABEL_SUBTYPE = {
    # 📢🧠 пропаганда — массовый нарратив, мобилизационная риторика
    "Flag_Waving": "📢🧠",
    "Appeal_to_Fear-Prejudice": "📢🧠",
    "Guilt_by_Association": "📢🧠",
    "Appeal_to_Hypocrisy": "📢🧠",
    "Whataboutism": "📢🧠",
    "Slogans": "📢🧠",
    "Appeal_to_Time": "📢🧠",
    "Appeal_to_Popularity": "📢🧠",
    # 🎭🧠 психологические манипуляции — логические и эмоциональные приёмы
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
NLP_FAMILY_MARKS = (  # marker-layer families = НЛП-инструментарий (Левинсон, эпистемическая дистанция, НЛП-пакеты)
    "пресуппозици", "номинализаци", "номinalизация", "эпистемическ", "нлп",
)


def marker_families_in(para_index, para_text, markers):
    """Marker families whose hit belongs to this paragraph.

    Абзац берётся из индекса hit-а (`para`), а не из подстроки: у repetition found —
    склеенная строка слов («видит, качество, цену»), которой нет ни в одном абзаце,
    и подтип по повтору абзацу не присваивался. Подстрока остаётся только fallback-ом
    для hit-ов без индекса."""
    t = (para_text or "").lower()
    fams = set()
    for h in markers:
        p = h.get("para")
        if isinstance(p, list):
            if para_index in p:
                fams.add(h.get("family", ""))
        elif p == para_index:
            fams.add(h.get("family", ""))
        elif not p and h.get("found", "").lower() in t:
            fams.add(h.get("family", ""))
    return fams


def attach_subtypes(classification, markers):
    """subtype per paragraph: classifier labels + marker families bound to the paragraph text.

    Primary = первая по приоритету (📢🧠 пропаганда → 🎭🧠 манипуляция → 🔪🧠 НЛП);
    subtype_extra = второй тип, если в абзаце смешаны (например, метка классификатора + НЛП-маркеры).
    """
    order = ["📢🧠", "🎭🧠", "🔪🧠"]
    for para in classification:
        labels = {l.get("label") for l in para.get("labels", [])}
        fams = marker_families_in(para.get("para"), para.get("text", ""), markers)
        present = []
        for s in order:
            if s == "📢🧠" and any(
                    LABEL_SUBTYPE.get(l.get("label")) == "📢🧠" for l in para.get("labels", [])):
                present.append(s)
            elif s == "🎭🧠" and any(
                    LABEL_SUBTYPE.get(l.get("label")) == "🎭🧠" for l in para.get("labels", [])):
                present.append(s)
            elif s == "🔪🧠":
                nlp = sum(1 for f in fams if any(m in f.lower() for m in NLP_FAMILY_MARKS))
                if nlp:
                    present.append(s)
        para["subtype"] = present[0] if present else None
        para["subtype_extra"] = present[1] if len(present) > 1 else None
    return classification


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", help="transcript file (default: stdin)")
    ap.add_argument("--no-gate", action="store_true",
                    help="не гейтить классификатор слоем маркеров (для сверки цифр)")
    a = ap.parse_args()
    text = open(a.file, encoding="utf-8").read() if a.file else sys.stdin.read()
    text = (text or "").strip()
    if not text:
        print(json.dumps({"status": "empty", "reason": "пустой транскрипт"}, ensure_ascii=False))
        return

    if not (os.path.isfile(PY) and os.path.isfile(os.path.join(SCRIPTS, "markers.py"))):
        print(json.dumps({"status": "unavailable",
                          "reason": f"brain-hack scripts or venv not found ({SCRIPTS})"},
                         ensure_ascii=False))
        return

    markers = call("markers.py", [text, "--json"], 30)
    hits = (markers or {}).get("markers", [])
    # гейт по правилу навыка: классификатор запускается только на абзацах, где
    # сработали маркеры. Раньше этот файл не гейтился — цифры расходились с плагином.
    gated = bool(hits) and not a.no_gate
    classifier = call("classifier.py", [text, "--json"] + (["--gated"] if gated else []), 120)
    # полный текст, кусками по 1500 символов: раньше этот файл опрашивал индекс только
    # по первым 1500, хотя плагин — по всему тексту до 4000. Замер на одном тексте:
    # полный запрос даёт «Ложная дилемма» (приём на позиции 3739), срез 1500 — нет.
    # Embedding-модель сама видит только ~512 токена, поэтому куски — единственный
    # способ достать устройства из хвоста длинного транскрипта.
    rag = call("rag_query.py", [text, "--top-k", "6", "--chunk", "1500"], 120)

    result: dict = {"status": "ok", "source": "transcript",
                    "paragraphs": len(paragraphs(text)), "gated": gated}
    if markers:
        result["markers_hits"] = markers.get("hits", 0)
        result["markers"] = markers.get("markers", [])
    if classifier and classifier.get("classification"):
        result["classification"] = attach_subtypes(
            classifier["classification"], (markers or {}).get("markers", []))
    if rag and rag.get("results"):
        result["rag"] = [{"kind": r["kind"], "label": r["label"], "score": r["score"],
                          "snippet": r["snippet"]} for r in rag["results"]]
    if "classification" not in result and "markers" not in result:
        result["status"] = "unavailable"
        result["reason"] = "слои brain-hack не ответили"
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
