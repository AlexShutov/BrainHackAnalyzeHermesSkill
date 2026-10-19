#!/usr/bin/env python3
"""Слой 3: независимый классификатор приёмов (второе мнение).

Модель: Mishtar4/persuasion-detector-xlm-roberta (XLM-RoBERTa-large, 23 метки
SemEval-2023 Task 3, мультиязычная, работает на русском напрямую — перевод
не нужен: XLM-R обучен на 100 языках, русский входит в таксономию модели).
Запускается через ONNX Runtime int8 — torch в рантайме не нужен.

Пороги взяты из thresholds.npy самого репозитория (настраивались под dev-набор).
Метки переводятся на наши устройства через psychology/taxonomy_map.json.

Использование:
  python3 classifier.py "текст" [--json] [--min 0.5]
"""
import json, os, sys, time
import numpy as np
from tokenizers import Tokenizer
import onnxruntime as ort

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import markers  # каноническое определение абзаца для всех слоёв
from util import load_json, read_lines
MODELS = os.path.join(os.path.dirname(HERE), "psychology", "models")
ONNX = os.path.join(MODELS, "classifier_int8.onnx")
TOKENIZER = os.path.join(MODELS, "tokenizer.json")
STATE = os.path.join(MODELS, "update_state.json")
LABELS = [l.strip() for l in read_lines(os.path.join(MODELS, "labels.txt"))]
THRESH = np.load(os.path.join(MODELS, "thresholds.npy"))
with open(os.path.join(os.path.dirname(HERE), "psychology", "taxonomy_map.json"), encoding="utf-8") as _f:
    TAXONOMY = json.load(_f)["labels"]
MAX_TOKENS = 256
# Куски перекрываются: приём, начавшийся в последних строках куска и закончившийся в
# начале следующего, терялся на стыке — модель видит только окно 256 токенов.
OVERLAP = 32

def chunk_starts(n_tokens):
    """Начала кусков с перекрытием. Без перекрытия шаг = MAX_TOKENS, и приём на стыке
    двух кусков не попадает ни в одно окно."""
    if n_tokens <= MAX_TOKENS:
        return [0]
    step = MAX_TOKENS - OVERLAP
    return list(range(0, n_tokens, step))

_sess = None
_tok = None

def session():
    global _sess
    if _sess is None:
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        _sess = ort.InferenceSession(ONNX, so, providers=["CPUExecutionProvider"])
    return _sess

def tokenizer():
    global _tok
    if _tok is None:
        _tok = Tokenizer.from_file(TOKENIZER)
    return _tok

def classify(text, min_conf=None):
    """Классификация абзаца. Абзац длиннее MAX_TOKENS режется на куски, а не отбрасывается.

    Раньше `enc.ids[:MAX_TOKENS]` молча резал: приём после 256 токенов был невидим
    (замер: абзац 2550 токенов с подменой в конце -> пустой результат, тот же приём
    отдельным абзацем -> Doubt 0.704). Теперь каждый кусок проходит отдельно, и по
    каждому метке берётся максимальная вероятность между кусками — порог thresholds.npy
    применяется к этому максимуму.
    """
    tok = tokenizer()
    enc = tok.encode(text, add_special_tokens=True)
    ids_all, mask_all = enc.ids, enc.attention_mask
    agg = {}
    for ci, i in enumerate(chunk_starts(len(ids_all)), 1):
        chunk = ids_all[i:i + MAX_TOKENS]
        # attention_mask — по фактическому срезу (у хвостового куска он короче), а не
        # с начала текста: при перекрытии кусок начинается не с нуля
        m = mask_all[i:i + len(chunk)]
        logits = session().run(None, {"input_ids": np.array([chunk], dtype=np.int64),
                                      "attention_mask": np.array([m], dtype=np.int64)})[0][0]
        p = 1 / (1 + np.exp(-logits))
        for i, v in enumerate(p):
            if LABELS[i] == "Other":
                continue
            if v > agg.get(i, (0.0, 0))[0]:
                agg[i] = (float(v), ci)
    thr = THRESH if min_conf is None else np.maximum(THRESH, min_conf)
    out = [{"label": LABELS[i], "p": round(v, 3), "threshold": round(float(thr[i]), 3),
            "chunk": c,
            "ru": TAXONOMY.get(LABELS[i], {}).get("ru", LABELS[i]),
            "device": TAXONOMY.get(LABELS[i], {}).get("device", "")}
           for i, (v, c) in agg.items() if v >= thr[i]]
    out.sort(key=lambda x: -x["p"])
    return out

def gated_paragraphs(text):
    """Абзацы, где маркеры сработали. Одна логика для CLI и worker — иначе два пути
    расходятся: у repetition found склеенная строка слов, которой нет ни в одном абзаце,
    и абзац, помеченный только повтором, должен доходить до модели."""
    paras = markers.paragraphs(text)
    hits = markers.scan(text)
    keep = set()
    for h in hits:
        pi = h.get("para")
        if isinstance(pi, list):
            keep.update(pi)
        elif pi:
            keep.add(pi)
        else:
            for i, pp in enumerate(paras, 1):
                if h["found"].lower() in pp.lower():
                    keep.add(i)
    return [(i, pp) for i, pp in enumerate(paras, 1) if i in keep]


def main():
    if len(sys.argv) < 2:
        print("usage: classifier.py <text> [--json] [--min 0.5] [--gated]", file=sys.stderr); return 1
    text = sys.argv[1]
    min_conf = None
    if "--min" in sys.argv:
        idx = sys.argv.index("--min") + 1
        if idx >= len(sys.argv):
            print("--min требует значения (например --min 0.5)", file=sys.stderr)
            return 1
        try:
            min_conf = float(sys.argv[idx])
        except ValueError:
            print(f"--min: не число «{sys.argv[idx]}»", file=sys.stderr)
            return 1
    t0 = time.time()
    session()
    load_ms = round((time.time() - t0) * 1000)
    gated = "--gated" in sys.argv
    paras = gated_paragraphs(text) if gated else list(enumerate(markers.paragraphs(text), 1))
    rows = []
    t1 = time.time()
    for i, p in paras:
        hits = classify(p, min_conf)
        if hits:
            # полная строка абзаца: JSON-потребитель (psych_layer) сопоставляет маркеры
            # по тексту абзаца, и маркер после 80-го символа иначе терялся
            rows.append({"para": i, "text": p, "labels": hits})
    call_ms = round((time.time() - t1) * 1000)
    if "--json" in sys.argv:
        upd = None
        try:
            upd = load_json(STATE)
        except Exception:
            upd = None
        ver = None
        try:
            ver = load_json(os.path.join(MODELS, "model_version.json"))
        except Exception:
            ver = None
        print(json.dumps({"model_load_ms": load_ms, "call_ms": call_ms,
                          "paragraphs": len(paras), "gated": gated,
                          "model": (ver or {}).get("model_id"),
                          "update": (upd or {}).get("status"),
                          "classification": rows}, ensure_ascii=False, indent=1))
    else:
        print(f"load {load_ms} ms | call {call_ms} ms | {len(paras)} абзацев")
        for r in rows:
            print(f"абзац {r['para']}: " + ", ".join(f"{h['ru']} ({h['p']})" for h in r["labels"]))
    return 0

if __name__ == "__main__":
    sys.exit(main())
