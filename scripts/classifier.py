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
MODELS = os.path.join(os.path.dirname(HERE), "psychology", "models")
ONNX = os.path.join(MODELS, "classifier_int8.onnx")
TOKENIZER = os.path.join(MODELS, "tokenizer.json")
STATE = os.path.join(MODELS, "update_state.json")
LABELS = [l.strip() for l in open(os.path.join(MODELS, "labels.txt"), encoding="utf-8") if l.strip()]
THRESH = np.load(os.path.join(MODELS, "thresholds.npy"))
TAXONOMY = json.load(open(os.path.join(os.path.dirname(HERE), "psychology", "taxonomy_map.json"), encoding="utf-8"))["labels"]
MAX_TOKENS = 256

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
    tok = tokenizer()
    enc = tok.encode(text, add_special_tokens=True)
    # tokenizers.Encoding is immutable — truncate into locals (Encoding.ids is not writable)
    ids_l = enc.ids[:MAX_TOKENS]
    mask_l = enc.attention_mask[:MAX_TOKENS]
    ids = np.array([ids_l], dtype=np.int64)
    mask = np.array([mask_l], dtype=np.int64)
    logits = session().run(None, {"input_ids": ids, "attention_mask": mask})[0][0]
    p = 1 / (1 + np.exp(-logits))
    thr = THRESH if min_conf is None else np.maximum(THRESH, min_conf)
    out = []
    for i, v in enumerate(p):
        if v >= thr[i] and LABELS[i] != "Other":
            out.append({"label": LABELS[i], "p": round(float(v), 3),
                        "threshold": round(float(thr[i]), 3),
                        "ru": TAXONOMY.get(LABELS[i], {}).get("ru", LABELS[i]),
                        "device": TAXONOMY.get(LABELS[i], {}).get("device", "")})
    out.sort(key=lambda x: -x["p"])
    return out

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
    paras = [p for p in text.split("\n") if p.strip()]
    gated = "--gated" in sys.argv
    if gated:
        import markers
        hits = markers.scan(text)
        # map marker hits to paragraph index by locating the found word
        keep = set()
        for h in hits:
            for i, p in enumerate(paras, 1):
                if h["found"].lower() in p.lower():
                    keep.add(i)
        paras = [(i, p) for i, p in enumerate(paras, 1) if i in keep]
    else:
        paras = list(enumerate(paras, 1))
    rows = []
    t1 = time.time()
    for i, p in paras:
        hits = classify(p, min_conf)
        if hits:
            rows.append({"para": i, "text": p[:80], "labels": hits})
    call_ms = round((time.time() - t1) * 1000)
    if "--json" in sys.argv:
        upd = None
        try:
            upd = json.load(open(STATE, encoding="utf-8"))
        except Exception:
            upd = None
        ver = None
        try:
            ver = json.load(open(os.path.join(MODELS, "model_version.json"), encoding="utf-8"))
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
