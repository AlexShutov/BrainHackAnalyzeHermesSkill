#!/usr/bin/env python3
"""Background update worker: download + convert the candidate model into staging,
verify it against the model in use, and mark the state. Never swaps models.

Run by model_update_check.py; safe to run manually as well.
"""
import json, os, subprocess, sys, datetime
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SKILL = os.path.dirname(HERE)
MODELS = os.path.join(SKILL, "psychology", "models")
STAGING = os.path.join(MODELS, "staging")
STATE = os.path.join(MODELS, "update_state.json")

CONTROL = [
    "Либо ты поддерживаешь этот курс, либо ты предатель.",
    "Эти ребята опять всё испортили. Я знаю, что ты не такой.",
    "По данным Росстата, в 2025 году инфляция в России составила 4,1 процента.",
]

# Пороги — свойство конкретной модели, а не общее. Если у кандидата своих порогов
# нет, их нельзя занимать у живой модели: сравниваем по фиксированной рабочей точке
# (гейт навыка — 0.70) и помечаем, что калибровка обязательна перед продвижением.
OPERATING_POINT = 0.70

def save_state(s):
    with open(STATE, "w", encoding="utf-8") as fh:
        json.dump(s, fh, ensure_ascii=False, indent=2)

def model_files(base):
    """Своя комплектация модели: labels и пороги берутся из её же каталога."""
    labels_path = os.path.join(base, "labels.txt")
    thr_path = os.path.join(base, "thresholds.npy")
    labels = [l.strip() for l in open(labels_path, encoding="utf-8") if l.strip()]
    thr = None
    if os.path.exists(thr_path):
        thr = np.load(thr_path)
        if len(thr) != len(labels):
            raise ValueError(f"порогов {len(thr)} на модели, а меток {len(labels)}: {base}")
    return labels, thr

def labels_of(session, tokenizer, text, labels, thr):
    enc = tokenizer.encode(text, add_special_tokens=True)
    ids = np.array([enc.ids], dtype=np.int64)
    mask = np.array([enc.attention_mask], dtype=np.int64)
    logits = session.run(None, {"input_ids": ids, "attention_mask": mask})[0][0]
    p = 1 / (1 + np.exp(-logits))
    cut = thr if thr is not None else [OPERATING_POINT] * len(labels)
    return {labels[i] for i in range(len(p)) if p[i] >= cut[i]}

def main():
    st = json.load(open(STATE, encoding="utf-8"))
    cand = st["candidate"]
    os.makedirs(STAGING, exist_ok=True)
    r = subprocess.run([sys.executable, os.path.join(HERE, "model_setup.py"),
                        "--model-id", cand["model_id"], "--revision", cand.get("revision") or "",
                        "--out-dir", STAGING, "--quiet"], capture_output=True, text=True, timeout=1800)
    if r.returncode != 0:
        st["status"] = "failed"
        st["error"] = (r.stderr or r.stdout).strip()[:400]
        save_state(st)
        print(json.dumps({"status": "failed", "error": st["error"]}, ensure_ascii=False))
        return 0

    try:
        import onnxruntime as ort
        from tokenizers import Tokenizer
        live = ort.InferenceSession(os.path.join(MODELS, "classifier_int8.onnx"),
                                    providers=["CPUExecutionProvider"])
        new = ort.InferenceSession(os.path.join(STAGING, "classifier_int8.onnx"),
                                   providers=["CPUExecutionProvider"])
        tok_live = Tokenizer.from_file(os.path.join(MODELS, "tokenizer.json"))
        tok_new = Tokenizer.from_file(os.path.join(STAGING, "tokenizer.json"))
        labels_live, thr_live = model_files(MODELS)
        labels_new, thr_new = model_files(STAGING)
        src_live = "thresholds.npy живой модели" if thr_live is not None else f"рабочая точка {OPERATING_POINT}"
        src_new = ("thresholds.npy кандидата" if thr_new is not None
                   else f"рабочая точка {OPERATING_POINT} — своих порогов у кандидата нет, калибровка обязательна")
        labels_match = labels_live == labels_new
        agree = 0
        detail = []
        for text in CONTROL:
            a = labels_of(live, tok_live, text, labels_live, thr_live)
            b = labels_of(new, tok_new, text, labels_new, thr_new)
            detail.append({"text": text[:60], "in_use": sorted(a), "new": sorted(b), "same": a == b})
            agree += int(a == b)
        st["status"] = "ready"
        st["verified_at"] = datetime.datetime.now().isoformat(timespec="seconds")
        st["agreement"] = f"{agree}/{len(CONTROL)}"
        st["detail"] = detail
        st["staging"] = STAGING
        st["thresholds"] = {"in_use": src_live, "new": src_new}
        st["labels_match"] = labels_match
        st["needs_calibration"] = thr_new is None
        if not labels_match:
            st["note"] = ("наборы меток разные — совпадение меток по имени некорректно, "
                          "нужен перенос через taxonomy_map.json")
        save_state(st)
        print(json.dumps({"status": "ready", "agreement": st["agreement"],
                          "thresholds": st["thresholds"], "labels_match": labels_match,
                          "needs_calibration": st["needs_calibration"],
                          "note": "новая модель проверена, но не установлена — используется старая"},
                         ensure_ascii=False))
    except Exception as e:
        st["status"] = "downloaded_unverified"
        st["error"] = str(e)[:300]
        save_state(st)
        print(json.dumps({"status": "downloaded_unverified", "error": st["error"]}, ensure_ascii=False))
    return 0

if __name__ == "__main__":
    sys.exit(main())
