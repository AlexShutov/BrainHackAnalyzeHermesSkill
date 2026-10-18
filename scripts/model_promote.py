#!/usr/bin/env python3
"""Promote a verified staging model to the runtime path. Explicit, never automatic.

The update worker only marks the state as ready; the model in use keeps serving
until this script is run. Refuses to promote an unverified or failed update.
"""
import json, os, shutil, sys
from util import load_json, read_bytes, read_lines

HERE = os.path.dirname(os.path.abspath(__file__))
SKILL = os.path.dirname(HERE)
MODELS = os.path.join(SKILL, "psychology", "models")
STATE = os.path.join(MODELS, "update_state.json")

def main():
    try:
        st = load_json(STATE)
    except Exception:
        print("нет состояния обновления — сначала run model_update_check.py", file=sys.stderr)
        return 1
    if st.get("status") != "ready":
        print(f"нельзя продвигать состояние '{st.get('status')}' — нужна успешная проверка", file=sys.stderr)
        return 1
    staging = st["staging"]
    src = os.path.join(staging, "classifier_int8.onnx")
    if not os.path.exists(src):
        print("в staging нет модели", file=sys.stderr)
        return 1

    # Пороги — свойство модели. Продвигать модель без своих порогов нельзя:
    # иначе останутся пороги прежней модели, посчитанные на её логитах.
    force = "--force" in sys.argv
    new_thr = os.path.join(staging, "thresholds.npy")
    live_thr = os.path.join(MODELS, "thresholds.npy")
    if not os.path.exists(new_thr):
        if not force:
            print("у кандидата нет своих порогов — нужна калибровка (или --force)", file=sys.stderr)
            return 1
    else:
        import hashlib
        h_new = hashlib.sha256(read_bytes(new_thr)).hexdigest()
        h_live = hashlib.sha256(read_bytes(live_thr)).hexdigest() if os.path.exists(live_thr) else ""
        if h_new == h_live and not force:
            print("пороги кандидата идентичны порогам живой модели — это пороги другой модели "
                  "(или --force)", file=sys.stderr)
            return 1
        try:
            import numpy as np
            thr = np.load(new_thr)
            labels = [l.strip() for l in read_lines(os.path.join(staging, "labels.txt"))]
            if len(thr) != len(labels):
                print(f"порогов {len(thr)}, меток {len(labels)} — не совпадает", file=sys.stderr)
                return 1
        except ImportError:
            pass

    shutil.copy(src, os.path.join(MODELS, "classifier_int8.onnx"))
    for f in ("tokenizer.json", "labels.txt", "thresholds.npy", "config.json", "model_version.json"):
        if os.path.exists(os.path.join(staging, f)):
            shutil.copy(os.path.join(staging, f), os.path.join(MODELS, f))
    st["status"] = "promoted"
    st["promoted_at"] = __import__("datetime").datetime.now().isoformat(timespec="seconds")
    json.dump(st, open(STATE, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(json.dumps({"status": "promoted", "model": st["candidate"]["model_id"],
                      "agreement": st.get("agreement")}, ensure_ascii=False))
    return 0

if __name__ == "__main__":
    sys.exit(main())
