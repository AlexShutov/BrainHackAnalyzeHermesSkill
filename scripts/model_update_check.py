#!/usr/bin/env python3
"""Version check for the classifier model — run once per session, on first use.

Compares the stored version (psychology/models/model_version.json) with what the
hub reports, and looks for similar models with a better reported metric that
cover Russian. If an update is warranted, marks it as needed and starts the
download in the background; the old model keeps serving until the new one is
verified. Never swaps models on its own.

Usage:
  python3 model_update_check.py [--force] [--json]

Exit 0 always (the check must not break the analysis path).
"""
import argparse, json, os, subprocess, sys, datetime

HERE = os.path.dirname(os.path.abspath(__file__))
SKILL = os.path.dirname(HERE)
MODELS = os.path.join(SKILL, "psychology", "models")
STATE = os.path.join(MODELS, "update_state.json")
STAGING = os.path.join(MODELS, "staging")

def load_version():
    try:
        return json.load(open(os.path.join(MODELS, "model_version.json"), encoding="utf-8"))
    except Exception:
        return None

def load_state():
    try:
        return json.load(open(STATE, encoding="utf-8"))
    except Exception:
        return None

def save_state(s):
    os.makedirs(MODELS, exist_ok=True)
    with open(STATE, "w", encoding="utf-8") as fh:
        json.dump(s, fh, ensure_ascii=False, indent=2)

def hub_get(url):
    try:
        import requests
        r = requests.get(url, timeout=15)
        if r.status_code != 200:
            return None
        return r.json()
    except Exception:
        return None

def best_metric(model):
    for entry in model.get("model-index") or []:
        for res in entry.get("results") or []:
            for m in res.get("metrics") or []:
                if m.get("type") == "f1":
                    return float(m.get("value") or 0), m.get("name", "f1")
    return None, None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="decide but do not start the download")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()

    v = load_version()
    if not v:
        print(json.dumps({"status": "no_version", "note": "model_version.json missing — run model_setup.py"},
                         ensure_ascii=False))
        return 0

    st = load_state()
    if st and st.get("status") in ("updating", "ready") and not a.force:
        print(json.dumps({"status": st["status"], "state": st}, ensure_ascii=False))
        return 0

    current = hub_get(f"https://huggingface.co/api/models/{v['model_id']}")
    if current is None:
        print(json.dumps({"status": "offline", "note": "hub unreachable — keeping the current model"},
                         ensure_ascii=False))
        return 0

    reasons = []
    candidate = None
    if current.get("sha") and current.get("sha") != v.get("revision"):
        reasons.append(f"newer revision of {v['model_id']}: {current['sha'][:12]} vs stored {str(v.get('revision'))[:12]}")
        candidate = {"model_id": v["model_id"], "revision": current["sha"], "kind": "revision"}

    # similar models: same task, must cover Russian, better reported metric
    cur_f1, cur_name = best_metric(current)
    found = hub_get("https://huggingface.co/api/models?search=persuasion&pipeline_tag=text-classification&limit=20") or []
    for m in found:
        if m["id"] == v["model_id"]:
            continue
        langs = set(t for t in m.get("tags", []) if len(t) == 2)
        if "ru" not in langs:
            continue
        f1, name = best_metric(m)
        if f1 and cur_f1 and f1 > cur_f1:
            candidate = {"model_id": m["id"], "revision": m.get("sha"), "kind": "better model",
                         "f1": f1, "f1_name": name}
            reasons.append(f"better reported metric: {m['id']} {f1:.2f} ({name}) vs {cur_f1:.2f} ({cur_name})")
            break

    if not reasons:
        print(json.dumps({"status": "current", "model": v["model_id"], "revision": v.get("revision"),
                          "f1": cur_f1}, ensure_ascii=False))
        return 0

    state = {"status": "needed", "reasons": reasons, "candidate": candidate,
            "in_use": {"model_id": v["model_id"], "revision": v.get("revision")},
            "marked_at": datetime.datetime.now().isoformat(timespec="seconds")}
    save_state(state)

    # background download; the runtime keeps using the old model
    os.makedirs(STAGING, exist_ok=True)
    py = sys.executable
    proc = None
    if not a.dry_run:
        proc = subprocess.Popen([py, os.path.join(HERE, "model_update_run.py")],
                                stdout=open(os.path.join(STAGING, "update.log"), "w"),
                                stderr=subprocess.STDOUT)
    state["status"] = "updating" if proc else "needed"
    if proc:
        state["pid"] = proc.pid
    state["staging"] = STAGING
    save_state(state)

    if proc:
        announce = ("Начато обновление психологической модели (фон): " + "; ".join(reasons) +
                    ". Текущая модель продолжает использоваться, пока новая не проверена.")
    else:
        announce = ("Помечено как нужное к обновлению: " + "; ".join(reasons) +
                    ". Продолжается использование текущей модели.")
    out = {"status": state["status"], "state": state, "announcement": announce}
    print(json.dumps(out, ensure_ascii=False))
    return 0

if __name__ == "__main__":
    sys.exit(main())
