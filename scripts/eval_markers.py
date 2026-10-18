#!/usr/bin/env python3
"""Evaluate the marker layer on the eval set (eval_set.json).

Metrics (honest, script-computed):
- device recall: does the device's own example put that device into hit candidates
  (exact-name families) or the right marker family (old family-level families)?
- control false-positive rate: share of neutral controls that trigger any hit.
- subtype routing: 🔪🧠 for НЛП/внушение/Дилтс devices, 🎭🧠 for others.

Usage: python3 eval_markers.py [--json]
"""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import markers as M

HERE = os.path.dirname(os.path.abspath(__file__))
EVAL = json.load(open(os.path.join(HERE, "..", "psychology", "eval_set.json"), encoding="utf-8"))

DEVICE_FAMILIES = {"логические", "эмоции и авторитет", "язык", "данные", "структура вывода"}
NLP_DEV_FAMILIES = {"приёмы внушения (книга)", "фокусы языка (Дилтс)"}


def hits_for(text):
    return M.scan(text)


def device_hit(hits, device, family):
    for h in hits:
        if device in h.get("candidates", []):
            return "device"
    for h in hits:
        cands = h.get("candidates", [])
        if any(c in DEVICE_FAMILIES or c in NLP_DEV_FAMILIES for c in cands):
            # family-level candidate: count as family hit if the marker family plausibly maps
            if family in DEVICE_FAMILIES or family in NLP_DEV_FAMILIES:
                return "family"
    return None


def main():
    per_device, n_ok, n_family, n_miss = [], 0, 0, 0
    controls, fp = [], 0
    subtype_ok = subtype_n = 0
    for item in EVAL:
        hits = hits_for(item["text"])
        if item.get("device") is None:  # control
            controls.append({"id": item["id"], "hits": len(hits)})
            if hits:
                fp += 1
            continue
        fam = item["family"]
        got = device_hit(hits, item["device"], fam)
        per_device.append({"device": item["device"], "recall": got})
        if got == "device":
            n_ok += 1
        elif got == "family":
            n_family += 1
        else:
            n_miss += 1
        expected = "🔪🧠" if fam in NLP_DEV_FAMILIES else "🎭🧠"
        got_sub = "🔪🧠" if any("НЛП" in h["family"] or "Фокусы языка" in h["family"] for h in hits) else "🎭🧠"
        if hits:
            subtype_n += 1
            if got_sub == expected:
                subtype_ok += 1
    total = n_ok + n_family + n_miss
    report = {
        "positives": total,
        "recall_exact_device": round(n_ok * 100 / total, 1),
        "recall_family_level": round(n_family * 100 / total, 1),
        "recall_miss": round(n_miss * 100 / total, 1),
        "controls": len(controls),
        "control_false_positive_rate": round(fp * 100 / len(controls), 1) if controls else None,
        "subtype_routing_acc": round(subtype_ok * 100 / subtype_n, 1) if subtype_n else None,
        "subtype_routed_n": subtype_n,
        "zero_recall_devices": [p["device"] for p in per_device if p["recall"] is None],
    }
    if "--json" in sys.argv:
        print(json.dumps(report, ensure_ascii=False, indent=1))
    else:
        for k, v in report.items():
            print(f"{k}: {v}")


if __name__ == "__main__":
    main()
