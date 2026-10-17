#!/usr/bin/env python3
"""Download classifier weights and convert them to the runtime format.

Runtime format = ONNX int8 (executed by onnxruntime, no torch at runtime).
torch is needed only here, at conversion time; if a pre-exported ONNX exists in
the repo it is downloaded instead and no conversion happens.

Usage:
  python3 model_setup.py [--model-id ID] [--revision SHA] [--out-dir DIR] [--no-quantize]

Writes model_version.json into the out-dir (model id, revision, format, date).
"""
import argparse, json, os, sys, datetime

DEFAULT_MODEL = "Mishtar4/persuasion-detector-xlm-roberta"

def hermes_home():
    return os.environ.get("HERMES_HOME") or os.path.expanduser("~/.hermes")

def skill_dir():
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def out_dir_default():
    return os.path.join(skill_dir(), "psychology", "models")

def have(mod):
    try:
        __import__(mod); return True
    except ImportError:
        return False

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", default=DEFAULT_MODEL)
    ap.add_argument("--revision", default=None)
    ap.add_argument("--out-dir", default=out_dir_default())
    ap.add_argument("--quantize", action="store_true", default=True)
    ap.add_argument("--no-quantize", action="store_true")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args()
    out = a.out_dir
    os.makedirs(out, exist_ok=True)

    from huggingface_hub import snapshot_download
    snap = snapshot_download(a.model_id, revision=a.revision)
    files = {f for f in os.listdir(snap)}

    def grab(name):
        src = os.path.join(snap, name)
        if os.path.exists(src):
            import shutil
            shutil.copy(src, os.path.join(out, name))

    # prefer a pre-exported ONNX: no torch needed
    onnx_name = None
    for cand in ("model.onnx", "model_fp16.onnx"):
        if cand in files:
            onnx_name = cand
            break

    if onnx_name:
        import shutil
        shutil.copy(os.path.join(snap, onnx_name), os.path.join(out, "classifier.onnx"))
        converted = "downloaded ONNX"
    else:
        if not have("torch"):
            print("torch is required for ONNX export at conversion time "
                  "(pip install torch in the venv, or use a model that ships an ONNX export)",
                  file=sys.stderr)
            return 2
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        tok = AutoTokenizer.from_pretrained(a.model_id)
        model = AutoModelForSequenceClassification.from_pretrained(a.model_id)
        model.eval()
        d = tok("conversion probe", return_tensors="pt")
        target = os.path.join(out, "classifier.onnx")
        torch.onnx.export(model, (d["input_ids"], d["attention_mask"]), target,
                          opset_version=14, input_names=["input_ids", "attention_mask"],
                          output_names=["logits"],
                          dynamic_axes={"input_ids": {1: "seq"}, "attention_mask": {1: "seq"}},
                          dynamo=False, external_data=False)
        converted = "exported from weights"

    if a.quantize and not a.no_quantize:
        from onnxruntime.quantization import quantize_dynamic, QuantType
        raw = os.path.join(out, "classifier.onnx")
        q = os.path.join(out, "classifier_int8.onnx")
        quantize_dynamic(raw, q, weight_type=QuantType.QInt8)
        # the fp32 graph and any external-data leftovers are not needed after quantization
        for f in os.listdir(out):
            if f != os.path.basename(q):
                os.remove(os.path.join(out, f))
        final = q
    else:
        final = os.path.join(out, "classifier.onnx")

    for f in ("tokenizer.json", "labels.txt", "thresholds.npy", "config.json"):
        grab(f)

    info = {
        "model_id": a.model_id,
        "revision": a.revision or _sha(a.model_id) or "",
        "resolved_sha": _sha(a.model_id),
        "format": "onnx-int8" if a.quantize and not a.no_quantize else "onnx",
        "runtime": "onnxruntime (CPUExecutionProvider)",
        "converted": converted,
        "files": sorted(os.listdir(out)),
        "setup_at": datetime.datetime.now().isoformat(timespec="seconds"),
    }
    with open(os.path.join(out, "model_version.json"), "w", encoding="utf-8") as fh:
        json.dump(info, fh, ensure_ascii=False, indent=2)
    if not a.quiet:
        print(json.dumps(info, ensure_ascii=False, indent=2))
    return 0

def _sha(model_id):
    try:
        import requests
        r = requests.get(f"https://huggingface.co/api/models/{model_id}", timeout=15)
        return r.json().get("sha", "")
    except Exception:
        return ""

if __name__ == "__main__":
    sys.exit(main())
