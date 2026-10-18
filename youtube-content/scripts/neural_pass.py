#!/usr/bin/env python3
"""Neural pass for comments/transcript: emotion, toxicity, sarcasm.

Loads models WITHOUT mmap. Under memory pressure on this 16 GB machine the
default transformers path (safetensors mmap) crashes with SIGBUS inside Apple
Accelerate BLAS at the first Linear layer: macOS cannot fault in the mmap'd
weight pages. Reading the file bytes into RAM avoids the mmap entirely and the
forward pass works even when only ~200 MB is free.

Usage:
  neural_pass.py --texts file_with_one_text_per_line [--out out.json]
  neural_pass.py --text "одна фраза"
Interpreter: /Users/alex/.hermes/tools/psych-rag-env/bin/python
"""
import argparse, json, os, sys

os.environ.setdefault("OMP_NUM_THREADS", "1")
import torch
torch.set_num_threads(1)

from huggingface_hub import hf_hub_download
from safetensors.torch import load as load_bytes
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer

REPOS = {
    "emotion": "seara/rubert-base-cased-russian-emotion-detection-ru-go-emotions",
    "toxicity": "cointegrated/rubert-tiny-toxicity",
    "sarcasm": "Kostya165/rubert_tiny2_russian_emotion_sentiment",
}
LABELS = {
    "emotion": ["anger", "annoyance", "disapproval", "disgust", "disappointment",
                "sadness", "grief", "nervousness", "admiration", "joy", "amusement",
                "approval", "love", "gratitude", "optimism", "excitement", "pride",
                "relief", "caring", "desire", "neutral"],
    "toxicity": ["non-toxic", "insult", "obscenity", "threat", "dangerous"],
    "sarcasm": ["aggression", "anxiety", "neutral", "positive", "sarcasm"],
}


def load_no_mmap(repo):
    path = hf_hub_download(repo, "model.safetensors")
    cfg = AutoConfig.from_pretrained(repo)
    model = AutoModelForSequenceClassification.from_config(cfg)
    sd = load_bytes(open(path, "rb").read())          # bytes -> RAM, no mmap
    sd = {k: v for k, v in sd.items() if k != "bert.embeddings.position_ids"}
    model.load_state_dict(sd)
    return model.eval()


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--texts", help="file with one text per line")
    g.add_argument("--text", help="single text")
    ap.add_argument("--out", help="write JSON here")
    ap.add_argument("--models", default="toxicity,sarcasm",
                    help="comma list from emotion,toxicity,sarcasm")
    a = ap.parse_args()

    texts = [a.text] if a.text else [l.rstrip("\n") for l in open(a.texts) if l.strip()]
    out = []
    for name in a.models.split(","):
        repo = REPOS[name]
        tok = AutoTokenizer.from_pretrained(repo)
        model = load_no_mmap(repo)
        with torch.no_grad():
            for t in texts:
                enc = tok(t, return_tensors="pt", truncation=True, max_length=512)
                logits = model(**enc).logits[0]
                p = torch.softmax(logits, dim=-1).tolist() if name == "sarcasm" \
                    else torch.sigmoid(logits).tolist()
                top = max(zip(LABELS[name], p), key=lambda x: x[1])
                out.append({"model": name, "text": t, "top": top[0],
                            "p": round(top[1], 3),
                            "all": {k: round(v, 3) for k, v in zip(LABELS[name], p)}})
    print(json.dumps(out, ensure_ascii=False, indent=1))
    if a.out:
        json.dump(out, open(a.out, "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
