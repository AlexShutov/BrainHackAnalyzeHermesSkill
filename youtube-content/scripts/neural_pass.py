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
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "scripts"))
from util import read_bytes, read_lines, write_json

REPOS = {
    "emotion": "seara/rubert-base-cased-russian-emotion-detection-ru-go-emotions",
    "toxicity": "cointegrated/rubert-tiny-toxicity",
    "sarcasm": "Kostya165/rubert_tiny2_russian_emotion_sentiment",
}
LABELS = {
    # порядок ДОЛЖЕН совпадать с id2label модели — берём из config, не из этого списка.
    # Здесь только fallback для оффлайн-режима без config.json.
    "emotion": ["admiration", "amusement", "anger", "annoyance", "approval", "caring",
                "confusion", "curiosity", "desire", "disappointment", "disapproval",
                "disgust", "embarrassment", "excitement", "fear", "gratitude", "grief",
                "joy", "love", "nervousness", "optimism", "pride", "realization",
                "relief", "remorse", "sadness", "surprise", "neutral"],
    "toxicity": ["non-toxic", "insult", "obscenity", "threat", "dangerous"],
    "sarcasm": ["aggression", "anxiety", "neutral", "positive", "sarcasm"],
}
# multi-label модели — sigmoid; одиночный класс — softmax
MULTILABEL = {"emotion": True, "toxicity": True, "sarcasm": False}

# Правило: нейро-проход идёт пачками, а не по одному комментату — на нескольких
# сотнях комментариев поштучный forward ~10× медленнее.
BATCH = 32


def _probs(row, multilabel):
    t = row if hasattr(row, "sigmoid") else torch.tensor(row, dtype=torch.float32)
    return (torch.sigmoid(t) if multilabel else torch.softmax(t, dim=-1)).tolist()


def _enc_pair(enc):
    """BatchEncoding — Mapping, а не dict: `isinstance(enc, dict)` даёт False и путь
    падает на `enc.ids`. Пробую индексирование, иначе атрибуты; к тензору приводим
    только если это ещё не тензор (реальный токенаizer уже возвращает тензоры)."""
    try:
        ids, mask = enc["input_ids"], enc["attention_mask"]
    except (TypeError, KeyError, AttributeError):
        ids, mask = enc.ids, enc.attention_mask
    if not hasattr(ids, "ndim"):
        ids, mask = torch.tensor(ids, dtype=torch.long), torch.tensor(mask, dtype=torch.long)
    return ids, mask


def infer(tok, model, labels, texts, multilabel, name, batch=BATCH):
    """Один forward на пачку. Duck-typed tok/model — поэтому тестируется без моделей:
    фикс считает вызовы forward. Пустая строка не должна занимать место в пачке и
    не должна влиять на метки соседних текстов."""
    out = []
    with torch.no_grad():
        for i in range(0, len(texts), batch):
            chunk = texts[i:i + batch]
            enc = tok(chunk, return_tensors="pt", padding=True, truncation=True, max_length=512)
            ids, mask = _enc_pair(enc)
            res = model(input_ids=ids, attention_mask=mask)
            logits = res.logits if hasattr(res, "logits") else res
            for j, t in enumerate(chunk):
                p = _probs(logits[j], multilabel)
                top = max(zip(labels, p), key=lambda x: x[1])
                out.append({"model": name, "text": t, "top": top[0],
                            "p": round(top[1], 3),
                            "all": {k: round(v, 3) for k, v in zip(labels, p)}})
    return out



def load_no_mmap(repo):
    path = hf_hub_download(repo, "model.safetensors")
    cfg = AutoConfig.from_pretrained(repo)
    model = AutoModelForSequenceClassification.from_config(cfg)
    sd = load_bytes(read_bytes(path))          # bytes -> RAM, no mmap
    sd = {k: v for k, v in sd.items() if k != "bert.embeddings.position_ids"}
    model.load_state_dict(sd)
    # labels come from the model itself — a hardcoded list can silently disagree
    # with id2label order and print the wrong emotion for a comment
    id2label = cfg.id2label or {}
    if id2label:
        # keys may be ints (transformers) or strings (raw JSON) — accept both
        labels = [id2label.get(i) or id2label.get(str(i)) for i in range(len(id2label))]
        if None not in labels:
            return model.eval(), labels
        # пропуск в id2label: модель выдаёт класс, которого в списке нет, и top-метка
        # становится None — разбор молча теряет вердикт по этой модели
        key = repo_key(repo)
        if key and len(labels) == len(LABELS[key]):
            print(f"warning: id2label содержит пропуск, использую fallback-порядок {key}",
                  file=sys.stderr)
            labels = LABELS[key]
        else:
            labels = [l if l is not None else f"class_{i}" for i, l in enumerate(labels)]
    else:
        labels = LABELS[repo_key(repo)]
    return model.eval(), labels


def repo_key(repo):
    for k, v in REPOS.items():
        if v == repo:
            return k
    return None


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--texts", help="file with one text per line")
    g.add_argument("--text", help="single text")
    ap.add_argument("--out", help="write JSON here")
    ap.add_argument("--models", default="toxicity,sarcasm",
                    help="comma list from emotion,toxicity,sarcasm")
    ap.add_argument("--batch", type=int, default=BATCH,
                    help=f"размер пачки (правило: {BATCH}; 1 — поштучно, для сверки)")
    a = ap.parse_args()
    if a.batch < 1:
        sys.exit("--batch обязан быть >= 1")

    names = [n.strip() for n in a.models.split(",") if n.strip()]
    unknown = [n for n in names if n not in REPOS]
    if unknown:
        sys.exit(f"неизвестная модель {unknown}; доступны: {', '.join(sorted(REPOS))}")
    texts = [a.text] if a.text else [l.rstrip("\n") for l in read_lines(a.texts)]
    out = []
    for name in names:
        repo = REPOS[name]
        tok = AutoTokenizer.from_pretrained(repo)
        model, labels = load_no_mmap(repo)
        out.extend(infer(tok, model, labels, texts, MULTILABEL[name], name, a.batch))
    print(json.dumps(out, ensure_ascii=False, indent=1))
    if a.out:
        write_json(a.out, out)


if __name__ == "__main__":
    main()
