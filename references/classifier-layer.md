# Classifier layer (independent second opinion)

A fine-tuned encoder classifier answers one question only: **which technique is present in this span, according to annotated data**. It does not define the device and does not replace the closed list or the marker layer — it is a cross-check. Report disagreement, never adopt it silently.

## What it is good at (measured on published shared-task results)

| Task | Quality | Use |
|---|---|---|
| Binary detection of persuasive content at paragraph level | micro-F1 ~0.92 | decide **which** spans to analyse |
| Fine-grained per-technique classification | micro-F1 ~0.26–0.45 | weak; treat as a hint, not a verdict |

So the layer is a detector first, a namer second. The namer remains the LLM applying the closed list + the gate test.

## Candidate models (verified via HF API: size, languages, reported metric)

| Model | Params | Languages | Taxonomy | Note |
|---|---|---|---|---|
| `SlavicNLP/lrec2026-persuasion-sentence-classifier-russian` | 355M (ruRoberta-large) | ru | 25 techniques | trained on Russian **social media** — matches this use case |
| `Mishtar4/persuasion-detector-xlm-roberta` | 560M (XLM-R-large) | en, fr, de, it, pl, ru | 23 techniques | ships `thresholds.npy` (per-label thresholds for multi-label); best reported micro-F1 0.45 on Polish dev — Russian is cross-lingual transfer, not in training data |
| `SlavicNLP/SlavRoBERTa-Persuasion-Baseline` | 278M (XLM-R-base) | multilingual | 25 | baseline quality, smallest multilingual option |
| `q3fer/distilbert-base-fallacy-classification`, `heavyhelium/electra-base-logic-fallacy` | 67–110M | en only | logical fallacies | English-only texts; cheap second opinion |

Pick by domain match, not by size, and verify on your own texts before committing — published per-language F1 does not transfer to your samples. User preference for this layer: when it runs on CPU, take the most capable model, not the smallest; control the cost by gating (skip when the marker layer has no hits), not by shrinking the model.

Verification set: two manipulative texts, one text with a single unambiguous device, and one neutral factual control that must return zero hits — a model that fires on the control is noise. Measure quality and cost (load time, per-paragraph latency, peak RSS) on the same inputs for every candidate. Measured:

| Model | Neutral control | Clear false dilemma | Everyday text without terminology |
|---|---|---|---|
| `Mishtar4/persuasion-detector-xlm-roberta` (XLM-R-large) | clean, 0 hits | ложная дилемма 0.978 | нагруженная лексика 0.80, посеивание сомнения 0.47 |
| `SlavicNLP/lrec2026-...-russian` (ruRoberta-large) | "none" 0.96 | "none" — пропущена явная ложная дилемма | "none" 0.81 |

The Russian-domain model lost to the multilingual one on these samples, so the multilingual XLM-R-large is the layer in use (`psychology/models/persuasion_xlm_int8.onnx`). Russian input goes in directly; the reported 0.45 micro-F1 is Polish, Russian is cross-lingual transfer — hence "second opinion", not verdict.

## Do not translate the text to run a foreign-language model

Multilingual encoders (XLM-R family) take Russian input directly — translating to Polish adds a translation model (memory + latency), and translation **destroys the lexical evidence** the marker layer depends on: iterative triggers, appeal-to-common-sense phrases and counterfactual conditionals survive translation poorly, so the classifier then sees a different text than the one being analysed. Translate only when the model is monolingual and no multilingual alternative exists.

## Runtime recipe (memory- and speed-safe)

1. Export ONNX once with torch, then quantize int8 (2–4× smaller: 560M fp32 ≈ 1.1 GB → ~560 MB int8; 355M → ~355 MB). Pre-exported ONNX versions of these fine-tunes usually do not exist — export yourself.
2. Keep torch as an **export-time** dependency only; the runtime script loads `onnxruntime` + `tokenizers` (both already in the RAG venv), so the analysis path never imports torch.
3. Lazy-load the model on first use and keep it resident; **cache the tokenizer too** — re-reading `tokenizer.json` on every call cost 1.4 s per paragraph until it was cached. Measured (M1 CPU, int8 ONNX): session load 0.5 s, 19 ms per paragraph warm, peak RSS ~1.3 GB. On a GPU it is 30–50 ms.
4. Multi-label: use sigmoid + per-label thresholds (the model's `thresholds.npy` if present), not top-1 softmax — several techniques co-occur in one span.
5. Map labels to the closed list through `psychology/taxonomy_map.json`; unmapped labels stay as raw labels.
6. Run it only when the marker layer produced hits — no hits, no model load: the 0.5 s session load dominates a neutral message, while inference itself is milliseconds. A classifier on a single neutral sentence is noise.

## Output shape

Per paragraph: `label (p=0.xx) → device from the closed list`, plus the span that triggered it. If the classifier and the marker layer disagree, show both: the marker is the evidence, the classifier is the second opinion.
