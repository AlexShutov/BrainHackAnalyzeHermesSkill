# Fine-tuning the classifier layer

Order: dataset → training → threshold calibration → export → eval → promote. Every step is gated; nothing reaches the runtime path automatically.

## Dataset

- Three tiers, in increasing reliability: annotated shared-task data (already labelled, its own taxonomy), weak labels from the current stack (only where marker layer and classifier agree), gold labels for devices the current model cannot name at all.
- **Gold labels must not come from the model being trained** — self-confirmation inflates the metric. Gold for uncovered devices comes from a different model plus human spot-check.
- **Split train/dev/test by source document, not by paragraph.** Paragraphs of the same book/post in different parts leak, and the metric lies. This is the single most common way a fine-tune looks better than it is.
- Devices with no examples must be reported as uncovered before training, not silently skipped from the metric.

## Training

- Multi-label: sigmoid per label, binary cross-entropy over active labels. Not top-1 softmax — several devices co-occur in one span.
- Memory budget (peak, fp16, seq 256–512): base encoder full fine-tune 6–10 GB; large encoder full fine-tune 14–18 GB; LoRA on large 8–12 GB. A 16 GB machine trains base comfortably (LoRA + gradient checkpointing, batch 8); large needs a 24 GB GPU or LoRA. Choose by the budget, not by ambition.
- Keep sequence length short (256) unless the device is text-level; text-level devices need multi-paragraph inputs, so their training rows must be whole texts, not single paragraphs.

## Thresholds

- Per-label thresholds are a separate artifact from the weights (`thresholds.npy`). Recalibrate them on dev for the FULL label set — thresholds tuned for a 23-label model will keep the newly added labels permanently silent.
- Target the false-positive rate on neutral controls first (<10%), then recall; a label with too few examples is not trustworthy and should be reported, not thresholded into looking alive.

## Export and promote

- Export ONNX once with torch, quantize int8, carry the tokenizer over; the runtime stays onnxruntime + tokenizers (torch is export-time only).
- New model goes to `staging/`, version file records the revision, state must reach `ready` after verification on the control texts, and promotion is an explicit script run that refuses an unverified or failed state. Never promote automatically.
- Promote only if the new model is not worse on the neutral controls and better on recall, measured on the same eval set as the incumbent.

## Reporting

- Report in-domain and real-world subsets separately (see the eval rule in SKILL.md). A drop in the aggregate after adding real examples is a harder set, not a regression.
