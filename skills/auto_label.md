# Skill: auto_label(df, modality)

## Purpose
Automatically label a DataFrame using zero-shot classification (text) or other modality-specific models.
Adds columns: `label`, `confidence`, `flagged_for_review`.

---

## When to use
- Input data has no labels (or partial labels)
- Modality is text, audio, or image

---

## Supported modalities

| Modality | Model | Notes |
|----------|-------|-------|
| text | `facebook/bart-large-mnli` | Zero-shot via NLI, no training needed |
| audio | Whisper | Transcribe → then label as text |
| image | YOLO | Object detection / classification |

---

## Text — zero-shot classification

```python
from transformers import pipeline

classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli", device=-1)
result = classifier(
    "This movie was absolutely amazing!",
    candidate_labels=["positive", "negative", "neutral"]
)
# result['labels'][0] → 'positive'
# result['scores'][0] → 0.97
```

**How it works:** The model checks which candidate label hypothesis best follows from the input text using NLI (Natural Language Inference). No labeled examples needed — you define classes as plain strings.

---

## CLI usage

```bash
uv run agents/annotation_agent.py \
    --input data/raw/texts.csv \
    --modality text \
    --labels "positive,negative,neutral" \
    --confidence-threshold 0.75 \
    --output-dir data
```

---

## Output columns added

| Column | Type | Description |
|--------|------|-------------|
| `label` | str | Predicted class |
| `confidence` | float | Model confidence score 0-1 |
| `flagged_for_review` | bool | True if confidence < threshold |
| `labeled_at` | str | ISO timestamp |

---

## Rules
- Default confidence threshold: 0.75
- Flagged rows go to `data/low_confidence.csv` for human review (human-in-the-loop)
- Model loads ~1.6GB on first run — subsequent runs use cache
- Use `device=0` for GPU, `device=-1` for CPU
