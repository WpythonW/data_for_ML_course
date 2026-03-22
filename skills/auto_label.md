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

## After labeling — print metrics summary

Always print this block after `auto_label` completes:

```python
import pandas as pd, json

df = pd.read_csv('data/labeled.csv')
metrics = json.load(open('data/quality_metrics.json'))

total = len(df)
high_conf = (df['confidence'] >= 0.75).sum()
low_conf  = total - high_conf

print("=== Auto-label Results ===")
print(f"Total labeled     : {total}")
print(f"High confidence   : {high_conf} ({100*high_conf/total:.1f}%)  ← reliable, use as-is")
print(f"Flagged for review: {low_conf}  ({100*low_conf/total:.1f}%)  ← in data/low_confidence.csv")
print(f"Confidence mean   : {metrics['confidence_mean']:.3f}")
print()
print("Label distribution:")
for label, pct in metrics['label_dist'].items():
    bar = '█' * int(pct * 30)
    print(f"  {label:15s} {metrics['label_counts'][label]:5d}  ({pct*100:.1f}%)  {bar}")
```

**Interpret the output:**
- `confidence_mean > 0.75` → model fits the domain well
- `confidence_mean < 0.65` → likely language/domain mismatch — consider multilingual model or OpenRouter LLM
- `flagged > 50%` → too many uncertain examples — lower threshold OR fix class definitions
- Skewed label distribution (one class > 80%) → classes overlap or are ill-defined

---

## Rules
- Default confidence threshold: 0.75
- Flagged rows go to `data/low_confidence.csv` for human review (human-in-the-loop)
- Model loads ~120MB on first run — subsequent runs use cache
- Use `device=0` for GPU, `device=-1` for CPU
- Always run `check_quality` and print metrics after `auto_label` — see `skills/check_quality.md`
