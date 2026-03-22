# Skill: check_quality(df_labeled)

## Purpose
Evaluate annotation quality: Cohen's κ, label distribution, confidence stats, flagged examples.

---

## When to use
- After `auto_label` to assess model confidence
- After human annotation to compare model vs human (Cohen's κ)

---

## Metrics returned

```python
{
  "label_dist":       {"positive": 0.44, "negative": 0.33, "neutral": 0.22},
  "label_counts":     {"positive": 16, "negative": 12, "neutral": 8},
  "confidence_mean":  0.89,
  "confidence_std":   0.15,
  "confidence_min":   0.50,
  "flagged_count":    7,
  "flagged_pct":      19.4,
  "kappa":            0.72,   # only if reference_col provided
  "kappa_vs":         "human_label"
}
```

---

## Cohen's κ interpretation

| κ value | Agreement |
|---------|-----------|
| < 0.20 | Slight |
| 0.20–0.40 | Fair |
| 0.40–0.60 | Moderate |
| 0.60–0.80 | Substantial ← target |
| > 0.80 | Almost perfect |

---

## CLI — after human annotation

```bash
# 1. Add human_label column to labeled.csv
# 2. Run:
uv run agents/annotation_agent.py \
    --input data/labeled.csv \
    --reference-col human_label \
    --output-dir data
```

---

## Python API

```python
from agents.annotation_agent import AnnotationAgent

agent = AnnotationAgent()
metrics = agent.check_quality(df_labeled, reference_col='human_label')
# → dict with kappa, label_dist, confidence_mean, flagged_pct
```

---

## How to print and interpret metrics

After `check_quality`, always print a human-readable summary and explain each value:

```python
import json

metrics = json.load(open('data/quality_metrics.json'))

print("=== Annotation Quality Metrics ===")
print(f"Label distribution:")
for label, pct in metrics['label_dist'].items():
    count = metrics['label_counts'][label]
    print(f"  {label}: {count} rows ({pct*100:.1f}%)")

print(f"\nConfidence:")
print(f"  Mean:  {metrics['confidence_mean']:.3f}  ← avg model certainty (target > 0.75)")
print(f"  Std:   {metrics['confidence_std']:.3f}   ← spread; high std = many uncertain examples")
print(f"  Min:   {metrics['confidence_min']:.3f}")

print(f"\nFlagged for review: {metrics['flagged_count']} rows ({metrics['flagged_pct']:.1f}%)")
print(f"  → These go to data/low_confidence.csv for human annotation")

if metrics.get('kappa') is not None:
    k = metrics['kappa']
    if k < 0.20:   interp = "Slight — model and human strongly disagree"
    elif k < 0.40: interp = "Fair — low agreement, reconsider class definitions"
    elif k < 0.60: interp = "Moderate — acceptable for pilot"
    elif k < 0.80: interp = "Substantial ✓ — good agreement"
    else:          interp = "Almost perfect ✓✓"
    print(f"\nCohen's κ vs human: {k:.3f} — {interp}")
else:
    print(f"\nCohen's κ: not computed (no human labels yet)")
    print(f"  → To compute: add 'human_label' column to labeled.csv and re-run check_quality")
```

### What each metric means

| Metric | What it tells you |
|--------|-------------------|
| `label_dist` | Class balance — if one class > 80%, model may be biased |
| `confidence_mean` | Overall model certainty — < 0.65 means poor fit (wrong language / domain) |
| `confidence_std` | High std (> 0.15) = model is unsure on many examples |
| `flagged_pct` | % sent to human review — > 50% means model needs better classes or stronger model |
| `kappa` | Agreement with human — only meaningful after human annotation |

### Red flags to report

- `confidence_mean < 0.65` → model language mismatch or classes too vague
- `flagged_pct > 50%` → lower threshold or switch to multilingual model
- One class > 80% in `label_dist` → classes imbalanced, check definitions
- `kappa < 0.40` after human annotation → annotation spec needs revision

---

## Rules
- κ requires at least 2 rows with both model and human labels
- If no `reference_col` — returns confidence stats only, kappa=None
- Save metrics to `data/quality_metrics.json`
- Always print and explain metrics after auto_label — do not just save to file
