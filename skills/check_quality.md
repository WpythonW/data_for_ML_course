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

## Rules
- κ requires at least 2 rows with both model and human labels
- If no `reference_col` — returns confidence stats only, kappa=None
- Save metrics to `data/quality_metrics.json`
