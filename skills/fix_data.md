# Skill: fix(df, strategy)

## Purpose
Apply cleaning strategies to fix detected data quality issues. Chain outputs step by step.

---

## When to use
- After `detect_issues` has been run and QualityReport is available
- Never apply fixes without a prior detection step

---

## Available strategies

| Issue | Strategies |
|-------|-----------|
| missing | `mean`, `median`, `mode`, `ffill`, `drop_rows`, `knn`, `constant` |
| duplicates | `first`, `last`, `none` (drop all) |
| outliers | `clip_iqr`, `clip_zscore`, `cap_percentile`, `drop` |
| imbalance | `oversample`, `undersample`, `class_weights`, `skip` |

---

## Steps — chain outputs

```bash
# Step 1 — deduplicate
uv run scripts/quality/fix_duplicates.py \
    --input <original> --keep first \
    --output data/<name>_step1.csv

# Step 2 — fix missing values
uv run scripts/quality/fix_missing.py \
    --input data/<name>_step1.csv --strategy median \
    --output data/<name>_step2.csv

# Step 3 — fix outliers
uv run scripts/quality/fix_outliers.py \
    --input data/<name>_step2.csv --strategy clip_iqr \
    --output data/<name>_clean.csv
```

---

## Strategy selection guide

| ML task | Missing | Duplicates | Outliers | Imbalance |
|---------|---------|------------|----------|-----------|
| Classification | mode (cat) / median (num) | first | clip_iqr | leave for modeling |
| Regression | median | first | clip_iqr or drop | — |
| Clustering | median / drop_rows | first | clip_iqr | — |

---

## Rules
- Never overwrite original data — always save to a new file
- Report exact counts at each step: rows removed, values imputed, values clipped
- Imbalance ratio <5 → skip resampling, leave for modeling stage
