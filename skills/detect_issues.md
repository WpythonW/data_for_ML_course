# Skill: detect_issues(df)

## Purpose
Detect data quality problems in a tabular dataset: missing values, duplicates, outliers, class imbalance.

---

## When to use
- Before any cleaning — always detect first, never fix blind

---

## Steps

```bash
# 1. Profile — quick overview
uv run scripts/quality/profile.py --input <path>

# 2. Missing values
uv run scripts/quality/detect_missing.py --input <path> --output data/missing_<name>.json

# 3. Duplicates
uv run scripts/quality/detect_duplicates.py --input <path> --output data/duplicates_<name>.json

# 4. Outliers (IQR method)
uv run scripts/quality/detect_outliers.py --input <path> --method iqr --output data/outliers_<name>.json

# 5. Class imbalance (if label column exists)
uv run scripts/quality/detect_imbalance.py --input <path> --label <col> --output data/imbalance_<name>.json
```

---

## Output — QualityReport JSON

```json
{
  "shape": {"rows": N, "cols": M},
  "issues": {
    "missing":    {"columns": {...}, "total_pct": 0.0},
    "duplicates": {"count": N, "pct": 0.0},
    "outliers":   {"columns": {...}, "method": "iqr"},
    "imbalance":  {"ratio": 0.0, "distribution": {...}}
  },
  "severity": "low|medium|high|critical"
}
```

---

## Severity thresholds

| Issue | Low | Medium | High | Critical |
|-------|-----|--------|------|----------|
| Missing % | <5% | 5-20% | 20-40% | >40% |
| Duplicates % | <1% | 1-5% | 5-15% | >15% |
| Imbalance ratio | <3 | 3-5 | 5-10 | >10 |

---

## Rules
- Always run all 4 detectors for tabular data
- Never fix before detecting — show report first
- Sentinel nulls (`"?"`, `"N/A"`, `"-"`) → replace with NaN before running scripts
- Zero-inflated columns (>80% zeros) → skip IQR outlier fix, flag in report
