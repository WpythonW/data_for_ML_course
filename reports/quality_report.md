# Data Quality Report

**Dataset:** akoukas/chatgpt-classification-sentence-level
**Date:** 2026-03-22

## Issues Found

| Problem | Count | % | Severity |
|---------|-------|---|----------|
| Missing values | 0 | 0% | none |
| Duplicate rows | 47 | 0.64% | low |
| Outliers (numeric) | 0 | 0% | none |
| Class imbalance | 55% / 45% | ratio=1.2 | low |

## Strategy Applied: remove duplicates

Removed 47 duplicate rows. Result: 7 344 → 7 297 rows.

## Why This Strategy

The dataset was already very clean. Only duplicates warranted removal — they add no new information
and would bias evaluation metrics (same example in train and test). Mild class imbalance (1.2 ratio)
is well within acceptable range for binary classification and was left untouched.
