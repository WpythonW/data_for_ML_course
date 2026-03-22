# Annotation Report

**Dataset:** data/cleaned.csv (7 297 rows)
**Model:** cross-encoder/nli-MiniLM2-L6-H768
**Date:** 2026-03-22

## Task

Binary classification: human-written (0) vs AI-generated (1) sentences.

## Results

| Metric | Value |
|--------|-------|
| Rows labeled | 7 297 |
| Confidence mean | 0.80 |
| Confidence std | 0.12 |
| Flagged for review (< 0.75) | 2 271 (31%) |

## Label Distribution (auto-labeled)

| Label | Count | % |
|-------|-------|---|
| human | 6 822 | 93.5% |
| AI-generated | 475 | 6.5% |

## Note

The NLI zero-shot model skewed predictions heavily toward "human" — this is expected,
as a generic NLI model is not trained to distinguish AI-generated text.
Original dataset labels (0/1) are more reliable and used for Active Learning.
Low-confidence examples exported to `data/low_confidence.csv` for human review.
LabelStudio import: `data/labelstudio_import.json`
