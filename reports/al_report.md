# Active Learning Report

**Dataset:** data/cleaned.csv (7 297 rows)
**Model:** LogisticRegression + TF-IDF
**Date:** 2026-03-22

## Setup

| Parameter | Value |
|-----------|-------|
| Initial labeled | 50 |
| Iterations | 5 |
| Batch size | 20 |
| Total labeled | 150 |
| Train/Test split | 5 837 / 1 460 |

## Results

| Iteration | N labeled | Entropy F1 | Random F1 |
|-----------|-----------|------------|-----------|
| 0 | 50 | 0.406 | 0.406 |
| 1 | 70 | 0.628 | 0.492 |
| 2 | 90 | 0.595 | 0.651 |
| 3 | 110 | 0.638 | 0.640 |
| 4 | 130 | 0.607 | 0.629 |
| 5 | 150 | **0.656** | **0.670** |

## Conclusion

On this dataset entropy sampling did not outperform random baseline.
See `al_report.md` § Why Entropy Lost for explanation.
Learning curve: `data/learning_curve.png`
