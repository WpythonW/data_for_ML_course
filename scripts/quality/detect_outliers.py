"""
Detect outliers in numeric columns — IQR, z-score, or isolation forest.

Usage:
    uv run scripts/quality/tabular/detect_outliers.py --input data/train.csv
    uv run scripts/quality/tabular/detect_outliers.py --input data/train.csv \
        --method zscore --threshold 3.0 --cols age,salary,score
    uv run scripts/quality/tabular/detect_outliers.py --input data/train.csv \
        --method isolation_forest --contamination 0.05
"""

import argparse
import json
import sys
from pathlib import Path


def detect_iqr(series, multiplier=1.5):
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - multiplier * iqr
    upper = q3 + multiplier * iqr
    mask = (series < lower) | (series > upper)
    return mask, float(lower), float(upper)


def detect_zscore(series, threshold=3.0):
    mean = series.mean()
    std = series.std()
    if std == 0:
        return series.apply(lambda x: False), 0.0, 0.0
    z = (series - mean) / std
    mask = z.abs() > threshold
    return mask, float(mean - threshold * std), float(mean + threshold * std)


def main():
    parser = argparse.ArgumentParser(description="Detect outliers in numeric columns")
    parser.add_argument("--input", required=True)
    parser.add_argument("--sep", default=",")
    parser.add_argument("--method", choices=["iqr", "zscore", "isolation_forest"], default="iqr")
    parser.add_argument("--threshold", type=float, default=None,
                        help="IQR multiplier (default 1.5) or z-score threshold (default 3.0)")
    parser.add_argument("--contamination", type=float, default=0.05,
                        help="Expected fraction of outliers for isolation_forest (default: 0.05)")
    parser.add_argument("--cols", default="",
                        help="Comma-separated column names to check (default: all numeric)")
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    try:
        import pandas as pd
    except ImportError:
        print("pandas required: uv add pandas", file=sys.stderr); sys.exit(1)

    path = Path(args.input)
    ext = path.suffix.lower()
    df = pd.read_csv(path, sep=args.sep, low_memory=False) if ext == ".csv" else pd.read_parquet(path)

    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    if args.cols:
        cols = [c.strip() for c in args.cols.split(",") if c.strip() in numeric_cols]
    else:
        cols = numeric_cols

    results = {}

    if args.method == "isolation_forest":
        try:
            from sklearn.ensemble import IsolationForest
        except ImportError:
            print("scikit-learn required: uv add scikit-learn", file=sys.stderr); sys.exit(1)
        data = df[cols].dropna()
        clf = IsolationForest(contamination=args.contamination, random_state=42)
        preds = clf.fit_predict(data)
        outlier_idx = data.index[preds == -1].tolist()
        results["_global"] = {
            "method": "isolation_forest",
            "contamination": args.contamination,
            "outlier_count": len(outlier_idx),
            "outlier_pct": round(len(outlier_idx) / len(df) * 100, 2),
            "outlier_indices_sample": outlier_idx[:20],
        }
    else:
        threshold = args.threshold or (1.5 if args.method == "iqr" else 3.0)
        for col in cols:
            series = df[col].dropna()
            if args.method == "iqr":
                mask, lower, upper = detect_iqr(series, threshold)
            else:
                mask, lower, upper = detect_zscore(series, threshold)

            count = int(mask.sum())
            if count > 0:
                results[col] = {
                    "method": args.method,
                    "threshold": threshold,
                    "outlier_count": count,
                    "outlier_pct": round(count / len(series) * 100, 2),
                    "lower_bound": round(lower, 4),
                    "upper_bound": round(upper, 4),
                    "min_value": round(float(series.min()), 4),
                    "max_value": round(float(series.max()), 4),
                    "mean": round(float(series.mean()), 4),
                    "severity": "high" if count / len(series) > 0.1 else "medium" if count / len(series) > 0.02 else "low",
                }

    total_outlier_rows = sum(v.get("outlier_count", 0) for k, v in results.items() if k != "_global")
    report = {
        "method": args.method,
        "columns_checked": len(cols),
        "columns_with_outliers": len(results),
        "total_outlier_instances": total_outlier_rows,
        "columns": results,
    }

    out = json.dumps(report, ensure_ascii=False, indent=2)
    print(out)

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(out)
        print(f"Saved to {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
