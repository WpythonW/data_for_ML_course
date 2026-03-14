"""
Fix outliers in numeric columns with configurable strategies.

Usage:
    uv run scripts/quality/tabular/fix_outliers.py --input data/train.csv \
        --strategy clip_iqr --output data/train_clean.csv

    uv run scripts/quality/tabular/fix_outliers.py --input data/train.csv \
        --strategy cap_percentile --lower-p 1 --upper-p 99 \
        --cols age,salary --output data/train_clean.csv

    uv run scripts/quality/tabular/fix_outliers.py --input data/train.csv \
        --strategy drop --method zscore --threshold 3.5 --output data/train_clean.csv

Strategies:
    clip_iqr        — clip to [Q1 - k*IQR, Q3 + k*IQR]
    clip_zscore     — clip to [mean - k*std, mean + k*std]
    cap_percentile  — clip to [lower_p percentile, upper_p percentile]
    drop            — drop rows containing outliers
    winsorize       — same as cap_percentile but scipy-based
    flag_only       — add boolean flag columns, don't modify values
"""

import argparse
import json
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Fix outliers in tabular data")
    parser.add_argument("--input", required=True)
    parser.add_argument("--sep", default=",")
    parser.add_argument("--strategy", choices=["clip_iqr", "clip_zscore", "cap_percentile", "drop", "winsorize", "flag_only"],
                        default="clip_iqr")
    parser.add_argument("--method", choices=["iqr", "zscore"], default="iqr",
                        help="Detection method for --strategy drop (default: iqr)")
    parser.add_argument("--threshold", type=float, default=None,
                        help="IQR multiplier (default 1.5) or z-score k (default 3.0)")
    parser.add_argument("--lower-p", type=float, default=1.0,
                        help="Lower percentile for cap_percentile/winsorize (default: 1)")
    parser.add_argument("--upper-p", type=float, default=99.0,
                        help="Upper percentile for cap_percentile/winsorize (default: 99)")
    parser.add_argument("--cols", default="",
                        help="Comma-separated columns to process (default: all numeric)")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    try:
        import pandas as pd
        import numpy as np
    except ImportError:
        print("pandas + numpy required: uv add pandas numpy", file=sys.stderr); sys.exit(1)

    path = Path(args.input)
    df = pd.read_csv(path, sep=args.sep, low_memory=False) if path.suffix.lower() == ".csv" else pd.read_parquet(path)

    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    cols = [c.strip() for c in args.cols.split(",") if c.strip() in numeric_cols] if args.cols else numeric_cols

    original_rows = len(df)
    changes = {}

    for col in cols:
        series = df[col].dropna()
        if args.strategy == "clip_iqr":
            k = args.threshold or 1.5
            q1, q3 = series.quantile(0.25), series.quantile(0.75)
            iqr = q3 - q1
            lower, upper = q1 - k * iqr, q3 + k * iqr
            clipped = df[col].clip(lower=lower, upper=upper)
            changed = int((clipped != df[col]).sum())
            df[col] = clipped
            changes[col] = {"clipped": changed, "lower": round(float(lower), 4), "upper": round(float(upper), 4)}

        elif args.strategy == "clip_zscore":
            k = args.threshold or 3.0
            mean, std = series.mean(), series.std()
            lower, upper = mean - k * std, mean + k * std
            clipped = df[col].clip(lower=lower, upper=upper)
            changed = int((clipped != df[col]).sum())
            df[col] = clipped
            changes[col] = {"clipped": changed, "lower": round(float(lower), 4), "upper": round(float(upper), 4)}

        elif args.strategy in ("cap_percentile", "winsorize"):
            lower = float(series.quantile(args.lower_p / 100))
            upper = float(series.quantile(args.upper_p / 100))
            clipped = df[col].clip(lower=lower, upper=upper)
            changed = int((clipped != df[col]).sum())
            df[col] = clipped
            changes[col] = {"clipped": changed, "lower": round(lower, 4), "upper": round(upper, 4)}

        elif args.strategy == "flag_only":
            k = args.threshold or 1.5
            q1, q3 = series.quantile(0.25), series.quantile(0.75)
            iqr = q3 - q1
            lower, upper = q1 - k * iqr, q3 + k * iqr
            flag_col = f"{col}_outlier"
            df[flag_col] = ((df[col] < lower) | (df[col] > upper)).astype(int)
            flagged = int(df[flag_col].sum())
            changes[col] = {"flagged": flagged, "flag_col": flag_col}

    if args.strategy == "drop":
        k = args.threshold or (1.5 if args.method == "iqr" else 3.0)
        outlier_mask = pd.Series(False, index=df.index)
        for col in cols:
            series = df[col].dropna()
            if args.method == "iqr":
                q1, q3 = series.quantile(0.25), series.quantile(0.75)
                iqr = q3 - q1
                mask = (df[col] < q1 - k * iqr) | (df[col] > q3 + k * iqr)
            else:
                mean, std = series.mean(), series.std()
                mask = ((df[col] - mean) / std).abs() > k
            outlier_mask = outlier_mask | mask.fillna(False)
        rows_before = len(df)
        df = df[~outlier_mask]
        changes["_dropped_rows"] = rows_before - len(df)

    result = {
        "strategy": args.strategy,
        "cols_processed": len(cols),
        "rows_before": original_rows,
        "rows_after": len(df),
        "changes": changes,
    }
    print(json.dumps(result, indent=2))

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False, sep=args.sep) if out_path.suffix.lower() == ".csv" else df.to_parquet(out_path, index=False)
    print(f"Saved to {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
