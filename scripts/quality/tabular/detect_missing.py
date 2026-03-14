"""
Detect missing values in tabular data — patterns, severity, column-level stats.

Usage:
    uv run scripts/quality/tabular/detect_missing.py --input data/train.csv
    uv run scripts/quality/tabular/detect_missing.py --input data/train.csv \
        --threshold 0.05 --output data/missing_report.json
"""

import argparse
import json
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Detect missing values in tabular data")
    parser.add_argument("--input", required=True)
    parser.add_argument("--sep", default=",")
    parser.add_argument("--threshold", type=float, default=0.0,
                        help="Only report columns with missing pct above this (0-1, default: report all)")
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    try:
        import pandas as pd
    except ImportError:
        print("pandas required: uv add pandas", file=sys.stderr); sys.exit(1)

    path = Path(args.input)
    ext = path.suffix.lower()
    if ext == ".csv":
        df = pd.read_csv(path, sep=args.sep, low_memory=False)
    elif ext in (".parquet", ".pq"):
        df = pd.read_parquet(path)
    elif ext in (".jsonl", ".ndjson"):
        df = pd.read_json(path, lines=True)
    else:
        df = pd.read_json(path)

    total_cells = df.size
    null_counts = df.isnull().sum()
    null_pct = null_counts / len(df)

    columns = {}
    for col in df.columns:
        pct = float(null_pct[col])
        if pct > args.threshold:
            columns[col] = {
                "count": int(null_counts[col]),
                "pct": round(pct * 100, 2),
                "dtype": str(df[col].dtype),
                "severity": "critical" if pct > 0.5 else "high" if pct > 0.2 else "medium" if pct > 0.05 else "low",
            }

    # Detect missing patterns (rows with multiple missing)
    row_null_counts = df.isnull().sum(axis=1)
    rows_with_any = int((row_null_counts > 0).sum())
    rows_all_missing = int((row_null_counts == len(df.columns)).sum())

    report = {
        "total_rows": len(df),
        "total_cols": len(df.columns),
        "total_missing_cells": int(null_counts.sum()),
        "total_missing_pct": round(null_counts.sum() / total_cells * 100, 3),
        "rows_with_any_missing": rows_with_any,
        "rows_with_any_missing_pct": round(rows_with_any / len(df) * 100, 2),
        "rows_fully_missing": rows_all_missing,
        "columns_with_missing": len(columns),
        "columns": columns,
        "severity": (
            "critical" if null_counts.sum() / total_cells > 0.3 else
            "high" if null_counts.sum() / total_cells > 0.1 else
            "medium" if null_counts.sum() / total_cells > 0.02 else
            "low"
        ),
    }

    out = json.dumps(report, ensure_ascii=False, indent=2)
    print(out)

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(out)
        print(f"Saved to {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
