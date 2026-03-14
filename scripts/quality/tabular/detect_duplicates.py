"""
Detect duplicates in tabular data — exact row matches or subset of columns.

Usage:
    uv run scripts/quality/tabular/detect_duplicates.py --input data/train.csv
    uv run scripts/quality/tabular/detect_duplicates.py --input data/train.csv \
        --subset id,email --keep first
    uv run scripts/quality/tabular/detect_duplicates.py --input data/train.csv \
        --show-samples --output data/duplicates_report.json
"""

import argparse
import json
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Detect duplicate rows in tabular data")
    parser.add_argument("--input", required=True)
    parser.add_argument("--sep", default=",")
    parser.add_argument("--subset", default="",
                        help="Comma-separated columns to check for duplicates (default: all columns)")
    parser.add_argument("--keep", choices=["first", "last", "none"], default="first",
                        help="Which occurrence to keep when reporting (default: first)")
    parser.add_argument("--show-samples", action="store_true",
                        help="Include sample duplicate rows in output")
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    try:
        import pandas as pd
    except ImportError:
        print("pandas required: uv add pandas", file=sys.stderr); sys.exit(1)

    path = Path(args.input)
    df = pd.read_csv(path, sep=args.sep, low_memory=False) if path.suffix.lower() == ".csv" else pd.read_parquet(path)

    subset = [c.strip() for c in args.subset.split(",") if c.strip()] or None
    keep = False if args.keep == "none" else args.keep

    dup_mask = df.duplicated(subset=subset, keep=keep)
    dup_count = int(dup_mask.sum())
    dup_pct = round(dup_count / len(df) * 100, 2)

    report = {
        "total_rows": len(df),
        "duplicate_rows": dup_count,
        "duplicate_pct": dup_pct,
        "subset_cols": subset or "all",
        "severity": "high" if dup_pct > 10 else "medium" if dup_pct > 1 else "low",
    }

    if args.show_samples and dup_count > 0:
        samples = df[dup_mask].head(5).to_dict(orient="records")
        report["sample_duplicates"] = samples

    out = json.dumps(report, ensure_ascii=False, indent=2)
    print(out)

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(out)
        print(f"Saved to {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
