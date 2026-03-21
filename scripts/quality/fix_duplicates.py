"""
Drop duplicate rows from tabular data.

Usage:
    uv run scripts/quality/fix_duplicates.py --input data/train.csv --output data/train_clean.csv
    uv run scripts/quality/fix_duplicates.py --input data/train.csv --keep last --output data/train_clean.csv
    uv run scripts/quality/fix_duplicates.py --input data/train.csv --subset id,email --output data/train_clean.csv

Options:
    --keep      first (default) | last | none — which duplicate to keep
    --subset    comma-separated column names to check (default: all columns)
"""

import argparse
import json
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Drop duplicate rows from tabular data")
    parser.add_argument("--input", required=True)
    parser.add_argument("--sep", default=",")
    parser.add_argument("--keep", choices=["first", "last", "none"], default="first",
                        help="Which occurrence to keep (default: first)")
    parser.add_argument("--subset", default="",
                        help="Comma-separated columns to check (default: all)")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    try:
        import pandas as pd
    except ImportError:
        print("pandas required: uv add pandas", file=sys.stderr); sys.exit(1)

    path = Path(args.input)
    ext = path.suffix.lower()
    df = pd.read_csv(path, sep=args.sep, low_memory=False) if ext == ".csv" else pd.read_parquet(path)

    subset = [c.strip() for c in args.subset.split(",") if c.strip()] or None
    keep = False if args.keep == "none" else args.keep

    rows_before = len(df)
    dups = int(df.duplicated(subset=subset, keep=keep).sum())
    df = df.drop_duplicates(subset=subset, keep=keep)
    rows_after = len(df)

    result = {
        "rows_before": rows_before,
        "rows_after": rows_after,
        "duplicates_removed": rows_before - rows_after,
        "keep": args.keep,
        "subset": subset or "all",
    }
    print(json.dumps(result, indent=2))

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.suffix.lower() == ".csv":
        df.to_csv(out_path, index=False, sep=args.sep)
    else:
        df.to_parquet(out_path, index=False)
    print(f"Saved to {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
