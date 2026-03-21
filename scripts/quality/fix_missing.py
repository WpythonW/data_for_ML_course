"""
Fix missing values in tabular data with configurable strategies per column or globally.

Usage:
    # Global strategy for all columns
    uv run scripts/quality/tabular/fix_missing.py --input data/train.csv \
        --strategy median --output data/train_clean.csv

    # Per-column strategies
    uv run scripts/quality/tabular/fix_missing.py --input data/train.csv \
        --col-strategy "age=median,city=mode,score=mean,notes=drop_rows" \
        --output data/train_clean.csv

    # Drop columns above threshold
    uv run scripts/quality/tabular/fix_missing.py --input data/train.csv \
        --drop-cols-threshold 0.5 --strategy median --output data/train_clean.csv

Strategies:
    mean, median, mode         — fill with stat
    ffill, bfill               — forward/backward fill
    constant:<value>           — fill with a fixed value
    drop_rows                  — drop rows with any missing in this column
    drop_col                   — drop the entire column
    knn                        — KNN imputation (requires scikit-learn)
    interpolate                — linear interpolation (time series)
"""

import argparse
import json
import sys
from pathlib import Path


STRATEGIES = ["mean", "median", "mode", "ffill", "bfill", "drop_rows", "drop_col",
              "knn", "interpolate"]


def apply_strategy(df, col, strategy):
    import pandas as pd
    if strategy == "mean":
        df[col] = df[col].fillna(df[col].mean())
    elif strategy == "median":
        df[col] = df[col].fillna(df[col].median())
    elif strategy == "mode":
        mode = df[col].mode()
        if len(mode) > 0:
            df[col] = df[col].fillna(mode[0])
    elif strategy == "ffill":
        df[col] = df[col].ffill()
    elif strategy == "bfill":
        df[col] = df[col].bfill()
    elif strategy.startswith("constant:"):
        val = strategy.split(":", 1)[1]
        # Try numeric conversion
        try:
            val = float(val) if "." in val else int(val)
        except ValueError:
            pass
        df[col] = df[col].fillna(val)
    elif strategy == "interpolate":
        df[col] = df[col].interpolate()
    elif strategy == "drop_rows":
        df = df.dropna(subset=[col])
    elif strategy == "drop_col":
        df = df.drop(columns=[col])
    return df


def main():
    parser = argparse.ArgumentParser(description="Fix missing values in tabular data")
    parser.add_argument("--input", required=True)
    parser.add_argument("--sep", default=",")
    parser.add_argument("--strategy", default="median",
                        help=f"Global strategy for all columns. Options: {', '.join(STRATEGIES)}, constant:<value>")
    parser.add_argument("--col-strategy", default="",
                        help="Per-column overrides: 'col1=strategy1,col2=strategy2'")
    parser.add_argument("--drop-cols-threshold", type=float, default=1.0,
                        help="Drop columns with missing pct above this threshold (0-1, default: 1.0 = never)")
    parser.add_argument("--only-numeric", action="store_true",
                        help="Only apply numeric strategies (mean/median) to numeric columns")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    try:
        import pandas as pd
    except ImportError:
        print("pandas required: uv add pandas", file=sys.stderr); sys.exit(1)

    path = Path(args.input)
    df = pd.read_csv(path, sep=args.sep, low_memory=False) if path.suffix.lower() == ".csv" else pd.read_parquet(path)

    original_shape = df.shape
    original_nulls = int(df.isnull().sum().sum())

    # Parse per-column strategies
    col_strategies = {}
    if args.col_strategy:
        for pair in args.col_strategy.split(","):
            if "=" in pair:
                col, strat = pair.split("=", 1)
                col_strategies[col.strip()] = strat.strip()

    # Drop columns above threshold first
    if args.drop_cols_threshold < 1.0:
        null_pct = df.isnull().mean()
        cols_to_drop = null_pct[null_pct > args.drop_cols_threshold].index.tolist()
        if cols_to_drop:
            print(f"Dropping {len(cols_to_drop)} columns above {args.drop_cols_threshold*100:.0f}% threshold: {cols_to_drop}", file=sys.stderr)
            df = df.drop(columns=cols_to_drop)

    # KNN imputation — do all numeric columns at once
    global_strategy = args.strategy
    if global_strategy == "knn":
        try:
            from sklearn.impute import KNNImputer
            numeric_cols = df.select_dtypes(include="number").columns.tolist()
            imputer = KNNImputer(n_neighbors=5)
            df[numeric_cols] = imputer.fit_transform(df[numeric_cols])
            global_strategy = None  # already done
        except ImportError:
            print("scikit-learn required for knn: uv add scikit-learn", file=sys.stderr)
            sys.exit(1)

    # Apply per-column then global
    if global_strategy:
        cols_with_missing = df.columns[df.isnull().any()].tolist()
        for col in cols_with_missing:
            strat = col_strategies.get(col, global_strategy)
            if args.only_numeric and col not in df.select_dtypes(include="number").columns:
                continue
            df = apply_strategy(df, col, strat)

    final_nulls = int(df.isnull().sum().sum())
    changes = {
        "rows_before": original_shape[0], "rows_after": len(df),
        "cols_before": original_shape[1], "cols_after": len(df.columns),
        "nulls_before": original_nulls, "nulls_after": final_nulls,
        "nulls_fixed": original_nulls - final_nulls,
    }
    print(json.dumps(changes, indent=2))

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.suffix.lower() == ".csv":
        df.to_csv(out_path, index=False, sep=args.sep)
    else:
        df.to_parquet(out_path, index=False)
    print(f"Saved to {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
