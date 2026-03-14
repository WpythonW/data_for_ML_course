"""
Compare two datasets (before/after cleaning) — metrics side by side.
Works for tabular files. For other modalities, pass two profile JSONs.

Usage:
    # Compare two CSV files directly
    uv run scripts/quality/compare.py --before data/train.csv --after data/train_clean.csv

    # Compare two pre-computed profile JSONs (any modality)
    uv run scripts/quality/compare.py --profile-before data/profile_before.json \
                                       --profile-after data/profile_after.json

    # Save comparison report
    uv run scripts/quality/compare.py --before data/train.csv --after data/train_clean.csv \
                                       --output data/comparison.json
"""

import argparse
import json
import sys
from pathlib import Path


def load_tabular(path: Path, sep: str):
    try:
        import pandas as pd
    except ImportError:
        print("pandas required: uv add pandas", file=sys.stderr); sys.exit(1)
    ext = path.suffix.lower()
    if ext == ".csv":
        return pd.read_csv(path, sep=sep, low_memory=False)
    elif ext in (".parquet", ".pq"):
        return pd.read_parquet(path)
    elif ext in (".jsonl", ".ndjson"):
        return pd.read_json(path, lines=True)
    elif ext == ".json":
        return pd.read_json(path)
    else:
        print(f"Unsupported format: {ext}", file=sys.stderr); sys.exit(1)


def compare_dataframes(df_before, df_after) -> dict:
    import pandas as pd

    numeric_before = df_before.select_dtypes(include="number")
    numeric_after = df_after.select_dtypes(include="number")
    common_num = list(set(numeric_before.columns) & set(numeric_after.columns))

    def null_pct(df):
        return round(df.isnull().sum().sum() / max(df.size, 1) * 100, 3)

    rows_removed = len(df_before) - len(df_after)
    cols_removed = len(df_before.columns) - len(df_after.columns)

    metrics = {
        "rows": {"before": len(df_before), "after": len(df_after), "delta": -rows_removed},
        "cols": {"before": len(df_before.columns), "after": len(df_after.columns), "delta": -cols_removed},
        "missing_pct": {"before": null_pct(df_before), "after": null_pct(df_after)},
        "duplicates": {
            "before": int(df_before.duplicated().sum()),
            "after": int(df_after.duplicated().sum()),
        },
        "memory_mb": {
            "before": round(df_before.memory_usage(deep=True).sum() / 1e6, 2),
            "after": round(df_after.memory_usage(deep=True).sum() / 1e6, 2),
        },
    }

    # Per-column null comparison
    col_nulls = {}
    for col in df_before.columns:
        if col in df_after.columns:
            b = int(df_before[col].isnull().sum())
            a = int(df_after[col].isnull().sum())
            if b != a:
                col_nulls[col] = {"before": b, "after": a, "fixed": b - a}
    metrics["missing_by_col"] = col_nulls

    # Numeric stats comparison
    numeric_comparison = {}
    for col in common_num:
        sb = df_before[col].describe()
        sa = df_after[col].describe()
        numeric_comparison[col] = {
            "mean": {"before": round(float(sb["mean"]), 4), "after": round(float(sa["mean"]), 4)},
            "std": {"before": round(float(sb["std"]), 4), "after": round(float(sa["std"]), 4)},
            "min": {"before": round(float(sb["min"]), 4), "after": round(float(sa["min"]), 4)},
            "max": {"before": round(float(sb["max"]), 4), "after": round(float(sa["max"]), 4)},
        }
    metrics["numeric_stats"] = numeric_comparison

    return metrics


def compare_profiles(p_before: dict, p_after: dict) -> dict:
    """Generic comparison of two profile JSONs (any modality)."""
    def flatten(d, prefix=""):
        items = {}
        for k, v in d.items():
            key = f"{prefix}.{k}" if prefix else k
            if isinstance(v, dict):
                items.update(flatten(v, key))
            elif isinstance(v, (int, float, str)):
                items[key] = v
        return items

    flat_b = flatten(p_before)
    flat_a = flatten(p_after)
    all_keys = set(flat_b) | set(flat_a)

    comparison = {}
    for k in sorted(all_keys):
        vb = flat_b.get(k, "N/A")
        va = flat_a.get(k, "N/A")
        if vb != va:
            comparison[k] = {"before": vb, "after": va}

    return {"changed_fields": comparison, "before_modality": p_before.get("modality"), "after_modality": p_after.get("modality")}


def print_table(metrics: dict):
    """Print a readable comparison table."""
    print("\n" + "="*60)
    print("BEFORE vs AFTER comparison")
    print("="*60)

    top_keys = ["rows", "cols", "missing_pct", "duplicates", "memory_mb"]
    for k in top_keys:
        if k in metrics:
            v = metrics[k]
            if isinstance(v, dict) and "before" in v and "after" in v:
                delta = v.get("delta", v["after"] - v["before"] if isinstance(v["after"], (int, float)) else "")
                delta_str = f"  (Δ {delta:+})" if isinstance(delta, (int, float)) else ""
                print(f"  {k:<20} {str(v['before']):<15} → {str(v['after']):<15}{delta_str}")

    if metrics.get("missing_by_col"):
        print("\n  Missing values fixed:")
        for col, vals in metrics["missing_by_col"].items():
            print(f"    {col}: {vals['before']} → {vals['after']} (fixed {vals['fixed']})")

    print("="*60)


def main():
    parser = argparse.ArgumentParser(description="Compare dataset before/after cleaning")
    parser.add_argument("--before", default="", help="Path to original file (tabular)")
    parser.add_argument("--after", default="", help="Path to cleaned file (tabular)")
    parser.add_argument("--profile-before", default="", help="Path to profile JSON (any modality)")
    parser.add_argument("--profile-after", default="", help="Path to profile JSON (any modality)")
    parser.add_argument("--sep", default=",", help="CSV separator")
    parser.add_argument("--output", default="", help="Save comparison JSON to this path")
    args = parser.parse_args()

    if args.profile_before and args.profile_after:
        pb = json.loads(Path(args.profile_before).read_text())
        pa = json.loads(Path(args.profile_after).read_text())
        result = compare_profiles(pb, pa)
        print(json.dumps(result, indent=2))
    elif args.before and args.after:
        df_b = load_tabular(Path(args.before), args.sep)
        df_a = load_tabular(Path(args.after), args.sep)
        result = compare_dataframes(df_b, df_a)
        print_table(result)
        print(json.dumps(result, indent=2))
    else:
        print("Error: provide --before/--after or --profile-before/--profile-after", file=sys.stderr)
        sys.exit(1)

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(result, ensure_ascii=False, indent=2))
        print(f"\nSaved to {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
