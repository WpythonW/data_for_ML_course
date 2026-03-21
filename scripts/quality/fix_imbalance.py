"""
Fix class imbalance in tabular data — oversample, undersample, or output class weights.

Usage:
    uv run scripts/quality/fix_imbalance.py --input data/train.csv --label target \
        --strategy oversample --output data/train_balanced.csv

    uv run scripts/quality/fix_imbalance.py --input data/train.csv --label target \
        --strategy undersample --output data/train_balanced.csv

    uv run scripts/quality/fix_imbalance.py --input data/train.csv --label target \
        --strategy class_weights --output data/class_weights.json

Strategies:
    oversample      — random oversample minority classes to match majority
    smote           — SMOTE oversampling (requires imbalanced-learn)
    undersample     — random undersample majority class to match minority
    class_weights   — compute sklearn-style class weights (no data modification)
"""

import argparse
import json
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Fix class imbalance in tabular data")
    parser.add_argument("--input", required=True)
    parser.add_argument("--sep", default=",")
    parser.add_argument("--label", required=True, help="Label column name")
    parser.add_argument("--strategy", choices=["oversample", "smote", "undersample", "class_weights"],
                        default="oversample")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    try:
        import pandas as pd
    except ImportError:
        print("pandas required: uv add pandas", file=sys.stderr); sys.exit(1)

    path = Path(args.input)
    ext = path.suffix.lower()
    df = pd.read_csv(path, sep=args.sep, low_memory=False) if ext == ".csv" else pd.read_parquet(path)

    if args.label not in df.columns:
        print(f"Error: label column '{args.label}' not found. Available: {list(df.columns)}", file=sys.stderr)
        sys.exit(1)

    counts_before = df[args.label].value_counts().to_dict()
    rows_before = len(df)

    if args.strategy == "class_weights":
        from collections import Counter
        counts = Counter(df[args.label])
        total = len(df)
        n_classes = len(counts)
        weights = {str(cls): round(total / (n_classes * cnt), 4) for cls, cnt in counts.items()}
        result = {
            "strategy": "class_weights",
            "weights": weights,
            "distribution_before": {str(k): int(v) for k, v in counts_before.items()},
        }
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, indent=2))
        print(json.dumps(result, indent=2))
        print(f"Saved weights to {args.output}", file=sys.stderr)
        return

    elif args.strategy == "oversample":
        import numpy as np
        rng = np.random.default_rng(args.random_state)
        majority_count = df[args.label].value_counts().max()
        parts = [df]
        for cls, cnt in df[args.label].value_counts().items():
            if cnt < majority_count:
                deficit = majority_count - cnt
                cls_df = df[df[args.label] == cls]
                sampled = cls_df.sample(n=deficit, replace=True, random_state=args.random_state)
                parts.append(sampled)
        df_out = pd.concat(parts).sample(frac=1, random_state=args.random_state).reset_index(drop=True)

    elif args.strategy == "undersample":
        minority_count = df[args.label].value_counts().min()
        parts = []
        for cls in df[args.label].unique():
            cls_df = df[df[args.label] == cls]
            parts.append(cls_df.sample(n=minority_count, replace=False, random_state=args.random_state))
        df_out = pd.concat(parts).sample(frac=1, random_state=args.random_state).reset_index(drop=True)

    elif args.strategy == "smote":
        try:
            from imblearn.over_sampling import SMOTE
        except ImportError:
            print("imbalanced-learn required: uv add imbalanced-learn", file=sys.stderr); sys.exit(1)
        feature_cols = [c for c in df.columns if c != args.label]
        X = df[feature_cols].select_dtypes(include="number")
        y = df[args.label]
        sm = SMOTE(random_state=args.random_state)
        X_res, y_res = sm.fit_resample(X, y)
        df_out = X_res.copy()
        df_out[args.label] = y_res

    counts_after = df_out[args.label].value_counts().to_dict()
    result = {
        "strategy": args.strategy,
        "rows_before": rows_before,
        "rows_after": len(df_out),
        "distribution_before": {str(k): int(v) for k, v in counts_before.items()},
        "distribution_after": {str(k): int(v) for k, v in counts_after.items()},
    }
    print(json.dumps(result, indent=2))

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.suffix.lower() == ".csv":
        df_out.to_csv(out_path, index=False, sep=args.sep)
    else:
        df_out.to_parquet(out_path, index=False)
    print(f"Saved to {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
