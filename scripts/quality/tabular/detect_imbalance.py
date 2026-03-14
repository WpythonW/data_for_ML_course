"""
Detect class imbalance in a label column.

Usage:
    uv run scripts/quality/tabular/detect_imbalance.py --input data/train.csv --label target
    uv run scripts/quality/tabular/detect_imbalance.py --input data/train.csv --label label \
        --top 20 --output data/imbalance_report.json
"""

import argparse
import json
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Detect class imbalance in label column")
    parser.add_argument("--input", required=True)
    parser.add_argument("--sep", default=",")
    parser.add_argument("--label", required=True, help="Column name containing class labels")
    parser.add_argument("--top", type=int, default=50, help="Show top N classes (default: 50)")
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    try:
        import pandas as pd
    except ImportError:
        print("pandas required: uv add pandas", file=sys.stderr); sys.exit(1)

    path = Path(args.input)
    df = pd.read_csv(path, sep=args.sep, low_memory=False) if path.suffix.lower() == ".csv" else pd.read_parquet(path)

    if args.label not in df.columns:
        print(f"Error: column '{args.label}' not found. Available: {list(df.columns)}", file=sys.stderr)
        sys.exit(1)

    counts = df[args.label].value_counts()
    total = len(df)
    n_classes = len(counts)

    majority = int(counts.iloc[0])
    minority = int(counts.iloc[-1])
    imbalance_ratio = round(majority / minority, 2) if minority > 0 else float("inf")

    distribution = {
        str(k): {"count": int(v), "pct": round(v / total * 100, 2)}
        for k, v in counts.head(args.top).items()
    }

    report = {
        "label_col": args.label,
        "total_rows": total,
        "num_classes": n_classes,
        "majority_class": {"name": str(counts.index[0]), "count": majority, "pct": round(majority/total*100, 2)},
        "minority_class": {"name": str(counts.index[-1]), "count": minority, "pct": round(minority/total*100, 2)},
        "imbalance_ratio": imbalance_ratio,
        "severity": (
            "critical" if imbalance_ratio > 20 else
            "high" if imbalance_ratio > 10 else
            "medium" if imbalance_ratio > 3 else
            "low"
        ),
        "distribution": distribution,
        "missing_labels": int(df[args.label].isnull().sum()),
    }

    out = json.dumps(report, ensure_ascii=False, indent=2)
    print(out)

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(out)
        print(f"Saved to {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
