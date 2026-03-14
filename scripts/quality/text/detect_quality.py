"""
Detect text quality issues: empty strings, encoding errors, length outliers,
language anomalies, duplicate texts.

Usage:
    uv run scripts/quality/text/detect_quality.py --input data/train.csv --text-col text
    uv run scripts/quality/text/detect_quality.py --input data/corpus.jsonl \
        --text-col content --check-lang --lang en --output data/text_quality.json

Requirements: uv add pandas langdetect (langdetect optional, for --check-lang)
"""

import argparse
import json
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Detect text quality issues")
    parser.add_argument("--input", required=True)
    parser.add_argument("--sep", default=",")
    parser.add_argument("--text-col", required=True, help="Column containing text")
    parser.add_argument("--check-lang", action="store_true",
                        help="Detect language of each text (slow, uses langdetect)")
    parser.add_argument("--lang", default="en",
                        help="Expected language code (default: en). Used with --check-lang.")
    parser.add_argument("--min-length", type=int, default=10,
                        help="Flag texts shorter than this (chars, default: 10)")
    parser.add_argument("--max-length", type=int, default=0,
                        help="Flag texts longer than this (chars, default: 0 = no limit)")
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    try:
        import pandas as pd
    except ImportError:
        print("pandas required: uv add pandas", file=sys.stderr); sys.exit(1)

    path = Path(args.input)
    ext = path.suffix.lower()
    if ext in (".jsonl", ".ndjson"):
        df = pd.read_json(path, lines=True)
    elif ext == ".json":
        df = pd.read_json(path)
    elif ext == ".csv":
        df = pd.read_csv(path, sep=args.sep, low_memory=False)
    else:
        df = pd.read_parquet(path)

    if args.text_col not in df.columns:
        print(f"Error: column '{args.text_col}' not found. Available: {list(df.columns)}", file=sys.stderr)
        sys.exit(1)

    col = df[args.text_col]
    total = len(df)

    # Basic issues
    null_count = int(col.isnull().sum())
    empty_count = int((col == "").sum())
    whitespace_only = int(col.str.strip().eq("").sum() - empty_count)

    lengths = col.dropna().str.len()
    too_short = int((lengths < args.min_length).sum())
    too_long = int((lengths > args.max_length).sum()) if args.max_length > 0 else 0

    # Duplicates
    exact_dups = int(col.duplicated().sum())

    # Encoding issues (non-UTF8 indicators like replacement chars)
    encoding_issues = int(col.dropna().str.contains("\\ufffd|\\x00", regex=True, na=False).sum())

    report = {
        "total_rows": total,
        "text_col": args.text_col,
        "null": {"count": null_count, "pct": round(null_count/total*100, 2)},
        "empty": {"count": empty_count, "pct": round(empty_count/total*100, 2)},
        "whitespace_only": {"count": whitespace_only},
        "too_short": {"count": too_short, "threshold": args.min_length},
        "too_long": {"count": too_long, "threshold": args.max_length} if args.max_length else {},
        "exact_duplicates": {"count": exact_dups, "pct": round(exact_dups/total*100, 2)},
        "encoding_issues": {"count": encoding_issues},
        "length_stats": {
            "mean": round(float(lengths.mean()), 1),
            "min": int(lengths.min()),
            "max": int(lengths.max()),
            "p25": int(lengths.quantile(0.25)),
            "p50": int(lengths.quantile(0.50)),
            "p95": int(lengths.quantile(0.95)),
        },
    }

    # Language detection (optional, slow)
    if args.check_lang:
        try:
            from langdetect import detect, LangDetectException
            wrong_lang = 0
            unknown_lang = 0
            sample = col.dropna().sample(min(500, len(col.dropna())), random_state=42)
            for text in sample:
                try:
                    detected = detect(str(text))
                    if detected != args.lang:
                        wrong_lang += 1
                except LangDetectException:
                    unknown_lang += 1
            report["language"] = {
                "expected": args.lang,
                "sample_size": len(sample),
                "wrong_language_in_sample": wrong_lang,
                "unknown_in_sample": unknown_lang,
                "wrong_pct_estimate": round(wrong_lang / len(sample) * 100, 1),
            }
        except ImportError:
            print("langdetect required: uv add langdetect", file=sys.stderr)

    total_issues = null_count + empty_count + too_short + exact_dups + encoding_issues
    report["severity"] = (
        "critical" if total_issues / total > 0.3 else
        "high" if total_issues / total > 0.1 else
        "medium" if total_issues / total > 0.02 else
        "low"
    )

    out = json.dumps(report, ensure_ascii=False, indent=2)
    print(out)

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(out)
        print(f"Saved to {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
