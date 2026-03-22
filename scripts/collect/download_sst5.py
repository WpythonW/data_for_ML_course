"""
Download SetFit/sst5 from HuggingFace, save raw CSV and unified CSV.

Usage:
    uv run scripts/collect/download_sst5.py \
        --raw-output data/raw/sst5.csv \
        --unified-output data/raw/unified.csv \
        --max-rows 10000
"""

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from datasets import load_dataset


LABEL_MAP = {
    0: "very negative",
    1: "negative",
    2: "neutral",
    3: "positive",
    4: "very positive",
}

DATASET_ID = "SetFit/sst5"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-output", required=True)
    parser.add_argument("--unified-output", required=True)
    parser.add_argument("--max-rows", type=int, default=10000)
    args = parser.parse_args()

    print("Loading SetFit/sst5 from HuggingFace...", file=sys.stderr)
    ds = load_dataset(DATASET_ID, trust_remote_code=False)

    # Combine all available splits
    frames = []
    for split_name in ds.keys():
        df_split = ds[split_name].to_pandas()
        df_split["split"] = split_name
        frames.append(df_split)
        print(f"  Split '{split_name}': {len(df_split)} rows, columns: {list(df_split.columns)}", file=sys.stderr)

    df_raw = pd.concat(frames, ignore_index=True)
    print(f"Total rows: {len(df_raw)}", file=sys.stderr)

    # Sample if larger than max_rows
    if len(df_raw) > args.max_rows:
        df_raw = df_raw.sample(n=args.max_rows, random_state=42).reset_index(drop=True)
        print(f"Sampled to {len(df_raw)} rows (max_rows={args.max_rows})", file=sys.stderr)

    # Save raw CSV
    Path(args.raw_output).parent.mkdir(parents=True, exist_ok=True)
    df_raw.to_csv(args.raw_output, index=False)
    print(f"Raw CSV saved: {args.raw_output} ({len(df_raw)} rows)", file=sys.stderr)

    # Build unified schema
    # Determine text column name
    text_col = None
    for candidate in ["text", "sentence", "review", "content"]:
        if candidate in df_raw.columns:
            text_col = candidate
            break
    if text_col is None:
        # fallback: use first string column
        for col in df_raw.columns:
            if df_raw[col].dtype == object:
                text_col = col
                break

    # Determine label column
    label_col = None
    for candidate in ["label", "label_text", "category", "class"]:
        if candidate in df_raw.columns:
            label_col = candidate
            break
    if label_col is None:
        label_col = df_raw.columns[1]

    print(f"  text_col='{text_col}', label_col='{label_col}'", file=sys.stderr)

    collected_at = datetime.now(timezone.utc).isoformat()

    df_unified = pd.DataFrame({
        "text": df_raw[text_col],
        "audio": None,
        "image": None,
        "label": df_raw[label_col].apply(
            lambda x: LABEL_MAP.get(int(x), str(x)) if str(x).isdigit() else str(x)
        ),
        "source": f"hf:{DATASET_ID}",
        "collected_at": collected_at,
    })

    Path(args.unified_output).parent.mkdir(parents=True, exist_ok=True)
    df_unified.to_csv(args.unified_output, index=False)
    print(f"Unified CSV saved: {args.unified_output} ({len(df_unified)} rows)", file=sys.stderr)

    # Summary
    print("\n--- Summary ---", file=sys.stderr)
    print(f"Dataset: {DATASET_ID}", file=sys.stderr)
    print(f"Rows: {len(df_unified)}", file=sys.stderr)
    print(f"Label distribution:", file=sys.stderr)
    print(df_unified["label"].value_counts().to_string(), file=sys.stderr)


if __name__ == "__main__":
    main()
