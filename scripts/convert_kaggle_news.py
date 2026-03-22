#!/usr/bin/env python3
"""
Convert Topic-Labeled News Dataset from Kaggle to unified schema.
Input: data/tmp/labelled_newscatcher_dataset.csv (semicolon-delimited)
Output: data/raw/unified.csv
"""
import pandas as pd
import sys
from pathlib import Path
from datetime import datetime

def convert_to_unified_schema(input_file: str, output_file: str, max_rows: int = 5000):
    """Convert Kaggle news dataset to unified schema."""
    print(f"Loading dataset from {input_file}...", file=sys.stderr)

    # Read the CSV with semicolon delimiter
    df = pd.read_csv(input_file, sep=';', encoding='utf-8')

    print(f"Loaded {len(df)} rows", file=sys.stderr)
    print(f"Columns: {list(df.columns)}", file=sys.stderr)

    # Sample if larger than max_rows
    if len(df) > max_rows:
        print(f"Sampling to {max_rows} rows (was {len(df)})", file=sys.stderr)
        df = df.sample(n=max_rows, random_state=42)

    # Convert to unified schema
    unified = pd.DataFrame({
        'text': df['title'],  # Using title as main text
        'audio': None,
        'image': None,
        'label': df['topic'],
        'source': 'kaggle:kotartemiy/topic-labeled-news-dataset',
        'collected_at': pd.to_datetime(df['published_date']).dt.strftime('%Y-%m-%dT%H:%M:%SZ')
    })

    # Remove any rows with missing labels or text
    unified = unified.dropna(subset=['text', 'label'])

    print(f"After cleanup: {len(unified)} rows", file=sys.stderr)
    print(f"Unique labels: {unified['label'].nunique()}", file=sys.stderr)
    print(f"Label distribution:\n{unified['label'].value_counts()}", file=sys.stderr)

    # Save to output file
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    unified.to_csv(output_file, index=False)

    print(f"Saved to {output_file}", file=sys.stderr)
    print(f"Final dataset: {len(unified)} rows", file=sys.stderr)
    return len(unified)

if __name__ == '__main__':
    input_path = 'data/tmp/labelled_newscatcher_dataset.csv'
    output_path = 'data/raw/unified.csv'

    num_rows = convert_to_unified_schema(input_path, output_path)
    print(f"Conversion complete: {num_rows} rows saved to {output_path}")
