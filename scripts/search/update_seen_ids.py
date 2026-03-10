"""
Update seen_ids.txt with all IDs from a raw results JSON file.

Usage:
    uv run scripts/search/update_seen_ids.py \
        --input data/raw_results_wave1.json \
        --seen-file data/seen_ids.txt
"""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Append dataset IDs from results file to seen_ids.txt")
    parser.add_argument("--input", required=True, help="Raw results JSON file")
    parser.add_argument("--seen-file", default="data/seen_ids.txt", help="Path to seen_ids.txt")
    args = parser.parse_args()

    seen_path = Path(args.seen_file)
    existing = set()
    if seen_path.exists():
        existing = set(line.strip() for line in seen_path.read_text().splitlines() if line.strip())

    data = json.loads(Path(args.input).read_text(encoding="utf-8"))
    new_ids = {d["id"] for d in data if d.get("id")}
    all_ids = existing | new_ids

    seen_path.parent.mkdir(parents=True, exist_ok=True)
    seen_path.write_text("\n".join(sorted(all_ids)))
    print(f"seen_ids.txt: {len(existing)} existing + {len(new_ids - existing)} new = {len(all_ids)} total")


if __name__ == "__main__":
    main()
