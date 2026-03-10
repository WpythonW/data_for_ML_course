"""
Merge multiple raw result JSON files, deduplicate by dataset ID.

Usage:
    uv run scripts/search/merge_results.py \
        --inputs data/raw_results_wave1.json data/raw_results_wave2.json \
        --output data/raw_results_merged.json
"""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Merge and deduplicate raw result JSONs")
    parser.add_argument("--inputs", nargs="+", required=True, help="Input JSON files to merge")
    parser.add_argument("--output", required=True, help="Output merged JSON file")
    args = parser.parse_args()

    seen: dict[str, dict] = {}
    for path in args.inputs:
        p = Path(path)
        if not p.exists():
            print(f"  Skipping missing file: {path}")
            continue
        data = json.loads(p.read_text(encoding="utf-8"))
        before = len(seen)
        for d in data:
            ds_id = d.get("id", "")
            if ds_id and ds_id not in seen:
                seen[ds_id] = d
            elif ds_id in seen:
                # merge matched_queries
                existing_qs = set(seen[ds_id].get("matched_queries", []))
                new_qs = set(d.get("matched_queries", []))
                seen[ds_id]["matched_queries"] = list(existing_qs | new_qs)
                seen[ds_id]["query_match_count"] = len(existing_qs | new_qs)
        print(f"  {path}: {len(data)} records, +{len(seen)-before} new unique")

    merged = list(seen.values())
    merged.sort(key=lambda x: x.get("query_match_count", 0), reverse=True)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(merged, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Merged: {len(merged)} unique datasets → {args.output}")


if __name__ == "__main__":
    main()
