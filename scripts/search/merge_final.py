"""
Merge filtered results from all waves into a single final_results.json.

Usage:
    uv run scripts/search/merge_final.py \
        --wave-outputs data/filtered_results_wave1.json data/filtered_results_wave2.json \
        --output data/final_results.json
"""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Merge filtered wave results into final output")
    parser.add_argument("--wave-outputs", nargs="+", required=True, help="Filtered result JSONs per wave")
    parser.add_argument("--output", default="data/final_results.json", help="Final output path")
    args = parser.parse_args()

    all_datasets = []
    for path in args.wave_outputs:
        p = Path(path)
        if not p.exists():
            print(f"  Skipping missing: {path}")
            continue
        data = json.loads(p.read_text(encoding="utf-8"))
        all_datasets.extend(data)
        print(f"  {path}: {len(data)} datasets")

    seen = set()
    merged = []
    for d in sorted(all_datasets, key=lambda x: x.get("llm_relevance_score", 0), reverse=True):
        if d["id"] not in seen:
            seen.add(d["id"])
            merged.append(d)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(merged, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\nFinal: {len(merged)} unique datasets → {args.output}")
    for d in merged:
        score = d.get("llm_relevance_score", "?")
        verify = " ⚠️ verify" if d.get("needs_verification") else ""
        print(f"  [{score}/10]{verify} {d['id']}")


if __name__ == "__main__":
    main()
