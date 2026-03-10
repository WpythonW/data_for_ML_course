"""
Run one full search wave: HF + Kaggle bulk search → merge → filter.
One command replaces 5 separate script calls.

Usage:
    uv run scripts/search/run_wave.py \
        --wave 1 \
        --queries "car brand,vehicle make model,car classification" \
        --keywords "Toyota,BMW,Ford,make,brand,manufacturer" \
        --goal "car brand/make image classification dataset with labeled photos"

    # With HF task filter:
    uv run scripts/search/run_wave.py \
        --wave 1 \
        --queries "car brand,vehicle make" \
        --keywords "Toyota,BMW,Ford" \
        --goal "car brand classification" \
        --hf-task image-classification

    # Skip Kaggle (e.g. no credentials):
    uv run scripts/search/run_wave.py \
        --wave 1 \
        --queries "..." \
        --keywords "..." \
        --goal "..." \
        --no-kaggle

All intermediate files go to data/. Seen and rejected IDs persist automatically.
Final filtered output: data/filtered_results_wave{N}.json
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

SCRIPTS = Path("scripts/search")


def run(cmd: list[str], desc: str) -> bool:
    print(f"\n>>> {desc}")
    print(f"    {' '.join(cmd)}\n")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"  ERROR: command failed (exit {result.returncode})", file=sys.stderr)
        return False
    return True


def main():
    parser = argparse.ArgumentParser(description="Run one full search wave (HF + Kaggle + filter)")
    parser.add_argument("--wave", type=int, required=True, help="Wave number (1, 2, or 3)")
    parser.add_argument("--queries", required=True, help="Comma-separated search queries")
    parser.add_argument("--keywords", required=True, help="Comma-separated BM25 keywords")
    parser.add_argument("--goal", required=True, help="Rich natural language goal description")
    parser.add_argument("--hf-task", default="", help="HF server-side task filter, e.g. image-classification")
    parser.add_argument("--hf-license", default="", help="HF server-side license filter")
    parser.add_argument("--hf-limit", type=int, default=100, help="HF results per query (default: 100)")
    parser.add_argument("--kaggle-limit", type=int, default=50, help="Kaggle results per query (default: 50)")
    parser.add_argument("--bm25-top", type=int, default=0,
                        help="BM25 top-N (0 = auto based on raw count)")
    parser.add_argument("--no-kaggle", action="store_true", help="Skip Kaggle search")
    args = parser.parse_args()

    n = args.wave
    data = Path("data")
    data.mkdir(exist_ok=True)

    raw_hf      = data / f"raw_hf_wave{n}.json"
    raw_kaggle  = data / f"raw_kaggle_wave{n}.json"
    raw_merged  = data / f"raw_results_wave{n}.json"
    filtered    = data / f"filtered_results_wave{n}.json"
    seen_file   = data / "seen_ids.txt"
    rejected_file = data / "rejected_ids.txt"

    # ── 1. HuggingFace bulk search ──
    hf_cmd = [
        "uv", "run", str(SCRIPTS / "hf_bulk_search.py"),
        "--queries", args.queries,
        "--limit-per-query", str(args.hf_limit),
        "--output", str(raw_hf),
        "--exclude-ids-file", str(seen_file),
    ]
    if args.hf_task:
        hf_cmd += ["--task-filter", args.hf_task]
    if args.hf_license:
        hf_cmd += ["--license-filter", args.hf_license]

    if not run(hf_cmd, f"Wave {n}: HuggingFace search"):
        sys.exit(1)

    # Print HF result count so agent can decide whether to re-run with different filters
    try:
        hf_count = len(json.load(open(raw_hf)))
        print(f"\n  HF raw results: {hf_count} datasets")

    # ── 2. Kaggle bulk search ──
    inputs_for_merge = [str(raw_hf)]

    if not args.no_kaggle and os.getenv("KAGGLE_USERNAME") and os.getenv("KAGGLE_KEY"):
        kaggle_cmd = [
            "uv", "run", str(SCRIPTS / "kaggle_bulk_search.py"),
            "--queries", args.queries,
            "--limit-per-query", str(args.kaggle_limit),
            "--output", str(raw_kaggle),
            "--exclude-ids-file", str(seen_file),
        ]
        if run(kaggle_cmd, f"Wave {n}: Kaggle search"):
            inputs_for_merge.append(str(raw_kaggle))
        else:
            print("  Kaggle search failed, continuing with HF only")
    else:
        if args.no_kaggle:
            print("\n>>> Skipping Kaggle (--no-kaggle)")
        else:
            print("\n>>> Skipping Kaggle (KAGGLE_USERNAME/KEY not set in .env)")

    # ── 3. Merge ──
    merge_cmd = [
        "uv", "run", str(SCRIPTS / "merge_results.py"),
        "--inputs", *inputs_for_merge,
        "--output", str(raw_merged),
    ]
    if not run(merge_cmd, f"Wave {n}: Merge sources"):
        sys.exit(1)

    # ── 4. Update seen_ids ──
    run([
        "uv", "run", str(SCRIPTS / "update_seen_ids.py"),
        "--input", str(raw_merged),
        "--seen-file", str(seen_file),
    ], f"Wave {n}: Update seen_ids.txt")

    # ── 5. Auto bm25-top if not set ──
    bm25_top = args.bm25_top
    if bm25_top == 0:
        try:
            with open(raw_merged) as f:
                count = len(json.load(f))
            if count < 100:
                bm25_top = 30
            elif count < 300:
                bm25_top = 50
            elif count < 600:
                bm25_top = 70
            else:
                bm25_top = 100
            print(f"\n>>> Auto bm25-top: {count} raw datasets → --bm25-top {bm25_top}")
        except Exception:
            bm25_top = 60

    # ── 6. Filter ──
    filter_cmd = [
        "uv", "run", str(SCRIPTS / "semantic_filter.py"),
        "--input", str(raw_merged),
        "--goal", args.goal,
        "--queries", args.queries,
        "--keywords", args.keywords,
        "--bm25-top", str(bm25_top),
        "--rejected-ids-file", str(rejected_file),
        "--output", str(filtered),
    ]
    if not run(filter_cmd, f"Wave {n}: Filter pipeline"):
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"Wave {n} complete. Results: {filtered}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
