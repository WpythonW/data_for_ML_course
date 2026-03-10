"""
HuggingFace Bulk Dataset Search — deep metadata extraction.
Fetches dataset metadata, tags, and descriptions across many queries.
Saves rich raw results to JSON for downstream BM25 filtering.

Usage:
    uv run scripts/search/hf_bulk_search.py \
        --queries "car brand classification,vehicle make model,..." \
        --limit-per-query 100 \
        --output data/raw_results.json

    # With HF API filters (narrows results server-side):
    uv run scripts/search/hf_bulk_search.py \
        --queries "car brand,vehicle make" \
        --task-filter image-classification \
        --modality-filter image \
        --output data/raw_results.json

    # Excluding already-seen datasets:
    uv run scripts/search/hf_bulk_search.py \
        --queries "..." \
        --exclude-ids-file data/seen_ids.txt \
        --output data/raw_results_wave2.json

NOTE: Never use --fetch-cards here. Card fetching is done for finalists
      only inside semantic_filter.py (after Haiku step 1).
"""

import argparse
import json
import sys
import os
import time
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()


def fetch_dataset_card(api, dataset_id: str) -> str:
    """Fetch full README/card content for a single dataset."""
    try:
        from huggingface_hub import DatasetCard
        raw_card = DatasetCard.load(dataset_id)
        return str(raw_card)[:5000]
    except Exception:
        return ""


def search_one_query(api, query: str, limit: int, filters: dict) -> list[dict]:
    """Fetch datasets for a single query with metadata available in batch."""
    results = []
    try:
        kwargs = {"search": query, "limit": limit}
        # Apply server-side HF filters if provided — reduces API results immediately
        if filters.get("task"):
            kwargs["task_categories"] = filters["task"]
        if filters.get("license"):
            kwargs["license"] = filters["license"]

        datasets = list(api.list_datasets(**kwargs))

        for ds in datasets:
            card = getattr(ds, "cardData", None) or {}
            tags = ds.tags or []
            # description comes free in batch — available for ~40% of datasets
            description = getattr(ds, "description", None) or ""

            corpus_text = " ".join([
                ds.id,
                description,
                " ".join(tags),
                card.get("license", ""),
                " ".join(str(v) for v in card.get("task_categories", [])),
                " ".join(str(v) for v in card.get("task_ids", [])),
                str(card.get("pretty_name", "")),
            ])

            results.append({
                "id": ds.id,
                "name": ds.id.split("/")[-1],
                "description": description[:1000],
                "card_text": "",  # filled later for finalists only
                "corpus_text": corpus_text,
                "url": f"https://huggingface.co/datasets/{ds.id}",
                "downloads": getattr(ds, "downloads", 0) or 0,
                "likes": getattr(ds, "likes", 0) or 0,
                "tags": tags,
                "task_categories": card.get("task_categories", []),
                "license": card.get("license", "unknown"),
                "size_categories": card.get("size_categories", []),
                "language": card.get("language", []),
                "last_updated": str(ds.lastModified)[:10] if getattr(ds, "lastModified", None) else "unknown",
                "platform": "huggingface",
                "matched_queries": [query],
            })
    except Exception as e:
        print(f"  Warning: query '{query}' failed: {e}", file=sys.stderr)
    return results


def deduplicate_and_merge(results: list[dict]) -> list[dict]:
    """Deduplicate by id, merging matched_queries from duplicates."""
    seen: dict[str, dict] = {}
    for r in results:
        if r["id"] in seen:
            seen[r["id"]]["matched_queries"].extend(r["matched_queries"])
        else:
            seen[r["id"]] = r
    for ds in seen.values():
        ds["query_match_count"] = len(set(ds["matched_queries"]))
    return list(seen.values())


def main():
    parser = argparse.ArgumentParser(description="Bulk HuggingFace dataset search with optional server-side filters")
    parser.add_argument("--queries", required=True, help="Comma-separated search queries")
    parser.add_argument("--limit-per-query", type=int, default=100, help="Max datasets per query (default: 100)")
    parser.add_argument("--output", required=True, help="Output JSON file path")
    parser.add_argument("--exclude-ids", default="",
                        help="Comma-separated dataset IDs to exclude")
    parser.add_argument("--exclude-ids-file", default="",
                        help="File with dataset IDs to exclude, one per line (e.g. data/seen_ids.txt)")
    # HF API server-side filters — applied before results are returned
    parser.add_argument("--task-filter", default="",
                        help="HF task_categories filter, e.g. 'image-classification' (server-side)")
    parser.add_argument("--license-filter", default="",
                        help="HF license filter, e.g. 'cc-by-4.0' (server-side)")
    args = parser.parse_args()

    try:
        from huggingface_hub import HfApi
    except ImportError:
        print("Error: huggingface_hub not installed.", file=sys.stderr)
        sys.exit(1)

    token = os.getenv("HF_TOKEN")
    api = HfApi(token=token)

    queries = [q.strip() for q in args.queries.split(",") if q.strip()]

    exclude_ids = set(e.strip() for e in args.exclude_ids.split(",") if e.strip())
    if args.exclude_ids_file:
        ids_path = Path(args.exclude_ids_file)
        if ids_path.exists():
            file_ids = set(line.strip() for line in ids_path.read_text().splitlines() if line.strip())
            exclude_ids |= file_ids
            print(f"  Loaded exclude list from {args.exclude_ids_file}: {len(file_ids)} IDs")

    filters = {}
    if args.task_filter:
        filters["task"] = args.task_filter
    if args.license_filter:
        filters["license"] = args.license_filter

    print(f"Searching HuggingFace: {len(queries)} queries × {args.limit_per_query} results each")
    if filters:
        print(f"  Server-side filters: {filters}")
    if exclude_ids:
        print(f"  Excluding {len(exclude_ids)} already-seen dataset IDs")

    all_results = []
    for i, query in enumerate(queries, 1):
        print(f"  [{i:>3}/{len(queries)}] '{query}'...", end=" ", flush=True)
        results = search_one_query(api, query, args.limit_per_query, filters)
        print(f"{len(results)} found")
        all_results.extend(results)

    unique = deduplicate_and_merge(all_results)

    if exclude_ids:
        before = len(unique)
        unique = [d for d in unique if d["id"] not in exclude_ids]
        print(f"  Excluded {before - len(unique)} already-seen datasets")

    unique.sort(key=lambda x: x["query_match_count"], reverse=True)

    print(f"\nTotal unique datasets: {len(unique)} (from {len(all_results)} raw hits)")
    print(f"Datasets matched by 3+ queries: {sum(1 for d in unique if d['query_match_count'] >= 3)}")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(unique, f, ensure_ascii=False, indent=2)

    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    main()
