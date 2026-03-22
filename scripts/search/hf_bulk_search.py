"""
HuggingFace Bulk Dataset Search — async version.
Fetches dataset metadata across many queries in parallel via asyncio + httpx.

Usage:
    uv run scripts/search/hf_bulk_search.py \
        --queries "text classification,sentiment analysis,..." \
        --limit-per-query 100 \
        --output data/raw_results.json

    # With HF API filters:
    uv run scripts/search/hf_bulk_search.py \
        --queries "sentiment,opinion" \
        --task-filter text-classification \
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
import asyncio
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

HF_API_BASE = "https://huggingface.co/api/datasets"


async def search_one_query(session, query: str, limit: int, filters: dict, token: str) -> list[dict]:
    """Fetch datasets for a single query asynchronously."""
    import httpx

    params = {"search": query, "limit": limit, "full": "True"}
    if filters.get("task"):
        params["filter"] = filters["task"]
    if filters.get("license"):
        params["license"] = filters["license"]

    headers = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    try:
        resp = await session.get(HF_API_BASE, params=params, headers=headers, timeout=30)
        resp.raise_for_status()
        datasets = resp.json()
    except Exception as e:
        print(f"  Warning: query '{query}' failed: {e}", file=sys.stderr)
        return []

    results = []
    for ds in datasets:
        ds_id = ds.get("id", "")
        if not ds_id:
            continue
        card = ds.get("cardData") or {}
        tags = ds.get("tags") or []
        description = ds.get("description") or ""

        license_val = card.get("license", "") or ""
        if isinstance(license_val, list):
            license_val = " ".join(license_val)
        corpus_text = " ".join([
            ds_id,
            description,
            " ".join(tags),
            license_val,
            " ".join(str(v) for v in card.get("task_categories", [])),
            " ".join(str(v) for v in card.get("task_ids", [])),
            str(card.get("pretty_name", "")),
        ])

        # Extract dataset size in KB from cardData.dataset_info if available
        size_kb = None
        dataset_info = card.get("dataset_info") or {}
        if isinstance(dataset_info, dict):
            size_bytes = dataset_info.get("dataset_size") or dataset_info.get("size_in_bytes")
            if size_bytes:
                size_kb = round(size_bytes / 1024, 1)
        elif isinstance(dataset_info, list) and dataset_info:
            size_bytes = dataset_info[0].get("dataset_size") or dataset_info[0].get("size_in_bytes")
            if size_bytes:
                size_kb = round(size_bytes / 1024, 1)

        results.append({
            "id": ds_id,
            "name": ds_id.split("/")[-1],
            "description": description[:1000],
            "card_text": "",
            "corpus_text": corpus_text,
            "url": f"https://huggingface.co/datasets/{ds_id}",
            "downloads": ds.get("downloads", 0) or 0,
            "likes": ds.get("likes", 0) or 0,
            "tags": tags,
            "task_categories": card.get("task_categories", []),
            "license": card.get("license", "unknown"),
            "size_categories": card.get("size_categories", []),
            "size_kb": size_kb,
            "language": card.get("language", []),
            "last_updated": str(ds.get("lastModified", ""))[:10] or "unknown",
            "platform": "huggingface",
            "matched_queries": [query],
        })
    return results


def deduplicate_and_merge(results: list[dict]) -> list[dict]:
    seen: dict[str, dict] = {}
    for r in results:
        if r["id"] in seen:
            seen[r["id"]]["matched_queries"].extend(r["matched_queries"])
        else:
            seen[r["id"]] = r
    for ds in seen.values():
        ds["query_match_count"] = len(set(ds["matched_queries"]))
    return list(seen.values())


async def run_search(queries: list[str], limit: int, filters: dict, token: str) -> list[dict]:
    import httpx

    async with httpx.AsyncClient() as session:
        tasks = [search_one_query(session, q, limit, filters, token) for q in queries]
        results_per_query = await asyncio.gather(*tasks)

    print(f"  Queries done: " + ", ".join(
        f"'{q}' → {len(r)}" for q, r in zip(queries, results_per_query)
    ), file=sys.stderr)

    all_results = [item for sublist in results_per_query for item in sublist]
    return all_results


def main():
    parser = argparse.ArgumentParser(description="Async bulk HuggingFace dataset search")
    parser.add_argument("--queries", required=True, help="Comma-separated search queries")
    parser.add_argument("--limit-per-query", type=int, default=100, help="Max datasets per query (default: 100)")
    parser.add_argument("--output", required=True, help="Output JSON file path")
    parser.add_argument("--exclude-ids", default="", help="Comma-separated dataset IDs to exclude")
    parser.add_argument("--exclude-ids-file", default="",
                        help="File with dataset IDs to exclude, one per line")
    parser.add_argument("--task-filter", default="", help="HF task_categories filter (server-side)")
    parser.add_argument("--license-filter", default="", help="HF license filter (server-side)")
    args = parser.parse_args()

    try:
        import httpx
    except ImportError:
        print("Error: httpx not installed. Run: uv add httpx", file=sys.stderr)
        sys.exit(1)

    token = os.getenv("HF_TOKEN", "")
    queries = [q.strip() for q in args.queries.split(",") if q.strip()]

    exclude_ids = set(e.strip() for e in args.exclude_ids.split(",") if e.strip())
    if args.exclude_ids_file:
        ids_path = Path(args.exclude_ids_file)
        if ids_path.exists():
            file_ids = set(line.strip() for line in ids_path.read_text().splitlines() if line.strip())
            exclude_ids |= file_ids
            print(f"  Loaded exclude list: {len(file_ids)} IDs")

    filters = {}
    if args.task_filter:
        filters["task"] = args.task_filter
    if args.license_filter:
        filters["license"] = args.license_filter

    print(f"Searching HuggingFace async: {len(queries)} queries × {args.limit_per_query} (parallel)")
    if filters:
        print(f"  Server-side filters: {filters}")

    all_results = asyncio.run(run_search(queries, args.limit_per_query, filters, token))
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
