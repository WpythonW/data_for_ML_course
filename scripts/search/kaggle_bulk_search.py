"""
Kaggle Bulk Dataset Search — fetches datasets across many queries.
Output format matches hf_bulk_search.py so results can be merged
and passed to semantic_filter.py unchanged.

Usage:
    uv run scripts/search/kaggle_bulk_search.py \
        --queries "car brand classification,vehicle make model,..." \
        --limit-per-query 50 \
        --output data/raw_results_kaggle_wave1.json

    # Excluding already-seen datasets:
    uv run scripts/search/kaggle_bulk_search.py \
        --queries "..." \
        --exclude-ids-file data/seen_ids.txt \
        --output data/raw_results_kaggle_wave2.json
"""

import argparse
import json
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

os.environ.setdefault("KAGGLE_USERNAME", os.getenv("KAGGLE_USERNAME", ""))
os.environ.setdefault("KAGGLE_KEY", os.getenv("KAGGLE_KEY", ""))


def check_credentials():
    if not os.getenv("KAGGLE_USERNAME") or not os.getenv("KAGGLE_KEY"):
        print("Error: KAGGLE_USERNAME and KAGGLE_KEY must be set in .env", file=sys.stderr)
        print("Get credentials: kaggle.com/settings/account → API → Create New Token", file=sys.stderr)
        sys.exit(1)


def search_one_query(api, query: str, limit: int) -> list[dict]:
    results = []
    try:
        datasets = api.dataset_list(search=query, sort_by="hottest", page=1)
        for ds in list(datasets)[:limit]:
            tags = [t.name for t in (ds.tags or [])]
            description = (ds.subtitle or "")[:1000]

            corpus_text = " ".join([
                str(ds.ref),
                ds.title or "",
                description,
                " ".join(tags),
                ds.licenseName or "",
            ])

            results.append({
                "id": f"kaggle:{ds.ref}",
                "name": ds.title or ds.ref,
                "description": description,
                "card_text": "",
                "corpus_text": corpus_text,
                "url": f"https://www.kaggle.com/datasets/{ds.ref}",
                "downloads": ds.downloadCount or 0,
                "likes": ds.voteCount or 0,
                "tags": tags,
                "task_categories": [],
                "license": ds.licenseName or "unknown",
                "size_categories": [_format_size(ds.totalBytes)] if ds.totalBytes else [],
                "language": [],
                "last_updated": str(ds.lastUpdated)[:10] if ds.lastUpdated else "unknown",
                "platform": "kaggle",
                "matched_queries": [query],
            })
    except Exception as e:
        print(f"  Warning: query '{query}' failed: {e}", file=sys.stderr)
    return results


def _format_size(b):
    if not b:
        return "unknown"
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if b < 1024:
            return f"{b:.0f}{unit}"
        b /= 1024
    return f"{b:.0f}PB"


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


def main():
    parser = argparse.ArgumentParser(description="Bulk Kaggle dataset search")
    parser.add_argument("--queries", required=True, help="Comma-separated search queries")
    parser.add_argument("--limit-per-query", type=int, default=50, help="Max datasets per query (default: 50)")
    parser.add_argument("--output", required=True, help="Output JSON file path")
    parser.add_argument("--exclude-ids", default="", help="Comma-separated dataset IDs to exclude")
    parser.add_argument("--exclude-ids-file", default="",
                        help="File with dataset IDs to exclude, one per line")
    args = parser.parse_args()

    check_credentials()

    try:
        from kaggle.api.kaggle_api_extended import KaggleApiExtended
    except ImportError:
        print("Error: kaggle not installed. Run: uv add kaggle", file=sys.stderr)
        sys.exit(1)

    api = KaggleApiExtended()
    api.authenticate()

    queries = [q.strip() for q in args.queries.split(",") if q.strip()]

    exclude_ids = set(e.strip() for e in args.exclude_ids.split(",") if e.strip())
    if args.exclude_ids_file:
        ids_path = Path(args.exclude_ids_file)
        if ids_path.exists():
            file_ids = set(line.strip() for line in ids_path.read_text().splitlines() if line.strip())
            exclude_ids |= file_ids
            print(f"  Loaded exclude list: {len(file_ids)} IDs from {args.exclude_ids_file}")

    print(f"Searching Kaggle: {len(queries)} queries × {args.limit_per_query} results each")
    if exclude_ids:
        print(f"  Excluding {len(exclude_ids)} already-seen IDs")

    all_results = []
    for i, query in enumerate(queries, 1):
        print(f"  [{i:>3}/{len(queries)}] '{query}'...", end=" ", flush=True)
        results = search_one_query(api, query, args.limit_per_query)
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
