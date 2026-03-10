"""
Kaggle Bulk Dataset Search via Kaggle CLI.
Uses `kaggle datasets list` with --csv output — no Python API complexity.
Output format matches hf_bulk_search.py for seamless merging.

Requirements: kaggle CLI installed (comes with `pip install kaggle`).
Credentials: KAGGLE_USERNAME + KAGGLE_KEY in .env (or ~/.kaggle/kaggle.json).

Usage:
    uv run scripts/search/kaggle_bulk_search.py \
        --queries "car brand classification,vehicle make model,stanford cars" \
        --output data/raw_kaggle_wave1.json

    # With filters:
    uv run scripts/search/kaggle_bulk_search.py \
        --queries "car brand,vehicle make" \
        --tags "computer-vision,image-classification" \
        --file-type csv \
        --output data/raw_kaggle_wave1.json

    # Excluding already-seen:
    uv run scripts/search/kaggle_bulk_search.py \
        --queries "..." \
        --exclude-ids-file data/seen_ids.txt \
        --output data/raw_kaggle_wave2.json
"""

import argparse
import csv
import io
import json
import os
import subprocess
import sys
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()


def check_credentials() -> dict:
    """Return env with Kaggle credentials set, or exit if missing."""
    env = os.environ.copy()
    username = os.getenv("KAGGLE_USERNAME", "")
    key = os.getenv("KAGGLE_KEY", "")
    if not username or not key:
        print("Error: KAGGLE_USERNAME and KAGGLE_KEY must be set in .env", file=sys.stderr)
        print("Get credentials: kaggle.com/settings/account → API → Create New Token", file=sys.stderr)
        sys.exit(1)
    env["KAGGLE_USERNAME"] = username
    env["KAGGLE_KEY"] = key
    return env


def search_one_query(query: str, limit: int, filters: dict, env: dict) -> list[dict]:
    """Run `kaggle datasets list` for one query, paginate until limit reached."""
    results = []
    page = 1
    pages_needed = max(1, -(-limit // 20))  # ceil(limit/20), each page = 20 results

    while page <= pages_needed:
        cmd = [
            "kaggle", "datasets", "list",
            "--search", query,
            "--page", str(page),
            "--csv",
        ]
        if filters.get("sort_by"):
            cmd += ["--sort-by", filters["sort_by"]]
        if filters.get("file_type"):
            cmd += ["--file-type", filters["file_type"]]
        if filters.get("tags"):
            cmd += ["--tags", filters["tags"]]
        if filters.get("license"):
            cmd += ["--license", filters["license"]]

        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, env=env, timeout=30
            )
            if result.returncode != 0:
                print(f"  Warning: kaggle CLI error for '{query}' p{page}: {result.stderr.strip()[:200]}", file=sys.stderr)
                break

            rows = list(csv.DictReader(io.StringIO(result.stdout)))
            if not rows:
                break

            for row in rows:
                ref = row.get("ref", "").strip()
                if not ref:
                    continue
                title = row.get("title", ref).strip()
                tags_raw = row.get("tags", "")
                tags = [t.strip() for t in tags_raw.split(",") if t.strip()]
                size_str = row.get("size", "").strip()
                downloads = _parse_int(row.get("downloadCount") or row.get("totalBytes") or "0")
                votes = _parse_int(row.get("voteCount", "0"))
                license_name = row.get("licenseName", "unknown").strip()
                subtitle = row.get("subtitle", "").strip()[:500]
                last_updated = (row.get("lastUpdated") or row.get("creationDate") or "")[:10]

                corpus_text = " ".join([ref, title, subtitle, " ".join(tags), license_name])

                results.append({
                    "id": f"kaggle:{ref}",
                    "name": title,
                    "description": subtitle,
                    "card_text": "",
                    "corpus_text": corpus_text,
                    "url": f"https://www.kaggle.com/datasets/{ref}",
                    "downloads": downloads,
                    "likes": votes,
                    "tags": tags,
                    "task_categories": [],
                    "license": license_name,
                    "size_categories": [size_str] if size_str else [],
                    "language": [],
                    "last_updated": last_updated,
                    "platform": "kaggle",
                    "matched_queries": [query],
                })

            if len(rows) < 20:
                break  # last page

        except subprocess.TimeoutExpired:
            print(f"  Warning: timeout for '{query}' p{page}", file=sys.stderr)
            break
        except FileNotFoundError:
            print("Error: kaggle CLI not found. Run: uv add kaggle", file=sys.stderr)
            sys.exit(1)

        page += 1

    return results[:limit]


def _parse_int(s: str) -> int:
    try:
        return int(str(s).replace(",", "").strip())
    except (ValueError, AttributeError):
        return 0


def deduplicate_and_merge(results: list[dict]) -> list[dict]:
    seen: dict[str, dict] = {}
    for r in results:
        if r["id"] in seen:
            seen[r["id"]]["matched_queries"].extend(r["matched_queries"])
        else:
            seen[r["id"]] = dict(r)
    for ds in seen.values():
        ds["query_match_count"] = len(set(ds["matched_queries"]))
    return list(seen.values())


def main():
    parser = argparse.ArgumentParser(description="Bulk Kaggle dataset search via CLI")
    parser.add_argument("--queries", required=True, help="Comma-separated search queries")
    parser.add_argument("--limit-per-query", type=int, default=40,
                        help="Max results per query (default: 40, Kaggle pages of 20)")
    parser.add_argument("--output", required=True, help="Output JSON file")
    parser.add_argument("--exclude-ids", default="", help="Comma-separated IDs to exclude")
    parser.add_argument("--exclude-ids-file", default="",
                        help="File with IDs to exclude, one per line")
    # Kaggle-specific filters
    parser.add_argument("--sort-by", default="hottest",
                        choices=["hottest", "votes", "updated", "active"],
                        help="Sort order (default: hottest)")
    parser.add_argument("--file-type", default="",
                        choices=["", "all", "csv", "sqlite", "json", "bigQuery"],
                        help="Filter by file type")
    parser.add_argument("--tags", default="",
                        help="Comma-separated tag IDs to filter by")
    parser.add_argument("--license", default="",
                        choices=["", "all", "cc", "gpl", "odb", "other"],
                        help="Filter by license type")
    args = parser.parse_args()

    env = check_credentials()

    queries = [q.strip() for q in args.queries.split(",") if q.strip()]

    exclude_ids = set(e.strip() for e in args.exclude_ids.split(",") if e.strip())
    if args.exclude_ids_file:
        p = Path(args.exclude_ids_file)
        if p.exists():
            file_ids = set(l.strip() for l in p.read_text().splitlines() if l.strip())
            exclude_ids |= file_ids
            print(f"  Loaded {len(file_ids)} excluded IDs from {args.exclude_ids_file}")

    filters = {
        "sort_by": args.sort_by,
        "file_type": args.file_type,
        "tags": args.tags,
        "license": args.license,
    }

    print(f"Searching Kaggle: {len(queries)} queries × ~{args.limit_per_query} results each")
    if any(v for v in filters.values()):
        active = {k: v for k, v in filters.items() if v}
        print(f"  Filters: {active}")

    all_results = []
    for i, query in enumerate(queries, 1):
        print(f"  [{i:>3}/{len(queries)}] '{query}'...", end=" ", flush=True)
        results = search_one_query(query, args.limit_per_query, filters, env)
        print(f"{len(results)} found")
        all_results.extend(results)

    unique = deduplicate_and_merge(all_results)

    if exclude_ids:
        before = len(unique)
        unique = [d for d in unique if d["id"] not in exclude_ids]
        print(f"  Excluded {before - len(unique)} already-seen datasets")

    unique.sort(key=lambda x: x.get("query_match_count", 0), reverse=True)

    print(f"\nTotal unique datasets: {len(unique)} (from {len(all_results)} raw hits)")
    print(f"Datasets matched by 3+ queries: {sum(1 for d in unique if d.get('query_match_count', 0) >= 3)}")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(unique, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved to: {args.output}")


if __name__ == "__main__":
    main()
