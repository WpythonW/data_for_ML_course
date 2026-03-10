"""
Kaggle Dataset Search
Usage: python scripts/search/kaggle_search.py --query "medical imaging" --limit 10
Requires: KAGGLE_USERNAME and KAGGLE_KEY in .env
"""

import argparse
import json
import sys
import os
from dotenv import load_dotenv

load_dotenv()

# Kaggle API reads credentials from env vars
os.environ.setdefault("KAGGLE_USERNAME", os.getenv("KAGGLE_USERNAME", ""))
os.environ.setdefault("KAGGLE_KEY", os.getenv("KAGGLE_KEY", ""))


def check_credentials():
    username = os.getenv("KAGGLE_USERNAME")
    key = os.getenv("KAGGLE_KEY")
    if not username or not key:
        print("Error: KAGGLE_USERNAME and KAGGLE_KEY must be set in .env", file=sys.stderr)
        print("Get your credentials at: https://www.kaggle.com/settings/account", file=sys.stderr)
        sys.exit(1)


def search_kaggle(query, limit=10, sort_by="hottest", file_type=None, output_format="json"):
    check_credentials()

    try:
        from kaggle.api.kaggle_api_extended import KaggleApiExtended
    except ImportError:
        print("Error: kaggle not installed. Run: pip install kaggle", file=sys.stderr)
        sys.exit(1)

    api = KaggleApiExtended()
    api.authenticate()

    datasets = api.dataset_list(
        search=query,
        sort_by=sort_by,
        file_type=file_type,
        max_size=None,
        min_size=None,
        page=1,
    )

    results = []
    for ds in datasets[:limit]:
        results.append({
            "id": f"{ds.ref}",
            "name": ds.title,
            "description": (ds.subtitle or "")[:300],
            "url": f"https://www.kaggle.com/datasets/{ds.ref}",
            "size": format_size(ds.totalBytes),
            "downloads": ds.downloadCount or 0,
            "likes": ds.voteCount or 0,
            "tags": [t.name for t in (ds.tags or [])],
            "license": ds.licenseName or "unknown",
            "last_updated": str(ds.lastUpdated)[:10] if ds.lastUpdated else "unknown",
            "platform": "kaggle"
        })

    if output_format == "table":
        print_table(results)
    else:
        print(json.dumps(results, ensure_ascii=False, indent=2))


def format_size(bytes_val):
    if not bytes_val:
        return "unknown"
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if bytes_val < 1024:
            return f"{bytes_val:.1f} {unit}"
        bytes_val /= 1024
    return f"{bytes_val:.1f} PB"


def print_table(results):
    print(f"\n{'#':<4} {'Name':<40} {'Size':<10} {'Downloads':<12} {'Votes':<8} URL")
    print("-" * 110)
    for i, ds in enumerate(results, 1):
        name = ds["name"][:38]
        print(f"{i:<4} {name:<40} {ds['size']:<10} {ds['downloads']:<12} {ds['likes']:<8} {ds['url']}")


def main():
    parser = argparse.ArgumentParser(description="Search datasets on Kaggle")
    parser.add_argument("--query", required=True, help="Search query")
    parser.add_argument("--limit", type=int, default=10, help="Max results (default: 10)")
    parser.add_argument("--sort", dest="sort_by", choices=["hottest", "votes", "updated", "active"],
                        default="hottest", help="Sort order")
    parser.add_argument("--file-type", choices=["csv", "json", "sqlite", "bigQuery"],
                        help="Filter by file type")
    parser.add_argument("--format", dest="output_format", choices=["json", "table"], default="json")
    args = parser.parse_args()

    search_kaggle(
        query=args.query,
        limit=args.limit,
        sort_by=args.sort_by,
        file_type=args.file_type,
        output_format=args.output_format
    )


if __name__ == "__main__":
    main()
