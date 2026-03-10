"""
HuggingFace Dataset Search
Usage: python scripts/search/huggingface_search.py --query "medical imaging" --limit 10
"""

import argparse
import json
import sys
import os
from dotenv import load_dotenv

load_dotenv()


def search_huggingface(query, limit=10, tags=None, task=None, language=None, size=None, output_format="json"):
    try:
        from huggingface_hub import HfApi, DatasetFilter
    except ImportError:
        print("Error: huggingface_hub not installed. Run: pip install huggingface-hub", file=sys.stderr)
        sys.exit(1)

    token = os.getenv("HF_TOKEN")  # Optional, but higher rate limits
    api = HfApi(token=token)

    filter_kwargs = {}
    if task:
        filter_kwargs["task_categories"] = task
    if language:
        filter_kwargs["language"] = language
    if size:
        filter_kwargs["size_categories"] = size
    if tags:
        filter_kwargs["tags"] = tags if isinstance(tags, list) else tags.split(",")

    datasets = api.list_datasets(
        search=query,
        limit=limit,
        **filter_kwargs
    )

    results = []
    for ds in datasets:
        results.append({
            "id": ds.id,
            "name": ds.id.split("/")[-1],
            "description": (ds.description or "")[:300],
            "url": f"https://huggingface.co/datasets/{ds.id}",
            "size": str(ds.cardData.get("dataset_info", {}).get("dataset_size", "unknown")) if ds.cardData else "unknown",
            "downloads": ds.downloads or 0,
            "likes": ds.likes or 0,
            "tags": ds.tags or [],
            "license": ds.cardData.get("license", "unknown") if ds.cardData else "unknown",
            "last_updated": str(ds.lastModified)[:10] if ds.lastModified else "unknown",
            "platform": "huggingface"
        })

    if output_format == "table":
        print_table(results)
    else:
        print(json.dumps(results, ensure_ascii=False, indent=2))


def print_table(results):
    print(f"\n{'#':<4} {'Name':<40} {'Downloads':<12} {'Likes':<8} {'License':<15} URL")
    print("-" * 110)
    for i, ds in enumerate(results, 1):
        name = ds["name"][:38]
        print(f"{i:<4} {name:<40} {ds['downloads']:<12} {ds['likes']:<8} {str(ds['license'])[:13]:<15} {ds['url']}")


def main():
    parser = argparse.ArgumentParser(description="Search datasets on HuggingFace Hub")
    parser.add_argument("--query", required=True, help="Search query")
    parser.add_argument("--limit", type=int, default=10, help="Max results (default: 10)")
    parser.add_argument("--tags", help="Filter by tags (comma-separated)")
    parser.add_argument("--task", help="Filter by task type (e.g. text-classification)")
    parser.add_argument("--language", help="Filter by language code (e.g. ru, en)")
    parser.add_argument("--size", help="Filter by size (e.g. 1K<n<10K)")
    parser.add_argument("--format", dest="output_format", choices=["json", "table"], default="json")
    args = parser.parse_args()

    search_huggingface(
        query=args.query,
        limit=args.limit,
        tags=args.tags,
        task=args.task,
        language=args.language,
        size=args.size,
        output_format=args.output_format
    )


if __name__ == "__main__":
    main()
