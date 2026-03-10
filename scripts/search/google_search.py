"""
Google Dataset Search (scraping — no API key required)
Usage: python scripts/search/google_search.py --query "climate data" --limit 10
"""

import argparse
import json
import sys
import time
import re
from urllib.parse import urlencode, quote_plus


def search_google_datasets(query, limit=10, output_format="json"):
    try:
        import requests
        from bs4 import BeautifulSoup
    except ImportError:
        print("Error: requests and beautifulsoup4 not installed.", file=sys.stderr)
        print("Run: pip install requests beautifulsoup4", file=sys.stderr)
        sys.exit(1)

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        )
    }

    base_url = "https://datasetsearch.research.google.com/search"
    params = {"query": query}
    url = f"{base_url}?{urlencode(params)}"

    print(f"Note: Google Dataset Search requires JavaScript. Returning direct search URL.", file=sys.stderr)
    print(f"Search URL: {url}", file=sys.stderr)

    # Google Dataset Search is a JS-heavy SPA — direct scraping doesn't work well.
    # We return structured info with the search URL and suggest alternatives.
    results = [
        {
            "id": f"google-dataset-search/{quote_plus(query)}",
            "name": f"Google Dataset Search: {query}",
            "description": (
                "Google Dataset Search indexes datasets from across the web. "
                "Click the URL to browse results interactively."
            ),
            "url": url,
            "size": "varies",
            "downloads": None,
            "likes": None,
            "tags": [query],
            "license": "varies",
            "last_updated": "unknown",
            "platform": "google"
        }
    ]

    # Try to get results via Schema.org structured data from known dataset repositories
    try:
        results.extend(_search_via_schema_org(query, limit, headers))
    except Exception as e:
        print(f"Warning: Schema.org search failed: {e}", file=sys.stderr)

    results = results[:limit]

    if output_format == "table":
        print_table(results)
    else:
        print(json.dumps(results, ensure_ascii=False, indent=2))


def _search_via_schema_org(query, limit, headers):
    """Search data.gov as a fallback — it uses standard metadata and is scrapable."""
    import requests
    from bs4 import BeautifulSoup

    results = []
    api_url = "https://catalog.data.gov/api/3/action/package_search"
    params = {"q": query, "rows": limit}

    try:
        resp = requests.get(api_url, params=params, headers=headers, timeout=10)
        resp.raise_for_status()
        data = resp.json()

        for item in data.get("result", {}).get("results", []):
            results.append({
                "id": f"data.gov/{item.get('name', '')}",
                "name": item.get("title", "")[:100],
                "description": (item.get("notes") or "")[:300],
                "url": f"https://catalog.data.gov/dataset/{item.get('name', '')}",
                "size": "unknown",
                "downloads": None,
                "likes": None,
                "tags": [t["name"] for t in item.get("tags", [])],
                "license": item.get("license_title", "unknown"),
                "last_updated": (item.get("metadata_modified") or "")[:10],
                "platform": "data.gov"
            })
    except Exception as e:
        print(f"Warning: data.gov search failed: {e}", file=sys.stderr)

    return results


def print_table(results):
    print(f"\n{'#':<4} {'Name':<40} {'Platform':<12} {'License':<15} URL")
    print("-" * 110)
    for i, ds in enumerate(results, 1):
        name = ds["name"][:38]
        platform = ds["platform"][:10]
        license_ = str(ds["license"])[:13]
        print(f"{i:<4} {name:<40} {platform:<12} {license_:<15} {ds['url']}")


def main():
    parser = argparse.ArgumentParser(
        description="Search datasets via Google Dataset Search and data.gov"
    )
    parser.add_argument("--query", required=True, help="Search query")
    parser.add_argument("--limit", type=int, default=10, help="Max results (default: 10)")
    parser.add_argument("--format", dest="output_format", choices=["json", "table"], default="json")
    args = parser.parse_args()

    search_google_datasets(
        query=args.query,
        limit=args.limit,
        output_format=args.output_format
    )


if __name__ == "__main__":
    main()
