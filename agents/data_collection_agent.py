"""
DataCollectionAgent — сбор данных из нескольких источников в unified DataFrame.

Технический контракт:
    agent = DataCollectionAgent(config='config_annotation.yaml')
    df = agent.run(sources=[
        {'type': 'hf_dataset', 'name': 'imdb'},
        {'type': 'scrape', 'url': '...', 'selector': '...'},
        {'type': 'api', 'endpoint': '...', 'params': {}},
        {'type': 'kaggle', 'name': '...'},
    ])

Usage:
    uv run agents/data_collection_agent.py --config config_annotation.yaml
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

UNIFIED_COLS = ["text", "audio", "image", "label", "source", "collected_at"]


class DataCollectionAgent:
    """Collects data from multiple sources and returns a unified DataFrame."""

    def __init__(self, config: str | None = None):
        self.config: dict = {}
        if config and Path(config).exists():
            self.config = yaml.safe_load(Path(config).read_text())
            print(f"[init] Loaded config: {config}", file=sys.stderr)

    # ------------------------------------------------------------------
    # skill: load_dataset
    # ------------------------------------------------------------------

    def load_dataset(self, name: str, source: str = "hf", split: str = "train") -> pd.DataFrame:
        """
        Load an open dataset from HuggingFace or Kaggle.
        Returns DataFrame with unified columns pre-filled where possible.
        """
        print(f"[load_dataset] source={source} name={name} split={split}", file=sys.stderr)

        if source == "hf":
            from datasets import load_dataset as hf_load
            ds = hf_load(name, split=split, trust_remote_code=False)
            df = ds.to_pandas()
            # Try to find text and label columns automatically
            text_col = next((c for c in df.columns if "text" in c.lower()), df.columns[0])
            label_col = next((c for c in df.columns if "label" in c.lower()), None)
            result = pd.DataFrame()
            result["text"] = df[text_col].astype(str)
            result["label"] = df[label_col].astype(str) if label_col else None
            result["source"] = f"hf:{name}"

        elif source == "kaggle":
            # Expects dataset already downloaded to data/raw/<name>/
            csv_files = list(Path(f"data/raw/{name}").glob("*.csv"))
            if not csv_files:
                raise FileNotFoundError(f"No CSV found in data/raw/{name}/. Run: kaggle datasets download {name}")
            df = pd.read_csv(csv_files[0])
            text_col = next((c for c in df.columns if "text" in c.lower()), df.columns[0])
            label_col = next((c for c in df.columns if "label" in c.lower()), None)
            result = pd.DataFrame()
            result["text"] = df[text_col].astype(str)
            result["label"] = df[label_col].astype(str) if label_col else None
            result["source"] = f"kaggle:{name}"

        else:
            raise ValueError(f"Unknown source '{source}'. Use 'hf' or 'kaggle'.")

        result["collected_at"] = datetime.now(timezone.utc).isoformat()
        print(f"[load_dataset] Loaded {len(result)} rows from {source}:{name}", file=sys.stderr)
        return result

    # ------------------------------------------------------------------
    # skill: scrape
    # ------------------------------------------------------------------

    def scrape(self, url: str, selector: str, output: str | None = None) -> pd.DataFrame:
        """
        Scrape structured text data from a URL using Playwright.
        selector: CSS selector for target elements.
        """
        import asyncio
        from playwright.async_api import async_playwright

        print(f"[scrape] {url} selector={selector}", file=sys.stderr)

        async def _scrape():
            async with async_playwright() as p:
                browser = await p.chromium.launch(headless=True)
                page = await browser.new_page()
                await page.goto(url, wait_until="networkidle", timeout=30000)
                elements = await page.query_selector_all(selector)
                texts = []
                for el in elements:
                    t = await el.inner_text()
                    if t.strip():
                        texts.append(t.strip())
                await browser.close()
            return texts

        texts = asyncio.run(_scrape())
        df = pd.DataFrame({
            "text": texts,
            "label": None,
            "source": f"scrape:{url}",
            "collected_at": datetime.now(timezone.utc).isoformat(),
        })

        if output:
            Path(output).parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(output, index=False)

        print(f"[scrape] Scraped {len(df)} items from {url}", file=sys.stderr)
        return df

    # ------------------------------------------------------------------
    # skill: fetch_api
    # ------------------------------------------------------------------

    def fetch_api(self, endpoint: str, params: dict | None = None, text_field: str = "text") -> pd.DataFrame:
        """
        Fetch data from a REST API endpoint.
        text_field: key in the JSON response that contains the text.
        """
        import requests
        from dotenv import load_dotenv
        load_dotenv()

        print(f"[fetch_api] {endpoint} params={params}", file=sys.stderr)
        resp = requests.get(endpoint, params=params or {}, timeout=30)
        resp.raise_for_status()
        data = resp.json()

        # Handle both list and dict responses
        if isinstance(data, dict):
            records = data.get("results", data.get("data", data.get("items", [data])))
        else:
            records = data

        df = pd.json_normalize(records)
        text_col = text_field if text_field in df.columns else df.columns[0]

        result = pd.DataFrame({
            "text": df[text_col].astype(str),
            "label": None,
            "source": f"api:{endpoint}",
            "collected_at": datetime.now(timezone.utc).isoformat(),
        })
        print(f"[fetch_api] Fetched {len(result)} rows", file=sys.stderr)
        return result

    # ------------------------------------------------------------------
    # skill: merge
    # ------------------------------------------------------------------

    def merge(self, sources: list[pd.DataFrame]) -> pd.DataFrame:
        """
        Combine multiple DataFrames into unified schema.
        Missing unified columns are filled with None.
        """
        frames = []
        for df in sources:
            df = df.copy()
            for col in UNIFIED_COLS:
                if col not in df.columns:
                    df[col] = None
            frames.append(df[UNIFIED_COLS])

        unified = pd.concat(frames, ignore_index=True)
        print(f"[merge] Unified {len(unified)} rows from {len(sources)} sources", file=sys.stderr)
        return unified

    # ------------------------------------------------------------------
    # skill: run (facade)
    # ------------------------------------------------------------------

    def run(self, sources: list[dict[str, Any]] | None = None) -> pd.DataFrame:
        """
        Main entry point. Accepts a list of source configs, dispatches each
        to the appropriate skill, merges results.

        Source types:
          {'type': 'hf_dataset', 'name': 'imdb', 'split': 'train'}
          {'type': 'kaggle', 'name': 'user/dataset-name'}
          {'type': 'scrape', 'url': '...', 'selector': '...'}
          {'type': 'api', 'endpoint': '...', 'params': {}, 'text_field': 'text'}
        """
        if sources is None:
            sources = self.config.get("sources", [])

        if not sources:
            raise ValueError("No sources provided. Pass sources= or set them in config.yaml")

        collected = []
        for i, src in enumerate(sources, 1):
            src_type = src.get("type")
            print(f"\n[run] [{i}/{len(sources)}] Processing source type={src_type}...", file=sys.stderr)

            if src_type == "hf_dataset":
                df = self.load_dataset(src["name"], source="hf", split=src.get("split", "train"))
            elif src_type == "kaggle":
                df = self.load_dataset(src["name"], source="kaggle")
            elif src_type == "scrape":
                df = self.scrape(src["url"], src.get("selector", "p"), src.get("output"))
            elif src_type == "api":
                df = self.fetch_api(src["endpoint"], src.get("params"), src.get("text_field", "text"))
            else:
                print(f"[run] Unknown source type '{src_type}', skipping.", file=sys.stderr)
                continue

            collected.append(df)

        if not collected:
            raise RuntimeError("No data collected from any source.")

        unified = self.merge(collected)

        # Save output
        out_path = self.config.get("output", {}).get("path", "data/raw/unified.csv")
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        unified.to_csv(out_path, index=False)
        print(f"\n[run] Saved {len(unified)} rows → {out_path}", file=sys.stderr)
        return unified


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="DataCollectionAgent")
    parser.add_argument("--config", default="config_annotation.yaml")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    agent = DataCollectionAgent(config=args.config)
    df = agent.run()

    if args.output:
        df.to_csv(args.output, index=False)

    print(f"\nShape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(df.head(3).to_string())


if __name__ == "__main__":
    main()
