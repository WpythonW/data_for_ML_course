# Skill: scrape(url, selector)

## Purpose
Scrape structured data from a web page using Playwright and return a CSV saved to `data/raw/`.

---

## When to use
- Second data source is a public website (no API available)
- User provides a URL to scrape

---

## Steps

### 1. Write scraper to `scripts/scrape/<sitename>.py`

```python
"""
Scrape [site] for [data].
Usage: uv run scripts/scrape/<sitename>.py --output data/raw/<sitename>.csv
"""
import asyncio, argparse
import pandas as pd
from pathlib import Path
from playwright.async_api import async_playwright

async def scrape(url: str) -> list[dict]:
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()
        await page.goto(url, wait_until="networkidle")
        # site-specific extraction — adapt selector to target elements
        items = await page.query_selector_all(selector)
        results = []
        for item in items:
            results.append({"text": await item.inner_text()})
        await browser.close()
    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    rows = asyncio.run(scrape(args.url))
    df = pd.DataFrame(rows)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"Scraped {len(df)} rows → {args.output}", file=__import__("sys").stderr)

if __name__ == "__main__":
    main()
```

### 2. Install Playwright if needed
```bash
uv add playwright
uv run playwright install chromium
```

### 3. Run
```bash
uv run scripts/scrape/<sitename>.py --url "https://..." --output data/raw/<sitename>.csv
```

---

## Rules
- Always `headless=True`
- Save to `data/raw/` as CSV
- Use `wait_until="networkidle"` or explicit waits
- For paginated sites: loop until no next button or empty page
- Progress to stderr, data to stdout/file
