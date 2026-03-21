# Skill: fetch_api(endpoint, params)

## Purpose
Fetch data from a REST API and save the result as CSV to `data/raw/`.

---

## When to use
- Second data source has a public REST API
- User provides an endpoint or API name

---

## Steps

### 1. Write collector to `scripts/collect/<name>.py`

```python
"""
Fetch data from [API name].
Usage: uv run scripts/collect/<name>.py --output data/raw/<name>.csv
"""
import argparse, requests
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()

def fetch(endpoint: str, params: dict) -> pd.DataFrame:
    resp = requests.get(endpoint, params=params)
    resp.raise_for_status()
    data = resp.json()
    return pd.json_normalize(data)  # flatten nested JSON

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    df = fetch("https://api.example.com/data", params={})
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"Fetched {len(df)} rows → {args.output}", file=__import__("sys").stderr)

if __name__ == "__main__":
    main()
```

### 2. Run
```bash
uv run scripts/collect/<name>.py --output data/raw/<name>.csv
```

---

## Rules
- Always load `.env` via `python-dotenv` — never hardcode API keys
- Use `pd.json_normalize()` for nested responses
- Handle pagination: loop while `next_page` exists in response
- Save to `data/raw/` as CSV
- Print progress to stderr
