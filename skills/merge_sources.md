# Skill: merge(sources)

## Purpose
Combine multiple DataFrames from different sources into a single unified schema.

---

## Unified schema

```python
UNIFIED_COLS = ["text", "audio", "image", "label", "source", "collected_at"]
```

| Column | Type | Description |
|--------|------|-------------|
| `text` | str or None | Main text content |
| `audio` | str or None | Path to audio file |
| `image` | str or None | Path or URL to image |
| `label` | str or None | Target class, None if unlabeled |
| `source` | str | `"hf:<name>"`, `"kaggle:<name>"`, `"scrape:<url>"`, `"api:<endpoint>"` |
| `collected_at` | str | ISO 8601 timestamp |

---

## Implementation (in `agents/data_collection_agent.py`)

```python
import pandas as pd
from datetime import datetime, timezone

UNIFIED_COLS = ["text", "audio", "image", "label", "source", "collected_at"]

def merge(sources: list[pd.DataFrame]) -> pd.DataFrame:
    frames = []
    for df in sources:
        # add missing unified columns as None
        for col in UNIFIED_COLS:
            if col not in df.columns:
                df[col] = None
        frames.append(df[UNIFIED_COLS])
    unified = pd.concat(frames, ignore_index=True)
    return unified
```

---

## Rules
- Fill missing unified columns with `None`, never drop rows
- `source` tag must be set before calling merge (e.g. `df['source'] = 'hf:imdb'`)
- `collected_at` must be ISO timestamp: `datetime.now(timezone.utc).isoformat()`
- Save result to `data/raw/unified.csv`
