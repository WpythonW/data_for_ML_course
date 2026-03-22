---
name: data-pipeline
description: Full ML pipeline — data collection → cleaning → annotation → active learning. Runs all stages automatically with human-in-the-loop checkpoints.
---

# Data Pipeline

Meta-skill combining all 4 stages into a single ML dataset preparation pipeline.

## Usage

```
/data-pipeline <topic> --classes "class1,class2,class3" --task "task description"
```

Examples:
```
/data-pipeline "product reviews" --classes "positive,negative,neutral" --task "sentiment classification"
/data-pipeline "news" --classes "politics,sports,technology" --task "topic classification"
```

---

## Data Flow

```
┌──────────────────┐    ┌──────────────────┐    ┌──────────────────┐    ┌──────────────────┐
│  1. Collector    │    │  2. Detective    │    │  3. Annotator    │    │  4. ActiveLearn  │
│  Data collection │───▶│  Data cleaning   │───▶│  Auto-labeling   │───▶│  Optimization    │
└──────────────────┘    └──────────────────┘    └──────────────────┘    └──────────────────┘
        │                       │                       │                       │
        ▼                       ▼                       ▼                       ▼
data/raw/unified.csv   data/cleaned.csv        data/labeled.csv        data/learning_curve.png
data/eda/REPORT.md     data/comparison.json    data/labelstudio.json   data/al_savings.json
```

---

## Stage 0 — Setup

Check `.env` keys:
```bash
uv run scripts/search/check_env.py
```

Create required directories:
```bash
mkdir -p data/raw data/eda
```

---

## Stage 1 — Dataset Collector

**Goal:** collect data from at least 2 sources and merge into a unified schema.

**Actions:**
1. Search HuggingFace and Kaggle — run a search wave:
```bash
uv run scripts/search/run_wave.py \
    --wave 1 \
    --queries "<topic>, <topic> dataset, <topic> classification" \
    --keywords "<topic>, label, text" \
    --goal "<task description>"
```
2. Find a second source — a website to scrape or a public API (use WebSearch)
3. Write a scraper or API collector (see `skills/scrape.md` or `skills/fetch_api.md`)

> **⚠️ Every inline script must load keys before importing HF / Kaggle / OpenRouter:**
> ```python
> from dotenv import load_dotenv; load_dotenv()
> import os
> os.environ.setdefault('HF_TOKEN', os.getenv('HF_TOKEN', ''))
> os.environ.setdefault('KAGGLE_USERNAME', os.getenv('KAGGLE_USERNAME', ''))
> os.environ.setdefault('KAGGLE_KEY', os.getenv('KAGGLE_KEY', ''))
> ```
> Without this, libraries fall back to unauthenticated mode even when keys are in `.env`.

> **⚠️ Never use `load_dataset()` to count rows — it downloads the full dataset.**
> Use it only when you actually need the data.
>
> **HuggingFace — размер без скачивания** (Dataset Viewer `/size` endpoint):
> ```python
> import requests
> token = os.getenv('HF_TOKEN', '')
> headers = {"Authorization": f"Bearer {token}"} if token else {}
>
> def hf_dataset_size(dataset_id: str) -> dict:
>     url = f"https://datasets-server.huggingface.co/size?dataset={dataset_id}"
>     r = requests.get(url, headers=headers, timeout=10)
>     if not r.ok:
>         return {"num_rows": None, "size_mb": None}
>     stats = r.json().get("size", {}).get("dataset", {})
>     return {
>         "num_rows": stats.get("num_rows"),          # ✅ точное число строк
>         "size_mb": round(stats.get("num_bytes_original_files", 0) / 1024**2, 2),  # ✅
>     }
> # Если "partial": true в ответе — датасет слишком большой, num_rows неполный
> ```
>
> **⚠️ HuggingFace — загрузка данных: используй `load_dataset()` или Parquet, НЕ Viewer `/rows`**
>
> Viewer API (`/rows?offset=...&length=100`) пагинирует по 100 строк и попадает в Hub API
> rate-limit (1 000 req/5 мин). Для 10 000 строк это 100 запросов — гарантированный 429.
>
> ✅ **Метод 1 — `load_dataset`** (рекомендуется, скачивает всё одним запросом):
> ```python
> from dotenv import load_dotenv; load_dotenv()
> import os
> os.environ.setdefault('HF_TOKEN', os.getenv('HF_TOKEN', ''))
> from datasets import load_dataset
>
> ds = load_dataset("dataset/name", token=os.environ.get('HF_TOKEN'))
> # ds["train"], ds["test"], ds["validation"]
> ```
>
> ✅ **Метод 2 — прямой Parquet** (если нужен pandas без `datasets`):
> ```python
> import requests, pandas as pd, io
> TOKEN = os.getenv('HF_TOKEN', '')
> headers = {'Authorization': f'Bearer {TOKEN}'}
>
> # 1. Получить URL файлов
> urls = requests.get(
>     'https://huggingface.co/api/datasets/<owner>/<name>/parquet',
>     headers=headers,
> ).json()
> # urls["default"]["train"] → список URL parquet-файлов
>
> # 2. Скачать через requests + BytesIO (НЕ через storage_options — не работает)
> r = requests.get(urls["default"]["train"][0], headers=headers, timeout=60)
> df = pd.read_parquet(io.BytesIO(r.content))
> ```
>
> ❌ **Никогда не используй Viewer `/rows` для загрузки данных** — только для превью.
>
> **Kaggle** — `num_rows` API **не отдаёт совсем**. Размер через `_total_bytes`:
> ```python
> import kaggle
> kaggle.api.authenticate()
>
> def kaggle_dataset_size(owner: str, name: str) -> dict:
>     # ВАЖНО: search=f"{owner}/{name}" — полный ref, не только name!
>     results = kaggle.api.dataset_list(search=f"{owner}/{name}")
>     match = next((d for d in results if d.ref == f"{owner}/{name}"), None)
>     if not match:
>         return {"num_rows": None, "size_mb": None}
>     total_bytes = match._total_bytes or 0
>     return {
>         "num_rows": None,   # ❌ Kaggle API не возвращает num_rows никогда
>         "size_mb": round(total_bytes / 1024**2, 2),  # ✅
>     }
> ```
> Если `num_rows` недоступен — пиши `?` в таблице источников. **Не скачивай ради счётчика.**

4. Merge sources via `skills/merge_sources.md` → `data/raw/unified.csv`
5. Run EDA

**⏸ HUMAN CHECKPOINT #1:**
```
## Sources Found

### Datasets (HuggingFace / Kaggle):
> **Show ALL candidates from the search results — minimum 7 datasets. Do NOT pre-filter or pick for the user.**

| # | Name | Rows | Size (MB) | License | Why it fits |
|---|------|------|-----------|---------|-------------|
| 1 | [name](url) | 4,864 | 0.3 MB | CC-BY | ... |

> **Reliability rules:**
> - Rows from HF Dataset Viewer API → exact, no qualifier needed.
> - Rows from Kaggle API → never available; write `? (unreliable)`.
> - Size from HF Viewer → exact. Size from Kaggle metadata → add `(~approx)`.
> - If `"partial": true` in HF response → rows are partial, write `~N (partial)`.
> - **Always include a clickable link to the dataset.**
> - Never leave Rows or Size as bare `?` — explain WHY it's unknown.

### Second source:
| # | Type | URL / Name | ~Rows |
|---|------|------------|-------|
| 2 | scrape/api | [name](url) | ~N or ? (unreliable) |

Confirm these sources? Or replace something? [yes / specify replacement]
```

**After confirmation:**
- Download / collect data
- Apply unified schema: `text, label, source, collected_at`
- Save to `data/raw/unified.csv`
- Run EDA → `data/eda/REPORT.md`

---

## Stage 2 — Data Detective

**Goal:** find and fix data quality issues.

**Input:** `data/raw/unified.csv`

**Actions — run all detectors:**
```bash
uv run scripts/quality/profile.py --input data/raw/unified.csv
uv run scripts/quality/detect_missing.py --input data/raw/unified.csv --output data/missing.json
uv run scripts/quality/detect_duplicates.py --input data/raw/unified.csv --output data/duplicates.json
uv run scripts/quality/detect_outliers.py --input data/raw/unified.csv --method iqr --output data/outliers.json
uv run scripts/quality/detect_imbalance.py --input data/raw/unified.csv --label label --output data/imbalance.json
```

**⏸ HUMAN CHECKPOINT #2:**
```
## Data Quality Issues Found

| Issue | Count | % | Severity |
|-------|-------|---|----------|
| Missing values | ... | | low/medium/high |
| Duplicates | ... | | |
| Outliers | ... | | |
| Imbalance | ... | | |

### Available strategies:
- **aggressive** — drop missing, duplicates, outliers (IQR). Less data, cleaner.
- **conservative** — fill missing, keep outliers. More data.
- **balanced** — drop duplicates, fill missing with median, trim extreme outliers (z>3). Recommended.

Which strategy to apply? [aggressive / conservative / balanced]
```

**After confirmation:**
```bash
uv run scripts/quality/fix_duplicates.py --input data/raw/unified.csv --keep first --output data/step1.csv
uv run scripts/quality/fix_missing.py --input data/step1.csv --strategy <choice> --output data/step2.csv
uv run scripts/quality/fix_outliers.py --input data/step2.csv --strategy <choice> --output data/cleaned.csv
uv run scripts/quality/compare.py --before data/raw/unified.csv --after data/cleaned.csv --output data/comparison.json
```
- Show before/after comparison report

---

## Stage 3 — Annotation Agent

**Goal:** auto-label data, generate annotation spec, export to LabelStudio.

**Input:** `data/cleaned.csv`

**⏸ HUMAN CHECKPOINT #3:**
```
## Auto-labeling Settings

- File: data/cleaned.csv
- Rows: N
- Classes: <classes>
- Task: <task>
- Model: cross-encoder/nli-MiniLM2-L6-H768 (~120MB)
- Confidence threshold: 0.75 (below → manual review)

Proceed with labeling? [yes / change classes]
```

**After confirmation:**
```bash
uv run agents/annotation_agent.py \
    --input data/cleaned.csv \
    --task "<task>" \
    --labels "<classes>" \
    --confidence-threshold 0.75 \
    --output-dir data
```

- Show label distribution and confidence scores
- Share `data/annotation_spec.md` for manual review of a sample
- Export to LabelStudio: `data/labelstudio_import.json`
- Low-confidence flags: `data/low_confidence.csv`

---

## Stage 4 — Active Learner

**Goal:** show how many labels can be saved with entropy vs random sampling.

**Input:** `data/labeled.csv`

**⏸ HUMAN CHECKPOINT #4:**
```
## Active Learning Settings

- Seed set: 50 examples
- Iterations: 5 × 20 examples = 100 additional labels
- Strategies: entropy (smart) vs random (baseline)
- Model: LogisticRegression + TF-IDF

Proceed? [yes / change parameters]
```

**After confirmation:**
```bash
uv run agents/al_agent.py \
    --input data/labeled.csv \
    --strategy entropy \
    --n-start 50 \
    --n-iterations 5 \
    --batch-size 20 \
    --compare \
    --output-dir data
```

- Show learning curves (entropy vs random on one plot)
- Show savings: how many labels were saved

---

## Final Report

After all stages complete, show summary:

```
## Pipeline Complete

### Stage 1: Dataset Collector
- Sources: HF:<name> + scrape/api:<url>
- Rows collected: N
- File: data/raw/unified.csv
- EDA: data/eda/REPORT.md

### Stage 2: Data Detective
- Strategy: <chosen>
- Before: N rows → After: M rows (-X%)
- File: data/cleaned.csv

### Stage 3: Annotation Agent
- Labeled: M rows
- Avg confidence: X
- Flagged for manual review: K rows
- Spec: data/annotation_spec.md
- LabelStudio: data/labelstudio_import.json

### Stage 4: Active Learner
- Entropy F1: X at 150 labels
- Random F1: Y at 150 labels
- Saved: N labels (P%) vs random baseline
- Plot: data/learning_curve.png

### Ready ML files:
- data/labeled.csv — labeled dataset
- data/annotation_spec.md — class specification
- data/learning_curve.png — learning curves
```

---

## Rules

1. Follow stages strictly: 1 → 2 → 3 → 4
2. Wait for user confirmation at each checkpoint
3. Pass data automatically: output of stage N = input of stage N+1
4. Show progress: `[Stage X/4] Name...`
5. On error — show the problem and suggest a fix, do not abort the pipeline
6. Always use `uv run`, never call `python` directly
7. **All datasets merged into `unified.csv` MUST share the same domain and task.** Never mix datasets from different tasks (e.g. topic classification + sentiment analysis). If sources cover different domains, pick ONE or ask the user to choose before merging.
8. **`load_dotenv()` in `/tmp/` scripts requires an absolute path:** `load_dotenv('/Users/andrejustinov/Desktop/Data_for_ML/.env')`. Without it, keys are empty and HF/Kaggle calls fail with auth errors.
