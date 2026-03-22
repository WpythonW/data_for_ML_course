# Skill: Dataset Search

## Purpose
Find the best datasets for a given ML task using a multi-stage pipeline:
generate many queries + keywords → bulk search → BM25 ranking → LLM filtering.

---

## Pipeline Overview

```
User goal (natural language)
        ↓
[0] Clarify with user (ask questions)
        ↓
[1] Generate 30-100 search queries + 30-100 expanded keywords
        ↓
[2] hf_bulk_search.py — fetch raw candidates with rich metadata
        ↓  hundreds of datasets with descriptions, tags, card text
[3] semantic_filter.py — BM25 (queries+keywords) → LLM ranking
        ↓  top-N final datasets with reasons
[4] verify_samples.py — fetch real rows → LLM checks language/classes/task
        ↓  datasets that fail → rejected_ids.txt → auto re-search if too few pass
[5] Evaluate → search more if needed → present to user
```

---

## Step 0 — Clarify with the User

Before generating anything, ask the user about their requirements.
**No fixed list** — adapt questions to the domain. Typical areas to cover:

- ML task type (classification / detection / segmentation / NLP / etc.)
- Data modality (images: CT/MRI/histology, text, tabular, audio, video)
- Domain specifics (cancer type, language, industry, etc.)
- Size requirements (min samples, min GB)
- Annotation requirements (labels only / bounding boxes / segmentation masks)
- License constraints (open / research-use / commercial)
- Known exclusions (datasets already owned)
- Benchmark interest (well-known challenges vs. niche datasets)

Ask follow-up questions if answers are vague. Collect enough to generate precise queries.

---

## Step 1 — Generate Queries AND Keywords

Generate **two separate lists**:

### Search Queries (30-100)
Short phrases for the HuggingFace search API. Think in dimensions:
- Synonyms: different names for the same concept
- Subtopics: specific modalities, organs, conditions, tasks
- Benchmark names: known dataset names, competitions, challenges
- Adjacent domains: related fields with transferable data
- Technical terms: clinical/scientific/engineering terminology
- Combinations: task + modality + domain (e.g. "brain tumor MRI segmentation")

### Expanded Keywords (30-100)
Individual terms and short phrases for BM25 scoring. These extend search coverage:
- All terms from queries (broken apart)
- Abbreviations and acronyms (MRI, FLAIR, WHO, GT, ROI, WSI...)
- File formats relevant to domain (DICOM, NIfTI, .nii.gz, HDF5...)
- Annotation terminology (mask, contour, ground truth, annotation, label...)
- Quality signals (benchmark, curated, challenge, MICCAI, peer-reviewed...)
- Negative signals to penalize if needed (synthetic, toy, small, demo...)

Save both lists — you will pass them to the scripts.

---

## Step 2 — Run Bulk Search

```bash
uv run scripts/search/hf_bulk_search.py \
    --queries "query1,query2,query3,..." \
    --limit-per-query 100 \
    --output data/raw_results.json
```

- **Never use `--fetch-cards`** — extremely slow without HF_TOKEN, avoid always
- `--limit-per-query 100` is the default; use 50 if you have 80+ queries (API load)
- Check output: print total unique count, how many matched 3+ queries

---

## Step 3 — Run filtering pipeline

The pipeline runs 6 internal stages automatically. You only set `--bm25-top`.

**Setting --bm25-top** (be generous — Haiku narrows the bottleneck):
- < 100 raw results → 30
- 100–300 → 50
- 300–600 → 60–80
- > 600 → 80–100

```bash
uv run scripts/search/semantic_filter.py \
    --input data/raw_results_wave1.json \
    --goal "RICH NATURAL LANGUAGE DESCRIPTION — be specific about task, modality, domain, annotations" \
    --queries "query1,query2,..." \
    --keywords "keyword1,keyword2,..." \
    --bm25-top <N> \
    --rejected-ids-file data/rejected_ids.txt \
    --output data/filtered_results_wave1.json
```

The `--goal` must be a rich descriptive sentence, not just keywords.
Example: `"car brand/make image classification dataset with labeled photos of Toyota, BMW, Ford etc"`

Always pass `--rejected-ids-file data/rejected_ids.txt` — rejected IDs persist across waves.
Do NOT pass `--llm-top` — Haiku decides the final count itself.

---

## Step 4 — Verify with Real Samples

After filtering, run sample verification to catch mismatches (wrong language, wrong task, etc.) that metadata alone misses:

```bash
uv run scripts/search/verify_samples.py \
    --input data/filtered_results_wave1.json \
    --goal "text classification, Russian or English, 2-4 classes, 1k-10k rows" \
    --output data/verified_results_wave1.json \
    --rejected-ids-file data/rejected_ids.txt \
    --min-pass 2 \
    --sample-size 5
```

Or automatically as part of `run_wave.py` by adding `--verify`:

```bash
uv run scripts/search/run_wave.py \
    --wave 1 \
    --queries "..." \
    --keywords "..." \
    --goal "..." \
    --verify \
    --verify-min-pass 2
```

**What it checks per dataset:**
- Fetches 15 real rows via Dataset Viewer API (HF) or download (Kaggle)
- Asks LLM: does language / num_classes / task match the goal?
- `PASS` → kept in output | `FAIL` → added to `rejected_ids.txt`
- If fewer than `--min-pass` pass → exit code 2 → `run_wave.py` warns to run next wave

> **Why this matters:** metadata cards are often empty or misleading (e.g. dataset titled "news classification" but text is Indonesian, not English).

---

## Step 5 — Evaluate and Decide

After filtering, read `data/filtered_results.json` and evaluate:

**Ask yourself:**
- Are results on-topic?
- Do they match modality, task, and annotation requirements?
- Are there obvious gaps (wrong organ, wrong task, wrong modality)?
- Are there enough candidates (at least 5 good ones)?

**If insufficient → generate more queries (with deduplication):**
- Identify what was missing from current results
- Generate additional queries targeting those gaps
- Read `data/seen_ids.txt` — pass its contents to `--exclude-ids` in BOTH scripts
- Re-run bulk search with NEW queries only, excluding already-seen IDs
- After the new search, append new IDs to `data/seen_ids.txt`
- Re-run filter pipeline on new results (also with `--exclude-ids`)

**Rule: never show the LLM the same dataset twice. `data/seen_ids.txt` is the ground truth.**

**If sufficient → present to user.**

---

## Step 5b — Fetch Row Count and Size WITHOUT Downloading

**Never use `load_dataset()` just to count rows — it downloads the full dataset.**

Use the HuggingFace Dataset Viewer `/size` endpoint instead:

```python
from dotenv import load_dotenv; load_dotenv()
import os, requests

os.environ.setdefault('HF_TOKEN', os.getenv('HF_TOKEN', ''))
token = os.getenv('HF_TOKEN', '')
headers = {"Authorization": f"Bearer {token}"} if token else {}

def hf_dataset_size(dataset_id: str) -> dict:
    url = f"https://datasets-server.huggingface.co/size?dataset={dataset_id}"
    r = requests.get(url, headers=headers, timeout=10)
    if not r.ok:
        return {"num_rows": None, "size_mb": None}
    data = r.json()
    partial = data.get("partial", False)
    stats = data.get("size", {}).get("dataset", {})
    rows = stats.get("num_rows")
    return {
        "num_rows": f"~{rows} (partial)" if partial else rows,
        "size_mb": round(stats.get("num_bytes_parquet_files", 0) / 1024**2, 2),
    }
```

- Works for any dataset compatible with the Dataset Viewer (supports parquet export)
- If response contains `"partial": true` — dataset is too large to fully index; rows are approximate
- For Kaggle datasets: use `kaggle.api.dataset_list(search=...)` — returns `_total_bytes`, no row count

---

## Step 5c — Download Dataset Data (when you need actual rows)

**⚠️ Never use Viewer API (`/rows?offset=...`) to download data — it paginates by 100 rows,
hits the Hub API rate-limit (1 000 req/5 min), and causes 429 errors for any dataset > 1k rows.**

✅ **Method 1 — `load_dataset`** (recommended, single request, auto-retry on 429):
```python
from dotenv import load_dotenv; load_dotenv()
import os
os.environ.setdefault('HF_TOKEN', os.getenv('HF_TOKEN', ''))
from datasets import load_dataset

ds = load_dataset("owner/name", token=os.environ.get('HF_TOKEN'))
# ds["train"], ds["test"], ds["validation"]
```

✅ **Method 2 — direct Parquet** (if you need pandas without the `datasets` library):
```python
import requests, pandas as pd, io
TOKEN = os.getenv('HF_TOKEN', '')
headers = {'Authorization': f'Bearer {TOKEN}'}

# Step 1: get parquet file URLs
urls = requests.get(
    'https://huggingface.co/api/datasets/<owner>/<name>/parquet',
    headers=headers,
).json()
# urls["default"]["train"] → list of parquet URLs

# Step 2: download via requests + BytesIO (do NOT use storage_options — broken in pandas)
r = requests.get(urls["default"]["train"][0], headers=headers, timeout=60)
df = pd.read_parquet(io.BytesIO(r.content))
```

❌ **Never use Viewer `/rows` to download data** — for preview only.

---

## Step 6 — Present Results

```
## Top Datasets Found

1. **[dataset-name](url)** — Relevance: 9/10
   Downloads: 12,500 | License: CC-BY-4.0 | Rows: 5,000 | Size: 1.2 MB
   Why it fits: <llm_reason>
```

**Reliability rules for Rows and Size — ALWAYS apply:**
- HF Dataset Viewer API returns exact rows → show as-is: `5,000`
- HF Viewer with `"partial": true` → show as: `~5,000 (partial)`
- Kaggle API never returns rows → show as: `? (unreliable — Kaggle API)`
- Size from HF Viewer (parquet bytes) → exact, no qualifier
- Size from Kaggle `_total_bytes` metadata → show as: `~37 MB (approx)`
- **Always include a clickable link. Never bare `?` — explain why unknown.**
- Do NOT use `load_dataset()` to count rows — downloads everything.

Then ask: "Would you like to explore any of these further, search more sources (Kaggle, TCIA), or refine the search?"

---

## File Locations

| File | Description |
|------|-------------|
| `data/raw_results.json` | Raw bulk search output |
| `data/filtered_results.json` | Final ranked results |
| `scripts/search/hf_bulk_search.py` | Bulk search with rich metadata |
| `scripts/search/semantic_filter.py` | BM25 + LLM filter |
| `scripts/search/verify_samples.py` | Fetch real rows + LLM verify language/classes/task |
| `.env` | `HF_TOKEN`, `OPENROUTER_API_KEY`, `OPENROUTER_MODEL` |

---

## Environment Variables

| Key | Required | Default |
|-----|----------|---------|
| `OPENROUTER_API_KEY` | Yes | — |
| `OPENROUTER_MODEL` | No | `qwen/qwen3-30b-a3b` |
| `HF_TOKEN` | No | — (higher rate limits if set) |

---

## Prerequisites Check

```bash
python -c "
from dotenv import load_dotenv; import os; load_dotenv()
key = os.getenv('OPENROUTER_API_KEY')
print('OpenRouter:', 'OK' if key else 'MISSING')
"
```

All packages managed via `uv` — see `pyproject.toml`.
