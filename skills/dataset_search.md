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
[4] Evaluate → search more if needed → present to user
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

## Step 4 — Evaluate and Decide

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

## Step 5 — Present Results

```
## Top Datasets Found

1. **[dataset-name](url)** — Relevance: 9/10 ⚠️ verify
   Downloads: 12,500 | License: CC-BY-4.0 | Size: 10K–100K
   Why it fits: <llm_reason>

2. ...
```

Then ask: "Would you like to explore any of these further, search more sources (Kaggle, TCIA), or refine the search?"

---

## File Locations

| File | Description |
|------|-------------|
| `data/raw_results.json` | Raw bulk search output |
| `data/filtered_results.json` | Final ranked results |
| `scripts/search/hf_bulk_search.py` | Bulk search with rich metadata |
| `scripts/search/semantic_filter.py` | BM25 + LLM filter |
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
