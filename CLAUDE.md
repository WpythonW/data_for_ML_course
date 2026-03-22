# Dataset Pipeline — Instructions for Claude Code

This project is a collection of skills and scripts for working with ML datasets:
searching, labeling, and exploratory data analysis (EDA).

Claude is the orchestrator. There are no sub-agents — Claude reads skill files directly
and executes them step by step, with human-in-the-loop checkpoints between stages.

## Language

- **Always communicate with the user in Russian.**
- All skill files, prompts, scripts, and technical identifiers remain in English.

## Project Structure

```
dataset-agent/
├── CLAUDE.md                  # This file — skill registry and instructions
├── skills/
│   ├── data_pipeline.md       # Master pipeline prompt (orchestration entry point)
│   ├── dataset_search.md      # Stage 1: search HuggingFace / Kaggle / web
│   ├── scrape.md              # Stage 1: scrape web sources
│   ├── fetch_api.md           # Stage 1: fetch data via REST APIs
│   ├── merge_sources.md       # Stage 1: merge collected sources
│   ├── detect_issues.md       # Stage 2: detect data quality issues
│   ├── fix_data.md            # Stage 2: fix/clean data
│   ├── auto_label.md          # Stage 3: zero-shot auto-labeling
│   ├── check_quality.md       # Stage 3: compute annotation quality metrics
│   ├── export_labelstudio.md  # Stage 3: export to LabelStudio
│   └── active_learning.md     # Stage 4: select informative samples
├── scripts/                   # Python scripts called by skills
├── .env                       # API keys (not committed)
└── .env.example               # Template for API keys
```

## Skills Registry

| Skill | File | Stage |
|-------|------|-------|
| Data Pipeline | `skills/data_pipeline.md` | Entry point — read this first |
| Dataset Search | `skills/dataset_search.md` | Stage 1 |
| Scrape | `skills/scrape.md` | Stage 1 |
| Fetch API | `skills/fetch_api.md` | Stage 1 |
| Merge Sources | `skills/merge_sources.md` | Stage 1 |
| Detect Issues | `skills/detect_issues.md` | Stage 2 |
| Fix Data | `skills/fix_data.md` | Stage 2 |
| Auto Label | `skills/auto_label.md` | Stage 3 |
| Check Quality | `skills/check_quality.md` | Stage 3 |
| Export LabelStudio | `skills/export_labelstudio.md` | Stage 3 |
| Active Learning | `skills/active_learning.md` | Stage 4 |

## Orchestration Rules

- To run the full pipeline: read `skills/data_pipeline.md` and follow its instructions
- Claude executes each skill in sequence, pausing for user approval between stages
- Each skill describes what scripts to run and what output to expect
- **No sub-agents** — Claude is the sole orchestrator

## General Rules

- Always use `uv run` to execute Python scripts, never plain `python`
- **Running Python inline:** use `uv run python -c "..."` — NOT `uv run -c "..."` (unsupported)
- For multi-line inline scripts, write a temporary `_tmp_*.py` file, run with `uv run _tmp_*.py`, then delete it
- Always load `.env` via `python-dotenv` — never hardcode API keys
- **Inline scripts must always start with `load_dotenv()` and propagate keys to `os.environ`** before importing any library that reads env vars (HuggingFace, Kaggle, OpenRouter, etc.):
  ```python
  from dotenv import load_dotenv
  load_dotenv('/Users/andrejustinov/Desktop/Data_for_ML/.env')  # always use absolute path!
  import os
  # HuggingFace
  os.environ.setdefault('HF_TOKEN', os.getenv('HF_TOKEN', ''))
  # Kaggle
  os.environ.setdefault('KAGGLE_USERNAME', os.getenv('KAGGLE_USERNAME', ''))
  os.environ.setdefault('KAGGLE_KEY', os.getenv('KAGGLE_KEY', ''))
  ```
  **⚠️ Always pass the absolute path to `load_dotenv()`** — scripts written to `/tmp/` run outside the project directory, so `load_dotenv()` without a path will silently fail and all keys will be empty strings. This causes `httpx.LocalProtocolError: Illegal header value b'Bearer '` when calling HuggingFace API.
- Scripts output JSON/CSV to files in `data/`; progress goes to stderr
- All scripts accept `--help` flag for usage info
- All configs, prompts, and scripts must live inside the project directory — never write to `~/.claude/` or other user-specific paths
