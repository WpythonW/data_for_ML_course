# Dataset Agent — Instructions for Claude Code

This project is a collection of skills and scripts for working with ML datasets:
searching, labeling, and exploratory data analysis (EDA).

## Project Structure

```
dataset-agent/
├── CLAUDE.md                  # This file — skill registry and instructions
├── skills/
│   └── dataset_search.md      # Dataset search skill (Kaggle, HuggingFace, Google)
├── scripts/
│   └── search/
│       ├── huggingface_search.py
│       ├── kaggle_search.py
│       └── google_search.py
├── .env                       # API keys (not committed)
└── .env.example               # Template for API keys
```

## Agents Registry

| Agent | File | Trigger |
|-------|------|---------|
| dataset-searcher | `.claude/agents/dataset-searcher.md` | User asks to find/search/collect datasets |
| data-quality | `.claude/agents/data-quality.md` | User asks to analyze, clean, or audit a dataset |
| annotation-agent | `.claude/agents/annotation-agent.md` | User asks to annotate, label, or mark up a dataset |
| al-agent | `agents/al-agent.md` | User asks about active learning, smart data selection, labeling efficiency |
| data-pipeline | `skills/data_pipeline.md` | User wants to run the full pipeline end-to-end |

## Skills Registry

| Skill | File | Used by |
|-------|------|---------|
| Dataset Search | `skills/dataset_search.md` | dataset-searcher agent |
| Scrape | `skills/scrape.md` | dataset-searcher agent |
| Fetch API | `skills/fetch_api.md` | dataset-searcher agent |
| Merge Sources | `skills/merge_sources.md` | dataset-searcher agent |
| Detect Issues | `skills/detect_issues.md` | data-quality agent |
| Fix Data | `skills/fix_data.md` | data-quality agent |
| Auto Label | `skills/auto_label.md` | annotation-agent |
| Check Quality | `skills/check_quality.md` | annotation-agent |
| Export LabelStudio | `skills/export_labelstudio.md` | annotation-agent |
| Active Learning | `skills/active_learning.md` | al-agent |

## General Rules

- Always use `uv run` to execute Python scripts, never plain `python`
- Always load `.env` via `python-dotenv` — never hardcode API keys
- Scripts output JSON to files in `data/`; progress goes to stderr
- All scripts accept `--help` flag for usage info
- All configs, prompts, and scripts must live inside the project directory — never write to `~/.claude/` or other user-specific paths

## Agent Routing — STRICT

- User asks to find/search/collect datasets → **dataset-searcher** agent
- User asks to analyze/clean/audit data quality → **data-quality** agent
- User asks to annotate/label/markup a dataset → **annotation-agent** agent
- User asks about active learning / smart data selection → **al-agent** agent
- User wants to run the full end-to-end pipeline → **data-pipeline** skill
- **Never use general-purpose agent** as a substitute for the four project agents
