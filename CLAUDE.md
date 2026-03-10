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
| dataset-searcher | `.claude/agents/dataset-searcher.md` | User asks to find/search datasets |

## Skills Registry

| Skill | File | Used by |
|-------|------|---------|
| Dataset Search | `skills/dataset_search.md` | dataset-searcher agent |

## General Rules

- Always use `uv run` to execute Python scripts, never plain `python`
- Always load `.env` via `python-dotenv` — never hardcode API keys
- Scripts output JSON to files in `data/`; progress goes to stderr
- When the user asks to find datasets — launch the `dataset-searcher` agent
- All scripts accept `--help` flag for usage info
