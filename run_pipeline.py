"""
Full ML pipeline — one command entrypoint.

Usage:
    uv run run_pipeline.py

Note: This project uses Claude Code as the orchestrator.
Run the pipeline interactively with:
    /pipeline <task description>

This script documents the pipeline steps for reproducibility.
"""

print("""
ML Data Pipeline — Steps:

1. Dataset Collector  → data/raw/unified.csv
   uv run scripts/search/run_wave.py --wave 1 --queries "..." --keywords "..." --goal "..."

2. Data Quality       → data/cleaned.csv
   uv run scripts/quality/detect_missing.py --input data/raw/unified.csv --output data/missing.json
   uv run scripts/quality/detect_duplicates.py --input data/raw/unified.csv --output data/duplicates.json
   uv run scripts/quality/detect_outliers.py --input data/raw/unified.csv --output data/outliers.json
   uv run scripts/quality/detect_imbalance.py --input data/raw/unified.csv --label label --output data/imbalance.json
   uv run scripts/quality/fix_duplicates.py --input data/raw/unified.csv --keep first --output data/cleaned.csv

3. Annotation         → data/labeled.csv, data/labelstudio_import.json
   uv run agents/annotation_agent.py --input data/cleaned.csv --labels "human,AI-generated" --confidence-threshold 0.75 --output-dir data

4. Active Learning    → data/learning_curve.png
   uv run agents/al_agent.py --input data/cleaned.csv --strategy entropy --n-start 50 --n-iterations 5 --batch-size 20 --compare --output-dir data

For interactive HITL orchestration, use Claude Code:
    claude .
    /pipeline <task>
""")
