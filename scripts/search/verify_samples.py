"""
verify_samples.py — Stage 5 of the search pipeline.

For each finalist dataset, fetch 10-20 sample rows via Dataset Viewer API
and ask LLM to verify: language, number of classes, task type match the goal.

Datasets that fail verification are moved to rejected_ids.txt.
If fewer than --min-pass datasets pass, exits with code 2 so run_wave.py
can trigger a new search wave automatically.

Usage:
    uv run scripts/search/verify_samples.py \
        --input data/filtered_results_wave1.json \
        --goal "text classification, Russian or English, 2-4 classes, 1k-10k rows" \
        --output data/verified_results_wave1.json \
        --rejected-ids-file data/rejected_ids.txt \
        --min-pass 3
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import requests
from dotenv import load_dotenv

load_dotenv()
os.environ.setdefault("HF_TOKEN", os.getenv("HF_TOKEN", ""))
os.environ.setdefault("KAGGLE_USERNAME", os.getenv("KAGGLE_USERNAME", ""))
os.environ.setdefault("KAGGLE_KEY", os.getenv("KAGGLE_KEY", ""))

HF_TOKEN = os.getenv("HF_TOKEN", "")
HF_HEADERS = {"Authorization": f"Bearer {HF_TOKEN}"} if HF_TOKEN else {}

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "qwen/qwen3-30b-a3b")
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"


# ---------------------------------------------------------------------------
# Sample fetching
# ---------------------------------------------------------------------------

def fetch_hf_samples(dataset_id: str, n: int = 15) -> list[dict] | None:
    """Fetch up to n rows from first available split via Dataset Viewer API."""
    # Get splits
    try:
        r = requests.get(
            f"https://datasets-server.huggingface.co/splits?dataset={dataset_id}",
            headers=HF_HEADERS, timeout=10,
        )
        if not r.ok:
            return None
        splits = r.json().get("splits", [])
        if not splits:
            return None
        split = splits[0]["split"]
        config = splits[0]["config"]
    except Exception:
        return None

    # Fetch rows
    try:
        r = requests.get(
            f"https://datasets-server.huggingface.co/rows"
            f"?dataset={dataset_id}&config={config}&split={split}&offset=0&limit={n}",
            headers=HF_HEADERS, timeout=15,
        )
        if not r.ok:
            return None
        rows = [row["row"] for row in r.json().get("rows", [])]
        return rows if rows else None
    except Exception:
        return None


def fetch_kaggle_samples(dataset_ref: str, n: int = 10) -> list[dict] | None:
    """Download first CSV file from a Kaggle dataset and return first n rows as dicts."""
    try:
        import kaggle
        import tempfile, csv

        kaggle.api.authenticate()
        owner, name = dataset_ref.split("/", 1)

        with tempfile.TemporaryDirectory() as tmpdir:
            kaggle.api.dataset_download_files(
                f"{owner}/{name}", path=tmpdir, unzip=True, quiet=True
            )
            csv_files = list(Path(tmpdir).rglob("*.csv"))
            if not csv_files:
                return None
            with open(csv_files[0], encoding="utf-8", errors="replace") as f:
                reader = csv.DictReader(f)
                rows = []
                for i, row in enumerate(reader):
                    if i >= n:
                        break
                    rows.append(dict(row))
            return rows if rows else None
    except Exception:
        return None


# ---------------------------------------------------------------------------
# LLM verification
# ---------------------------------------------------------------------------

VERIFY_PROMPT = """\
You are a dataset quality verifier. Given a user's goal and sample rows from a dataset, determine if the dataset actually matches the goal.

USER GOAL:
{goal}

DATASET: {dataset_id}
SAMPLE ROWS (first {n} rows):
{samples}

Answer ONLY with valid JSON:
{{
  "pass": true or false,
  "language": "detected language(s) of the text",
  "num_classes": "number of label classes found in sample, or 'unknown'",
  "reason": "1-2 sentences: why it passes or fails"
}}

Be strict: if language doesn't match goal, fail it. If task type is wrong, fail it.
"""


def verify_with_llm(dataset_id: str, goal: str, samples: list[dict]) -> dict:
    samples_str = json.dumps(samples[:15], ensure_ascii=False, indent=2)
    prompt = VERIFY_PROMPT.format(
        goal=goal,
        dataset_id=dataset_id,
        n=len(samples[:15]),
        samples=samples_str[:3000],  # truncate to avoid token overflow
    )

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": OPENROUTER_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "response_format": {"type": "json_object"},
    }

    try:
        r = requests.post(OPENROUTER_URL, headers=headers, json=payload, timeout=30)
        r.raise_for_status()
        content = r.json()["choices"][0]["message"]["content"]
        # Strip <think>...</think> if present
        if "<think>" in content:
            content = content[content.rfind("</think>") + 8:].strip()
        return json.loads(content)
    except Exception as e:
        return {"pass": None, "language": "unknown", "num_classes": "unknown", "reason": f"LLM error: {e}"}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Verify finalist datasets by sampling real rows")
    parser.add_argument("--input", required=True, help="filtered_results_waveN.json")
    parser.add_argument("--goal", required=True, help="Goal description (same as used in search)")
    parser.add_argument("--output", required=True, help="verified_results_waveN.json")
    parser.add_argument("--rejected-ids-file", default="data/rejected_ids.txt")
    parser.add_argument("--min-pass", type=int, default=2,
                        help="Minimum datasets that must pass (exit code 2 if fewer)")
    parser.add_argument("--sample-size", type=int, default=5,
                        help="Number of rows to fetch per dataset")
    args = parser.parse_args()

    datasets = json.loads(Path(args.input).read_text())
    print(f"\n{'='*60}")
    print(f"Sample verification: {len(datasets)} datasets")
    print(f"Goal: {args.goal}")
    print(f"{'='*60}\n")

    rejected_file = Path(args.rejected_ids_file)
    rejected_ids: set[str] = set()
    if rejected_file.exists():
        rejected_ids = set(rejected_file.read_text().splitlines())

    passed = []
    failed = []

    for ds in datasets:
        ds_id: str = ds.get("id", ds.get("dataset_id", ""))
        is_kaggle = ds_id.startswith("kaggle:")
        display_id = ds_id.replace("kaggle:", "")

        print(f"  Checking: {ds_id}")

        # Fetch samples
        if is_kaggle:
            samples = fetch_kaggle_samples(display_id, n=args.sample_size)
        else:
            samples = fetch_hf_samples(ds_id, n=args.sample_size)

        if not samples:
            print(f"    ⚠ Could not fetch samples — skipping (keeping as candidate)")
            ds["verification"] = {"pass": None, "reason": "Could not fetch samples", "language": "unknown", "num_classes": "unknown"}
            passed.append(ds)
            continue

        # Verify with LLM
        result = verify_with_llm(ds_id, args.goal, samples)
        ds["verification"] = result

        status = result.get("pass")
        lang = result.get("language", "?")
        classes = result.get("num_classes", "?")
        reason = result.get("reason", "")

        if status is True:
            print(f"    ✓ PASS | lang={lang} | classes={classes} | {reason}")
            passed.append(ds)
        elif status is False:
            print(f"    ✗ FAIL | lang={lang} | classes={classes} | {reason}")
            failed.append(ds)
            rejected_ids.add(ds_id)
        else:
            print(f"    ? UNCERTAIN | {reason} — keeping as candidate")
            passed.append(ds)

        time.sleep(0.3)

    # Save results
    Path(args.output).write_text(json.dumps(passed, indent=2, ensure_ascii=False))
    rejected_file.write_text("\n".join(sorted(rejected_ids)))

    print(f"\n{'='*60}")
    print(f"Verification complete:")
    print(f"  Passed : {len(passed)}")
    print(f"  Failed : {len(failed)}")
    if failed:
        print(f"  Rejected: {', '.join(d.get('id', '') for d in failed)}")
    print(f"  Output : {args.output}")
    print(f"{'='*60}")

    if len(passed) < args.min_pass:
        print(f"\n⚠ Only {len(passed)} datasets passed (min={args.min_pass}). Need new search wave.", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
