"""
LLM skill — Claude API explains data quality issues and recommends a cleaning strategy.
Reads a QualityReport JSON, sends to Claude, returns Markdown explanation + strategy dict.

Usage:
    uv run scripts/quality/explain.py \
        --report data/quality_report.json \
        --task "binary classification of customer churn, tabular data, XGBoost model"

    uv run scripts/quality/explain.py \
        --report data/quality_report.json \
        --task "image classification of car brands, CNN model" \
        --model claude-haiku-4-5-20251001 \
        --output data/strategy.json
"""

import argparse
import json
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()


def explain(report: dict, task: str, model: str) -> dict:
    try:
        import anthropic
    except ImportError:
        print("anthropic required: uv add anthropic", file=sys.stderr); sys.exit(1)

    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        print("Error: ANTHROPIC_API_KEY not set in .env", file=sys.stderr); sys.exit(1)

    client = anthropic.Anthropic(api_key=api_key)

    report_str = json.dumps(report, ensure_ascii=False, indent=2)

    prompt = f"""You are a senior ML data engineer. A data quality agent has analyzed a dataset and produced this report:

```json
{report_str}
```

The ML task this dataset will be used for:
{task}

Your job:
1. **Explain** each detected issue in plain language — what it means, why it matters for this specific ML task
2. **Assess severity** — which issues will most impact model performance
3. **Recommend a cleaning strategy** — for each issue type, recommend the best method and explain why it suits this task
4. **Return** a strategy dict that can be passed directly to the fix() method

Format your response as JSON with two fields:
{{
  "explanation": "Markdown text — explain issues and reasoning",
  "strategy": {{
    "missing": "method_name",
    "duplicates": "method_name",
    "outliers": "method_name",
    "imbalance": "method_name"
  }},
  "priority_issues": ["issue1", "issue2"],
  "warnings": ["any caveats or trade-offs to be aware of"]
}}

Only include issue types that are present in the report. Be specific and practical."""

    message = client.messages.create(
        model=model,
        max_tokens=2000,
        messages=[{"role": "user", "content": prompt}],
    )

    content = message.content[0].text.strip()

    # Parse JSON response
    if "```" in content:
        for part in content.split("```"):
            part = part.strip()
            if part.startswith("json"):
                part = part[4:].strip()
            if part.startswith("{"):
                content = part
                break

    try:
        return json.loads(content)
    except json.JSONDecodeError:
        # Return raw if parsing fails
        return {"explanation": content, "strategy": {}, "parse_error": True}


def main():
    parser = argparse.ArgumentParser(description="LLM skill: explain quality issues and recommend strategy")
    parser.add_argument("--report", required=True, help="Path to QualityReport JSON")
    parser.add_argument("--task", required=True, help="Description of the ML task")
    parser.add_argument("--model", default="claude-haiku-4-5-20251001",
                        help="Claude model to use (default: claude-haiku-4-5-20251001)")
    parser.add_argument("--output", default="", help="Save result JSON to this path")
    args = parser.parse_args()

    report = json.loads(Path(args.report).read_text(encoding="utf-8"))

    print(f"Asking {args.model} to analyze quality report...", file=sys.stderr)
    result = explain(report, args.task, args.model)

    # Print explanation as markdown
    if "explanation" in result:
        print("\n" + "="*60)
        print(result["explanation"])
        print("="*60)

    if "strategy" in result:
        print("\nRecommended strategy:")
        print(json.dumps(result["strategy"], indent=2))

    if "warnings" in result:
        print("\nWarnings:")
        for w in result["warnings"]:
            print(f"  ⚠️  {w}")

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(result, ensure_ascii=False, indent=2))
        print(f"\nSaved to {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
