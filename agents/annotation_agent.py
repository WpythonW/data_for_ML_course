"""
AnnotationAgent — автоматическая разметка, генерация спецификации,
оценка качества и экспорт в LabelStudio.

Usage:
    uv run agents/annotation_agent.py --input data/raw/texts.csv --modality text --task sentiment_classification
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.metrics import cohen_kappa_score


# ---------------------------------------------------------------------------
# AnnotationAgent
# ---------------------------------------------------------------------------

class AnnotationAgent:
    """Main agent: auto_label → generate_spec → check_quality → export_to_labelstudio."""

    def __init__(self, modality: str = "text", confidence_threshold: float = 0.75):
        self.modality = modality
        self.confidence_threshold = confidence_threshold
        self._classifier = None  # lazy-loaded

    # ------------------------------------------------------------------
    # skill: auto_label
    # ------------------------------------------------------------------

    def auto_label(
        self,
        df: pd.DataFrame,
        candidate_labels: list[str] | None = None,
        text_col: str = "text",
    ) -> pd.DataFrame:
        """
        Auto-label DataFrame rows.
        Adds columns: label, confidence, flagged_for_review.
        Supports modality='text' via zero-shot classification (bart-large-mnli).
        """
        if self.modality != "text":
            raise NotImplementedError(f"Modality '{self.modality}' not supported yet. Use 'text'.")

        if text_col not in df.columns:
            raise ValueError(f"Column '{text_col}' not found. Available: {list(df.columns)}")

        if candidate_labels is None:
            candidate_labels = ["positive", "negative", "neutral"]

        print(f"[auto_label] Loading zero-shot model...", file=sys.stderr)
        classifier = self._get_classifier()

        labels_out = []
        scores_out = []

        texts = df[text_col].fillna("").tolist()
        total = len(texts)
        BATCH = 8  # process in batches — much faster than one-by-one

        for batch_start in range(0, total, BATCH):
            batch = texts[batch_start: batch_start + BATCH]
            # replace empty strings so the model doesn't choke
            batch_clean = [t if t.strip() else "." for t in batch]

            results = classifier(batch_clean, candidate_labels=candidate_labels)
            if isinstance(results, dict):  # single-item batch returns dict, not list
                results = [results]

            for orig, res in zip(batch, results):
                if not orig.strip():
                    labels_out.append(None)
                    scores_out.append(0.0)
                else:
                    labels_out.append(res["labels"][0])
                    scores_out.append(round(res["scores"][0], 4))

            done = min(batch_start + BATCH, total)
            print(f"[auto_label] {done}/{total} ({done*100//total}%)", file=sys.stderr)

        df = df.copy()
        df["label"] = labels_out
        df["confidence"] = scores_out
        df["flagged_for_review"] = df["confidence"] < self.confidence_threshold
        df["labeled_at"] = datetime.now(timezone.utc).isoformat()

        flagged_count = df["flagged_for_review"].sum()
        print(
            f"[auto_label] Done. {len(df)} rows labeled. "
            f"{flagged_count} flagged (confidence < {self.confidence_threshold}).",
            file=sys.stderr,
        )
        return df

    # ------------------------------------------------------------------
    # skill: generate_spec
    # ------------------------------------------------------------------

    def generate_spec(
        self,
        df: pd.DataFrame,
        task: str,
        candidate_labels: list[str] | None = None,
        text_col: str = "text",
        output_path: str | Path = "annotation_spec.md",
    ) -> str:
        """
        Generate an annotation specification Markdown file.
        Returns the Markdown string and saves to output_path.
        """
        if candidate_labels is None:
            candidate_labels = ["positive", "negative", "neutral"]

        label_col = "label" if "label" in df.columns else None

        # Build examples per class
        examples_per_class: dict[str, list[str]] = {lbl: [] for lbl in candidate_labels}
        if label_col and text_col in df.columns:
            for lbl in candidate_labels:
                subset = df[df[label_col] == lbl][text_col].dropna().tolist()
                examples_per_class[lbl] = subset[:5]

        label_definitions = {
            "positive": "Text expresses a positive opinion, satisfaction, or approval.",
            "negative": "Text expresses a negative opinion, dissatisfaction, or criticism.",
            "neutral": "Text is factual, ambiguous, or does not clearly lean positive or negative.",
        }
        edge_cases = [
            "Sarcasm ('Great, another delay...') — label as **negative** based on intent.",
            "Mixed sentiment ('Good product but bad delivery') — label the **dominant** sentiment.",
            "Questions ('Is this any good?') — label as **neutral** unless tone is clear.",
            "Very short texts ('Ok') — label as **neutral** unless context is obvious.",
            "Emojis without text ('👍') — label as **positive**; ('👎') — **negative**.",
        ]

        lines = [
            f"# Annotation Specification — {task}",
            "",
            f"**Generated:** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
            f"**Task:** {task}",
            f"**Modality:** {self.modality}",
            f"**Total samples:** {len(df)}",
            "",
            "---",
            "",
            "## Task Description",
            "",
            f"Classify each text sample into one of the following categories: "
            f"{', '.join(f'`{l}`' for l in candidate_labels)}.",
            "",
            "---",
            "",
            "## Classes and Definitions",
            "",
        ]

        for lbl in candidate_labels:
            defn = label_definitions.get(lbl, f"Text that corresponds to {lbl}.")
            lines += [f"### `{lbl}`", "", defn, "", "**Examples:**", ""]
            examples = examples_per_class.get(lbl, [])
            if examples:
                for ex in examples[:3]:
                    lines.append(f'- "{ex[:120]}"')
            else:
                lines += [
                    f'- "Example text for {lbl} class #1"',
                    f'- "Example text for {lbl} class #2"',
                    f'- "Example text for {lbl} class #3"',
                ]
            lines.append("")

        lines += [
            "---",
            "",
            "## Edge Cases and Ambiguous Examples",
            "",
        ]
        for ec in edge_cases:
            lines.append(f"- {ec}")

        lines += [
            "",
            "---",
            "",
            "## Labeling Instructions",
            "",
            "1. Read the full text before assigning a label.",
            "2. Choose the label that best reflects the **overall** tone.",
            "3. If genuinely unsure between two labels — pick `neutral`.",
            "4. Do not rely on individual words alone; consider full context.",
            "5. Flag examples you find impossible to label consistently.",
            "",
            "---",
            "",
            "## Quality Bar",
            "",
            "- Expected inter-annotator agreement (Cohen's κ): **≥ 0.7**",
            "- If your agreement with auto-labeling is < 0.6, review edge case guidelines.",
            "",
        ]

        spec_md = "\n".join(lines)
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_text(spec_md, encoding="utf-8")
        print(f"[generate_spec] Saved → {output_path}", file=sys.stderr)
        return spec_md

    # ------------------------------------------------------------------
    # skill: check_quality
    # ------------------------------------------------------------------

    def check_quality(
        self,
        df_labeled: pd.DataFrame,
        reference_col: str | None = None,
        label_col: str = "label",
        confidence_col: str = "confidence",
    ) -> dict[str, Any]:
        """
        Compute quality metrics on a labeled DataFrame.
        If reference_col is provided, compute Cohen's kappa against it.
        Returns dict: {kappa, label_dist, confidence_mean, confidence_std, flagged_pct}.
        """
        metrics: dict[str, Any] = {}

        # Label distribution
        if label_col in df_labeled.columns:
            dist = df_labeled[label_col].value_counts(normalize=True).round(4).to_dict()
            metrics["label_dist"] = dist
            metrics["label_counts"] = df_labeled[label_col].value_counts().to_dict()
        else:
            metrics["label_dist"] = {}
            metrics["label_counts"] = {}

        # Confidence stats
        if confidence_col in df_labeled.columns:
            metrics["confidence_mean"] = round(float(df_labeled[confidence_col].mean()), 4)
            metrics["confidence_std"] = round(float(df_labeled[confidence_col].std()), 4)
            metrics["confidence_min"] = round(float(df_labeled[confidence_col].min()), 4)
        else:
            metrics["confidence_mean"] = None
            metrics["confidence_std"] = None
            metrics["confidence_min"] = None

        # Flagged for review
        if "flagged_for_review" in df_labeled.columns:
            metrics["flagged_pct"] = round(
                float(df_labeled["flagged_for_review"].mean()) * 100, 2
            )
            metrics["flagged_count"] = int(df_labeled["flagged_for_review"].sum())
        else:
            metrics["flagged_pct"] = None
            metrics["flagged_count"] = None

        # Cohen's kappa vs reference
        if reference_col and reference_col in df_labeled.columns and label_col in df_labeled.columns:
            mask = df_labeled[label_col].notna() & df_labeled[reference_col].notna()
            if mask.sum() >= 2:
                kappa = cohen_kappa_score(
                    df_labeled.loc[mask, reference_col],
                    df_labeled.loc[mask, label_col],
                )
                metrics["kappa"] = round(float(kappa), 4)
                metrics["kappa_vs"] = reference_col
            else:
                metrics["kappa"] = None
                metrics["kappa_vs"] = reference_col
        else:
            metrics["kappa"] = None
            metrics["kappa_note"] = (
                "Pass reference_col='human_label' after human annotation to compute kappa."
            )

        print(f"[check_quality] Metrics: {metrics}", file=sys.stderr)
        return metrics

    # ------------------------------------------------------------------
    # skill: export_to_labelstudio
    # ------------------------------------------------------------------

    def export_to_labelstudio(
        self,
        df: pd.DataFrame,
        text_col: str = "text",
        label_col: str = "label",
        confidence_col: str = "confidence",
        output_path: str | Path = "data/labeled/labelstudio_import.json",
        flag_low_confidence: bool = True,
        low_confidence_path: str | Path = "data/labeled/low_confidence.csv",
    ) -> list[dict]:
        """
        Export DataFrame to LabelStudio import JSON format.
        Also saves low-confidence rows to a separate CSV for manual review.
        Returns the list of LabelStudio task dicts.
        """
        tasks = []

        for idx, row in df.iterrows():
            text = str(row.get(text_col, "")) if text_col in df.columns else ""
            label = str(row.get(label_col, "")) if label_col in df.columns else ""
            confidence = float(row.get(confidence_col, 0.0)) if confidence_col in df.columns else 0.0

            task: dict[str, Any] = {
                "id": int(idx),
                "data": {
                    "text": text,
                },
                "meta": {
                    "confidence": confidence,
                    "source_row": int(idx),
                    "flagged_for_review": bool(row.get("flagged_for_review", False)),
                },
            }

            # Pre-annotations (predictions from auto_label)
            if label:
                task["predictions"] = [
                    {
                        "model_version": "annotation_agent_v1",
                        "score": confidence,
                        "result": [
                            {
                                "id": f"result_{idx}",
                                "type": "choices",
                                "value": {"choices": [label]},
                                "from_name": "sentiment",
                                "to_name": "text",
                            }
                        ],
                    }
                ]

            tasks.append(task)

        # Save main export
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_text(json.dumps(tasks, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[export_to_labelstudio] {len(tasks)} tasks → {output_path}", file=sys.stderr)

        # Save low-confidence subset for manual review (BONUS: human-in-the-loop)
        if flag_low_confidence and "flagged_for_review" in df.columns:
            low_conf_df = df[df["flagged_for_review"]].copy()
            if len(low_conf_df) > 0:
                Path(low_confidence_path).parent.mkdir(parents=True, exist_ok=True)
                low_conf_df.to_csv(low_confidence_path, index=False)
                print(
                    f"[export_to_labelstudio] {len(low_conf_df)} low-confidence rows → {low_confidence_path}",
                    file=sys.stderr,
                )
            else:
                print("[export_to_labelstudio] No low-confidence rows to flag.", file=sys.stderr)

        return tasks

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_classifier(self):
        if self._classifier is None:
            from transformers import pipeline
            # MiniLM NLI: ~120MB vs bart-large-mnli ~1.6GB, ~8x faster on CPU
            print("[model] Loading cross-encoder/nli-MiniLM2-L6-H768 (~120MB, first run only)...", file=sys.stderr)
            self._classifier = pipeline(
                "zero-shot-classification",
                model="cross-encoder/nli-MiniLM2-L6-H768",
                device=-1,  # CPU; set to 0 for GPU
            )
            print("[model] Ready.", file=sys.stderr)
        return self._classifier


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="AnnotationAgent — auto-label and export")
    parser.add_argument("--input", required=True, help="Path to input CSV")
    parser.add_argument("--modality", default="text", choices=["text"], help="Data modality")
    parser.add_argument("--task", default="sentiment_classification", help="ML task name")
    parser.add_argument("--text-col", default="text", help="Name of the text column")
    parser.add_argument(
        "--labels",
        default="positive,negative,neutral",
        help="Comma-separated candidate labels",
    )
    parser.add_argument("--confidence-threshold", type=float, default=0.75)
    parser.add_argument("--output-dir", default="data/labeled", help="Directory for output files")
    parser.add_argument(
        "--reference-col",
        default=None,
        help="Column with human labels for Cohen's kappa (optional)",
    )
    args = parser.parse_args()

    candidate_labels = [l.strip() for l in args.labels.split(",")]
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Load data
    df = pd.read_csv(args.input)
    print(f"[main] Loaded {len(df)} rows from {args.input}", file=sys.stderr)

    agent = AnnotationAgent(modality=args.modality, confidence_threshold=args.confidence_threshold)

    # Step 1 — auto label
    df_labeled = agent.auto_label(df, candidate_labels=candidate_labels, text_col=args.text_col)
    labeled_path = out / "labeled.csv"
    df_labeled.to_csv(labeled_path, index=False)
    print(f"[main] Labeled data → {labeled_path}", file=sys.stderr)

    # Step 2 — generate spec
    agent.generate_spec(
        df_labeled,
        task=args.task,
        candidate_labels=candidate_labels,
        text_col=args.text_col,
        output_path=out / "annotation_spec.md",
    )

    # Step 3 — check quality
    metrics = agent.check_quality(
        df_labeled,
        reference_col=args.reference_col,
        label_col="label",
    )
    metrics_path = out / "quality_metrics.json"
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"[main] Quality metrics → {metrics_path}", file=sys.stderr)

    # Step 4 — export to LabelStudio
    agent.export_to_labelstudio(
        df_labeled,
        text_col=args.text_col,
        output_path=out / "labelstudio_import.json",
        low_confidence_path=out / "low_confidence.csv",
    )

    # Print summary
    print("\n=== Summary ===")
    print(f"Rows labeled      : {len(df_labeled)}")
    print(f"Label distribution: {metrics.get('label_dist')}")
    print(f"Confidence mean   : {metrics.get('confidence_mean')}")
    print(f"Flagged for review: {metrics.get('flagged_count')} ({metrics.get('flagged_pct')}%)")
    if metrics.get("kappa") is not None:
        print(f"Cohen's kappa     : {metrics['kappa']} (vs {metrics['kappa_vs']})")
    else:
        print(f"Cohen's kappa     : {metrics.get('kappa_note', 'N/A')}")
    print(f"\nOutput files in: {out}/")


if __name__ == "__main__":
    main()
