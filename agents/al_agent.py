"""
ActiveLearningAgent — умный отбор данных для разметки.

Цикл: старт с N labeled → итеративно выбираем наиболее информативные
примеры из unlabeled pool → переобучаем модель → оцениваем качество.

Usage:
    uv run agents/al_agent.py \
        --input data/labeled.csv \
        --text-col text \
        --label-col label \
        --strategy entropy \
        --n-iterations 5 \
        --batch-size 20 \
        --n-start 50
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder


# ---------------------------------------------------------------------------
# ActiveLearningAgent
# ---------------------------------------------------------------------------

class ActiveLearningAgent:
    """
    Active Learning agent with three query strategies:
      - entropy:  pick examples where model is most uncertain (high entropy)
      - margin:   pick examples with smallest margin between top-2 classes
      - random:   random baseline
    """

    SUPPORTED_MODELS = {"logreg", "svm"}

    def __init__(self, model: str = "logreg", random_state: int = 42):
        if model not in self.SUPPORTED_MODELS:
            raise ValueError(f"model must be one of {self.SUPPORTED_MODELS}")
        self.model_name = model
        self.random_state = random_state
        self.pipeline: Pipeline | None = None
        self.label_encoder = LabelEncoder()
        self._is_fitted = False

    # ------------------------------------------------------------------
    # skill: fit
    # ------------------------------------------------------------------

    def fit(self, labeled_df: pd.DataFrame, text_col: str = "text", label_col: str = "label") -> Pipeline:
        """
        Train a text classification pipeline on labeled_df.
        Returns the fitted sklearn Pipeline.
        """
        X = labeled_df[text_col].fillna("").tolist()
        y_raw = labeled_df[label_col].tolist()
        y = self.label_encoder.fit_transform(y_raw)

        if self.model_name == "logreg":
            clf = LogisticRegression(max_iter=1000, random_state=self.random_state, C=1.0)
        else:
            from sklearn.svm import SVC
            clf = SVC(probability=True, random_state=self.random_state)

        self.pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(max_features=10_000, ngram_range=(1, 2), sublinear_tf=True)),
            ("clf", clf),
        ])
        self.pipeline.fit(X, y)
        self._is_fitted = True
        return self.pipeline

    # ------------------------------------------------------------------
    # skill: query
    # ------------------------------------------------------------------

    def query(
        self,
        pool: pd.DataFrame,
        strategy: str = "entropy",
        batch_size: int = 20,
        text_col: str = "text",
    ) -> list[int]:
        """
        Select batch_size most informative indices from the unlabeled pool.
        Strategies: 'entropy', 'margin', 'random'.
        Returns list of positional indices into pool.
        """
        if not self._is_fitted:
            raise RuntimeError("Call fit() before query().")
        if strategy not in ("entropy", "margin", "random"):
            raise ValueError("strategy must be 'entropy', 'margin', or 'random'")

        if strategy == "random":
            rng = np.random.default_rng(self.random_state)
            indices = rng.choice(len(pool), size=min(batch_size, len(pool)), replace=False).tolist()
            return sorted(indices)

        X_pool = pool[text_col].fillna("").tolist()
        proba = self.pipeline.predict_proba(X_pool)  # (n_samples, n_classes)

        if strategy == "entropy":
            # Shannon entropy: higher = more uncertain
            log_proba = np.log(proba + 1e-10)
            scores = -np.sum(proba * log_proba, axis=1)
        else:  # margin
            # Margin: smaller margin = more uncertain
            sorted_proba = np.sort(proba, axis=1)[:, ::-1]
            scores = -(sorted_proba[:, 0] - sorted_proba[:, 1])

        top_indices = np.argsort(scores)[::-1][:batch_size].tolist()
        return sorted(top_indices)

    # ------------------------------------------------------------------
    # skill: evaluate
    # ------------------------------------------------------------------

    def evaluate(
        self,
        labeled_df: pd.DataFrame,
        test_df: pd.DataFrame,
        text_col: str = "text",
        label_col: str = "label",
    ) -> dict[str, float]:
        """
        Evaluate current model on test_df.
        Returns {'accuracy': float, 'f1': float, 'n_labeled': int}.
        """
        if not self._is_fitted:
            raise RuntimeError("Call fit() before evaluate().")

        X_test = test_df[text_col].fillna("").tolist()
        y_true_raw = test_df[label_col].tolist()

        # Handle unseen labels in test set gracefully
        known = set(self.label_encoder.classes_)
        y_true_filtered, X_test_filtered = [], []
        for x, y in zip(X_test, y_true_raw):
            if y in known:
                X_test_filtered.append(x)
                y_true_filtered.append(y)

        if not X_test_filtered:
            return {"accuracy": 0.0, "f1": 0.0, "n_labeled": len(labeled_df)}

        y_true = self.label_encoder.transform(y_true_filtered)
        y_pred = self.pipeline.predict(X_test_filtered)

        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)

        return {
            "accuracy": round(float(acc), 4),
            "f1": round(float(f1), 4),
            "n_labeled": len(labeled_df),
        }

    # ------------------------------------------------------------------
    # skill: report
    # ------------------------------------------------------------------

    def report(
        self,
        history: list[dict],
        output_path: str | Path = "data/learning_curve.png",
        compare_history: list[dict] | None = None,
        compare_label: str = "random",
    ) -> None:
        """
        Plot learning curve: quality (accuracy + F1) vs n_labeled.
        If compare_history provided, overlay it on the same plot.
        Saves to output_path.
        """
        import matplotlib.pyplot as plt

        n = [h["n_labeled"] for h in history]
        acc = [h["accuracy"] for h in history]
        f1 = [h["f1"] for h in history]

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        strategy_label = history[0].get("strategy", "strategy")

        # Accuracy
        axes[0].plot(n, acc, marker="o", linewidth=2, label=strategy_label, color="steelblue")
        if compare_history:
            n_c = [h["n_labeled"] for h in compare_history]
            acc_c = [h["accuracy"] for h in compare_history]
            axes[0].plot(n_c, acc_c, marker="s", linewidth=2, linestyle="--",
                         label=compare_label, color="tomato")
        axes[0].set_title("Learning Curve — Accuracy")
        axes[0].set_xlabel("Number of labeled examples")
        axes[0].set_ylabel("Accuracy")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # F1
        axes[1].plot(n, f1, marker="o", linewidth=2, label=strategy_label, color="steelblue")
        if compare_history:
            n_c = [h["n_labeled"] for h in compare_history]
            f1_c = [h["f1"] for h in compare_history]
            axes[1].plot(n_c, f1_c, marker="s", linewidth=2, linestyle="--",
                         label=compare_label, color="tomato")
        axes[1].set_title("Learning Curve — F1")
        axes[1].set_xlabel("Number of labeled examples")
        axes[1].set_ylabel("F1 (weighted)")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.suptitle(f"Active Learning: {strategy_label} vs {compare_label}", fontsize=13)
        plt.tight_layout()
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150)
        plt.close()
        print(f"[report] Learning curve saved → {output_path}", file=sys.stderr)

    # ------------------------------------------------------------------
    # skill: run_cycle (main facade)
    # ------------------------------------------------------------------

    def run_cycle(
        self,
        labeled_df: pd.DataFrame,
        pool_df: pd.DataFrame,
        strategy: str = "entropy",
        n_iterations: int = 5,
        batch_size: int = 20,
        test_df: pd.DataFrame | None = None,
        text_col: str = "text",
        label_col: str = "label",
    ) -> list[dict[str, Any]]:
        """
        Run the full active learning cycle.

        1. Fit on labeled_df
        2. Evaluate on test_df
        3. Query batch_size examples from pool_df
        4. Move queried examples to labeled_df
        5. Repeat n_iterations times

        Returns history: list of {iteration, n_labeled, accuracy, f1, strategy}
        """
        history = []
        current_labeled = labeled_df.copy()
        current_pool = pool_df.copy().reset_index(drop=True)

        total_iters = n_iterations + 1
        print(f"\n[AL] strategy={strategy} | start={len(current_labeled)} labeled | pool={len(current_pool)} | {n_iterations} iters × batch {batch_size}", file=sys.stderr)
        print(f"{'Iter':>5}  {'N labeled':>10}  {'Accuracy':>10}  {'F1':>8}  {'Progress'}", file=sys.stderr)
        print("-" * 60, file=sys.stderr)

        for iteration in range(total_iters):
            # Fit on current labeled set
            self.fit(current_labeled, text_col=text_col, label_col=label_col)

            # Evaluate
            eval_df = test_df if test_df is not None else current_labeled
            metrics = self.evaluate(current_labeled, eval_df, text_col=text_col, label_col=label_col)
            metrics["iteration"] = iteration
            metrics["strategy"] = strategy
            history.append(metrics)

            bar_len = 20
            filled = int(bar_len * iteration / max(n_iterations, 1))
            bar = "█" * filled + "░" * (bar_len - filled)
            print(
                f"{iteration:>5}  {metrics['n_labeled']:>10}  {metrics['accuracy']:>10.4f}  {metrics['f1']:>8.4f}  [{bar}]",
                file=sys.stderr,
            )

            if iteration == n_iterations or len(current_pool) == 0:
                break

            # Query next batch
            batch_size_actual = min(batch_size, len(current_pool))
            indices = self.query(current_pool, strategy=strategy, batch_size=batch_size_actual, text_col=text_col)

            # Move queried rows from pool to labeled
            queried = current_pool.iloc[indices].copy()
            current_labeled = pd.concat([current_labeled, queried], ignore_index=True)
            current_pool = current_pool.drop(current_pool.index[indices]).reset_index(drop=True)

        print(f"[run_cycle] Done. Final: {len(current_labeled)} labeled.", file=sys.stderr)
        return history


# ---------------------------------------------------------------------------
# Savings examples cost analysis
# ---------------------------------------------------------------------------

def savings_analysis(entropy_history: list[dict], random_history: list[dict], target_metric: str = "f1") -> dict:
    """
    Compute how many fewer labels entropy strategy needs to reach
    the same quality as random at its final iteration.
    """
    random_final = random_history[-1][target_metric]

    # Find first iteration where entropy reaches random_final quality
    entropy_threshold_n = None
    for h in entropy_history:
        if h[target_metric] >= random_final:
            entropy_threshold_n = h["n_labeled"]
            break

    random_final_n = random_history[-1]["n_labeled"]

    result = {
        "target_metric": target_metric,
        "random_final_quality": random_final,
        "random_final_n_labeled": random_final_n,
        "entropy_reached_same_quality_at_n": entropy_threshold_n,
        "labels_saved": (random_final_n - entropy_threshold_n) if entropy_threshold_n else None,
        "savings_pct": (
            round((random_final_n - entropy_threshold_n) / random_final_n * 100, 1)
            if entropy_threshold_n else None
        ),
    }
    return result


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="ActiveLearningAgent — run AL cycle")
    parser.add_argument("--input", required=True, help="Path to labeled CSV")
    parser.add_argument("--text-col", default="text")
    parser.add_argument("--label-col", default="label")
    parser.add_argument("--model", default="logreg", choices=["logreg", "svm"])
    parser.add_argument("--strategy", default="entropy", choices=["entropy", "margin", "random"])
    parser.add_argument("--n-start", type=int, default=50, help="Initial labeled set size")
    parser.add_argument("--n-iterations", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--test-size", type=float, default=0.2, help="Fraction held out as test set")
    parser.add_argument("--compare", action="store_true", help="Also run random baseline for comparison")
    parser.add_argument("--output-dir", default="data")
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Load data
    df = pd.read_csv(args.input)
    df = df[df[args.label_col].notna()].reset_index(drop=True)
    print(f"[main] Loaded {len(df)} labeled rows.", file=sys.stderr)

    # Split: test set (held out) + working set
    train_df, test_df = train_test_split(
        df, test_size=args.test_size, random_state=42, stratify=df[args.label_col]
    )
    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    # Split working set: initial labeled (n_start) + pool
    n_start = min(args.n_start, len(train_df) - args.batch_size)
    labeled_df = train_df.iloc[:n_start].copy()
    pool_df = train_df.iloc[n_start:].copy().reset_index(drop=True)

    print(f"[main] Train={len(train_df)}, Test={len(test_df)}, Initial labeled={len(labeled_df)}, Pool={len(pool_df)}", file=sys.stderr)

    # Run main strategy
    agent = ActiveLearningAgent(model=args.model)
    history = agent.run_cycle(
        labeled_df=labeled_df,
        pool_df=pool_df,
        strategy=args.strategy,
        n_iterations=args.n_iterations,
        batch_size=args.batch_size,
        test_df=test_df,
        text_col=args.text_col,
        label_col=args.label_col,
    )

    # Save history
    history_path = out / f"al_history_{args.strategy}.json"
    history_path.write_text(json.dumps(history, indent=2))
    print(f"[main] History → {history_path}", file=sys.stderr)

    # Run random baseline for comparison
    random_history = None
    if args.compare or args.strategy != "random":
        print("[main] Running random baseline...", file=sys.stderr)
        agent_random = ActiveLearningAgent(model=args.model)
        random_history = agent_random.run_cycle(
            labeled_df=labeled_df.copy(),
            pool_df=pool_df.copy(),
            strategy="random",
            n_iterations=args.n_iterations,
            batch_size=args.batch_size,
            test_df=test_df,
            text_col=args.text_col,
            label_col=args.label_col,
        )
        random_path = out / "al_history_random.json"
        random_path.write_text(json.dumps(random_history, indent=2))

    # Report
    agent.report(
        history,
        output_path=out / "learning_curve.png",
        compare_history=random_history,
        compare_label="random",
    )

    # Savings analysis
    if random_history:
        savings = savings_analysis(history, random_history, target_metric="f1")
        savings_path = out / "al_savings.json"
        savings_path.write_text(json.dumps(savings, indent=2))

        print("\n=== Savings Analysis ===")
        print(f"Random baseline final F1     : {savings['random_final_quality']} at {savings['random_final_n_labeled']} labels")
        if savings["entropy_reached_same_quality_at_n"]:
            print(f"Entropy reached same F1 at   : {savings['entropy_reached_same_quality_at_n']} labels")
            print(f"Labels saved                 : {savings['labels_saved']} ({savings['savings_pct']}%)")
        else:
            print("Entropy did not reach random's final quality within the cycle.")

    # Summary
    print("\n=== Final Results ===")
    final = history[-1]
    print(f"Strategy   : {args.strategy}")
    print(f"N labeled  : {final['n_labeled']}")
    print(f"Accuracy   : {final['accuracy']}")
    print(f"F1         : {final['f1']}")
    print(f"\nOutput files in: {out}/")


if __name__ == "__main__":
    main()
