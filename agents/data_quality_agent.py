"""
DataQualityAgent — детекция и устранение проблем качества данных.

Технический контракт:
    agent = DataQualityAgent()
    report = agent.detect_issues(df)
    # → {'missing': {...}, 'duplicates': N, 'outliers': {...}, 'imbalance': {...}}

    df_clean = agent.fix(df, strategy={
        'missing': 'median',
        'duplicates': 'drop',
        'outliers': 'clip_iqr',
    })

    comparison = agent.compare(df, df_clean)
    # → DataFrame: было / стало по каждой метрике

Usage:
    uv run agents/data_quality_agent.py --input data/raw/unified.csv --label label
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score


class DataQualityAgent:
    """Detect, fix, and compare data quality issues in tabular datasets."""

    # ------------------------------------------------------------------
    # skill: detect_issues
    # ------------------------------------------------------------------

    def detect_issues(self, df: pd.DataFrame, label_col: str | None = None) -> dict:
        """
        Detect all quality issues in df.
        Returns QualityReport dict.
        """
        print("[detect] Profiling dataset...", file=sys.stderr)
        report: dict = {"shape": {"rows": len(df), "cols": len(df.columns)}, "issues": {}}

        # 1. Missing values
        missing = df.isnull().sum()
        missing_pct = (missing / len(df) * 100).round(2)
        report["issues"]["missing"] = {
            "columns": {col: {"count": int(missing[col]), "pct": float(missing_pct[col])}
                        for col in df.columns if missing[col] > 0},
            "total_pct": round(float(df.isnull().mean().mean() * 100), 2),
        }
        print(f"[detect] Missing: {report['issues']['missing']['total_pct']}% overall", file=sys.stderr)

        # 2. Duplicates
        dup_count = int(df.duplicated().sum())
        report["issues"]["duplicates"] = {
            "count": dup_count,
            "pct": round(dup_count / len(df) * 100, 2),
        }
        print(f"[detect] Duplicates: {dup_count} rows ({report['issues']['duplicates']['pct']}%)", file=sys.stderr)

        # 3. Outliers (IQR) — numeric columns only
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        outliers_info = {}
        for col in numeric_cols:
            q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
            iqr = q3 - q1
            mask = (df[col] < q1 - 1.5 * iqr) | (df[col] > q3 + 1.5 * iqr)
            n_out = int(mask.sum())
            if n_out > 0:
                outliers_info[col] = {"count": n_out, "pct": round(n_out / len(df) * 100, 2)}
        report["issues"]["outliers"] = {"columns": outliers_info, "method": "iqr"}
        print(f"[detect] Outliers: {sum(v['count'] for v in outliers_info.values())} total across {len(outliers_info)} columns", file=sys.stderr)

        # 4. Class imbalance
        if label_col and label_col in df.columns:
            dist = df[label_col].value_counts()
            ratio = round(float(dist.iloc[0] / dist.iloc[-1]), 2) if len(dist) > 1 else 1.0
            report["issues"]["imbalance"] = {
                "ratio": ratio,
                "distribution": dist.to_dict(),
                "label_col": label_col,
            }
            print(f"[detect] Imbalance ratio: {ratio}", file=sys.stderr)
        else:
            report["issues"]["imbalance"] = {"ratio": None, "note": "No label column provided"}

        # Severity
        missing_pct_total = report["issues"]["missing"]["total_pct"]
        dup_pct = report["issues"]["duplicates"]["pct"]
        imbalance_ratio = report["issues"]["imbalance"].get("ratio") or 1.0

        if missing_pct_total > 40 or dup_pct > 15 or imbalance_ratio > 10:
            severity = "critical"
        elif missing_pct_total > 20 or dup_pct > 5 or imbalance_ratio > 5:
            severity = "high"
        elif missing_pct_total > 5 or dup_pct > 1 or imbalance_ratio > 3:
            severity = "medium"
        else:
            severity = "low"

        report["severity"] = severity
        print(f"[detect] Overall severity: {severity}", file=sys.stderr)
        return report

    # ------------------------------------------------------------------
    # skill: fix
    # ------------------------------------------------------------------

    def fix(self, df: pd.DataFrame, strategy: dict | None = None) -> pd.DataFrame:
        """
        Apply cleaning strategies to df.

        strategy keys and options:
          missing:    'mean' | 'median' | 'mode' | 'ffill' | 'drop_rows' | 'constant'
          duplicates: 'drop' | 'keep_last' | 'keep_none'
          outliers:   'clip_iqr' | 'clip_zscore' | 'drop'
          imbalance:  'oversample' | 'undersample' | 'skip'
        """
        if strategy is None:
            strategy = {"missing": "median", "duplicates": "drop", "outliers": "clip_iqr"}

        df = df.copy()
        print(f"[fix] Strategy: {strategy}", file=sys.stderr)
        print(f"[fix] Input shape: {df.shape}", file=sys.stderr)

        # Step 1 — duplicates
        dup_strategy = strategy.get("duplicates", "drop")
        before = len(df)
        if dup_strategy == "drop":
            df = df.drop_duplicates(keep="first").reset_index(drop=True)
        elif dup_strategy == "keep_last":
            df = df.drop_duplicates(keep="last").reset_index(drop=True)
        elif dup_strategy == "keep_none":
            df = df[~df.duplicated(keep=False)].reset_index(drop=True)
        print(f"[fix] Duplicates: removed {before - len(df)} rows", file=sys.stderr)

        # Step 2 — missing values
        missing_strategy = strategy.get("missing", "median")
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        cat_cols = df.select_dtypes(exclude=[np.number]).columns
        imputed = 0

        if missing_strategy == "drop_rows":
            before = len(df)
            df = df.dropna().reset_index(drop=True)
            print(f"[fix] Missing: dropped {before - len(df)} rows", file=sys.stderr)
        elif missing_strategy in ("mean", "median"):
            for col in numeric_cols:
                n = df[col].isnull().sum()
                if n > 0:
                    val = df[col].mean() if missing_strategy == "mean" else df[col].median()
                    df[col] = df[col].fillna(val)
                    imputed += n
            for col in cat_cols:
                n = df[col].isnull().sum()
                if n > 0:
                    df[col] = df[col].fillna(df[col].mode().iloc[0] if not df[col].mode().empty else "unknown")
                    imputed += n
            print(f"[fix] Missing: imputed {imputed} values", file=sys.stderr)
        elif missing_strategy == "mode":
            for col in df.columns:
                n = df[col].isnull().sum()
                if n > 0:
                    mode = df[col].mode()
                    df[col] = df[col].fillna(mode.iloc[0] if not mode.empty else "unknown")
                    imputed += n
            print(f"[fix] Missing: imputed {imputed} values with mode", file=sys.stderr)
        elif missing_strategy == "ffill":
            df = df.ffill()
            print(f"[fix] Missing: forward-filled", file=sys.stderr)
        elif missing_strategy == "constant":
            df = df.fillna(0 if len(numeric_cols) else "unknown")
            print(f"[fix] Missing: filled with constant", file=sys.stderr)

        # Step 3 — outliers
        outlier_strategy = strategy.get("outliers", "clip_iqr")
        clipped = 0
        if outlier_strategy in ("clip_iqr", "clip_zscore", "drop"):
            for col in numeric_cols:
                if outlier_strategy == "clip_iqr":
                    q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
                    iqr = q3 - q1
                    lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
                    mask = (df[col] < lo) | (df[col] > hi)
                    clipped += mask.sum()
                    df[col] = df[col].clip(lo, hi)
                elif outlier_strategy == "clip_zscore":
                    mu, sigma = df[col].mean(), df[col].std()
                    if sigma > 0:
                        lo, hi = mu - 3 * sigma, mu + 3 * sigma
                        mask = (df[col] < lo) | (df[col] > hi)
                        clipped += mask.sum()
                        df[col] = df[col].clip(lo, hi)
                elif outlier_strategy == "drop":
                    q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
                    iqr = q3 - q1
                    mask = (df[col] < q1 - 1.5 * iqr) | (df[col] > q3 + 1.5 * iqr)
                    df = df[~mask].reset_index(drop=True)
            print(f"[fix] Outliers ({outlier_strategy}): handled {clipped} values", file=sys.stderr)

        print(f"[fix] Output shape: {df.shape}", file=sys.stderr)
        return df

    # ------------------------------------------------------------------
    # skill: compare
    # ------------------------------------------------------------------

    def compare(self, df_before: pd.DataFrame, df_after: pd.DataFrame) -> pd.DataFrame:
        """
        Compare before/after DataFrames across quality metrics.
        Returns a comparison DataFrame: metric | before | after | change.
        """
        numeric_cols = df_before.select_dtypes(include=[np.number]).columns

        rows = [
            {
                "metric": "rows",
                "before": len(df_before),
                "after": len(df_after),
                "change": f"{len(df_after) - len(df_before):+d}",
                "change_pct": f"{(len(df_after) - len(df_before)) / len(df_before) * 100:+.1f}%",
            },
            {
                "metric": "missing_values",
                "before": int(df_before.isnull().sum().sum()),
                "after": int(df_after.isnull().sum().sum()),
                "change": f"{int(df_after.isnull().sum().sum()) - int(df_before.isnull().sum().sum()):+d}",
                "change_pct": "",
            },
            {
                "metric": "duplicates",
                "before": int(df_before.duplicated().sum()),
                "after": int(df_after.duplicated().sum()),
                "change": f"{int(df_after.duplicated().sum()) - int(df_before.duplicated().sum()):+d}",
                "change_pct": "",
            },
        ]

        for col in numeric_cols:
            if col in df_after.columns:
                q1_b, q3_b = df_before[col].quantile(0.25), df_before[col].quantile(0.75)
                iqr_b = q3_b - q1_b
                out_b = int(((df_before[col] < q1_b - 1.5 * iqr_b) | (df_before[col] > q3_b + 1.5 * iqr_b)).sum())
                q1_a, q3_a = df_after[col].quantile(0.25), df_after[col].quantile(0.75)
                iqr_a = q3_a - q1_a
                out_a = int(((df_after[col] < q1_a - 1.5 * iqr_a) | (df_after[col] > q3_a + 1.5 * iqr_a)).sum())
                rows.append({
                    "metric": f"outliers_{col}",
                    "before": out_b,
                    "after": out_a,
                    "change": f"{out_a - out_b:+d}",
                    "change_pct": f"{(out_a - out_b) / max(out_b, 1) * 100:+.1f}%" if out_b else "",
                })

        result = pd.DataFrame(rows)
        print("[compare] Comparison table:", file=sys.stderr)
        print(result.to_string(index=False), file=sys.stderr)
        return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="DataQualityAgent")
    parser.add_argument("--input", required=True)
    parser.add_argument("--label", default=None)
    parser.add_argument("--missing-strategy", default="median",
                        choices=["mean", "median", "mode", "ffill", "drop_rows", "constant"])
    parser.add_argument("--outlier-strategy", default="clip_iqr",
                        choices=["clip_iqr", "clip_zscore", "drop"])
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    print(f"[main] Loaded {df.shape[0]} rows, {df.shape[1]} cols", file=sys.stderr)

    agent = DataQualityAgent()

    # Detect
    report = agent.detect_issues(df, label_col=args.label)
    print(f"\n[main] QualityReport severity: {report['severity']}")

    # Fix
    df_clean = agent.fix(df, strategy={
        "missing": args.missing_strategy,
        "duplicates": "drop",
        "outliers": args.outlier_strategy,
    })

    # Compare
    comparison = agent.compare(df, df_clean)
    print("\n[main] Before/After comparison:")
    print(comparison.to_string(index=False))

    # Save
    out_path = args.output or args.input.replace(".csv", "_clean.csv")
    df_clean.to_csv(out_path, index=False)
    print(f"\n[main] Saved cleaned data → {out_path}")

    report_path = Path(out_path).parent / "quality_report.json"
    report_path.write_text(json.dumps(report, indent=2, default=str))
    print(f"[main] Quality report → {report_path}")


if __name__ == "__main__":
    main()
