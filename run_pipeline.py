"""
Full ML Pipeline — Financial News Sentiment Classification
==========================================================
Запуск: uv run run_pipeline.py

Этапы:
  1. Сбор данных (DataCollectionAgent) — 3 HF датасета
  2. Очистка (DataQualityAgent) — дедупликация, выбросы, дисбаланс
  3. Авторазметка (AnnotationAgent) + HITL — правка low-confidence примеров
  4. Active Learning (ActiveLearningAgent) — entropy vs random, сохранение модели

Human-in-the-loop: после каждого этапа пайплайн останавливается и ждёт подтверждения.
После авторазметки — явная точка правки: data/labeled/low_confidence.csv.
"""

# ── 0. Env setup — до любых импортов библиотек ───────────────────────────────
from dotenv import load_dotenv
load_dotenv('/Users/andrejustinov/Desktop/Data_for_ML/.env')
import os
os.environ.setdefault('HF_TOKEN', os.getenv('HF_TOKEN', ''))
os.environ.setdefault('KAGGLE_USERNAME', os.getenv('KAGGLE_USERNAME', ''))
os.environ.setdefault('KAGGLE_KEY', os.getenv('KAGGLE_KEY', ''))

# ── Стандартные импорты ───────────────────────────────────────────────────────
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split

from agents.data_collection_agent import DataCollectionAgent
from agents.data_quality_agent import DataQualityAgent
from agents.annotation_agent import AnnotationAgent
from agents.al_agent import ActiveLearningAgent, savings_analysis

# ── Константы ─────────────────────────────────────────────────────────────────
TASK = "financial news sentiment classification"
LABELS = ["negative", "neutral", "positive"]
LABEL_MAP = {
    0: "negative", 1: "neutral", 2: "positive",
    "0": "negative", "1": "neutral", "2": "positive",
}

PATHS = {
    "raw":         Path("data/raw/unified.csv"),
    "cleaned":     Path("data/cleaned/unified_clean.csv"),
    "labeled":     Path("data/labeled/labeled.csv"),
    "low_conf":    Path("data/labeled/low_confidence.csv"),
    "labelstudio": Path("data/labeled/labelstudio_import.json"),
    "al_dir":      Path("data/al"),
    "models":      Path("models"),
    "reports":     Path("reports"),
}

NOW = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


# ── Helpers ───────────────────────────────────────────────────────────────────

def hitl_pause(message: str) -> None:
    print(f"\n{'='*60}")
    print(f"HUMAN CHECKPOINT: {message}")
    print("="*60)
    input("Нажмите Enter для продолжения (Ctrl+C для выхода)...\n")


def write_report(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    print(f"[report] Saved -> {path}", file=sys.stderr)


def normalize_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Привести все варианты меток к negative/neutral/positive."""
    df = df.copy()
    df["label"] = df["label"].map(
        lambda v: LABEL_MAP.get(v, v) if v is not None else v
    )
    known = set(LABELS)
    unknown = set(df["label"].dropna().unique()) - known
    if unknown:
        print(f"[normalize] Dropping {len(df[df['label'].isin(unknown)])} rows with unknown labels: {unknown}",
              file=sys.stderr)
        df = df[df["label"].isin(known)].reset_index(drop=True)
    return df


# ── Stage 1: Сбор данных ──────────────────────────────────────────────────────

def stage_1_collect() -> pd.DataFrame:
    print("\n" + "-"*60)
    print("[Stage 1/4] DataCollectionAgent — сбор данных")
    print("-"*60)

    PATHS["raw"].parent.mkdir(parents=True, exist_ok=True)

    agent = DataCollectionAgent()

    # Датасет 1: zeroshot — колонки: text, label (0=bearish/1=bullish/2=neutral)
    df1 = agent.load_dataset("zeroshot/twitter-financial-news-sentiment", source="hf", split="train")

    # Датасет 2: prithvi1029 — колонки: news_headline, sentiment
    df2 = agent.load_dataset("prithvi1029/sentiment-analysis-for-financial-news", source="hf", split="train")

    # Датасет 3: jean-baptiste — нестандартные колонки, грузим явно
    from datasets import load_dataset as hf_load
    ds3 = hf_load("Jean-Baptiste/financial_news_sentiment_mixte_with_phrasebank_75",
                  split="train", trust_remote_code=False)
    raw3 = ds3.to_pandas()
    df3 = pd.DataFrame({
        "text":         raw3["summary_detail_with_title"].astype(str),
        "label":        raw3["labels"].astype(str),
        "source":       "hf:Jean-Baptiste/financial_news_sentiment_mixte_with_phrasebank_75",
        "collected_at": datetime.now(timezone.utc).isoformat(),
    })

    unified = agent.merge([df1, df2, df3])
    unified = normalize_labels(unified)
    unified = unified.dropna(subset=["text", "label"]).reset_index(drop=True)

    unified.to_csv(PATHS["raw"], index=False)
    print(f"[Stage 1] Сохранено {len(unified)} строк -> {PATHS['raw']}")
    print(f"  Метки:\n{unified['label'].value_counts().to_string()}")

    write_report(
        PATHS["reports"] / "quality_report.md",
        f"""# Quality Report — {NOW}

## Этап 1: Сбор данных

| Параметр | Значение |
|----------|----------|
| Источников | 3 |
| Строк всего | {len(unified)} |
| Колонки | {list(unified.columns)} |

### Источники
```
{unified['source'].value_counts().to_string()}
```

### Распределение меток
```
{unified['label'].value_counts().to_string()}
```
"""
    )

    hitl_pause(
        f"Этап 1 завершён.\n"
        f"   Файл: {PATHS['raw']}\n"
        f"   Строк: {len(unified)}\n"
        f"   Проверьте метки и источники."
    )
    return unified


# ── Stage 2: Очистка данных ───────────────────────────────────────────────────

def stage_2_quality(df: pd.DataFrame) -> pd.DataFrame:
    print("\n" + "-"*60)
    print("[Stage 2/4] DataQualityAgent — очистка данных")
    print("-"*60)

    PATHS["cleaned"].parent.mkdir(parents=True, exist_ok=True)

    agent = DataQualityAgent()
    report = agent.detect_issues(df, label_col="label")

    print(f"[Stage 2] Severity: {report['severity']}")
    print(f"  Дубликаты: {report['issues']['duplicates']['count']} ({report['issues']['duplicates']['pct']}%)")
    print(f"  Дисбаланс: {report['issues']['imbalance'].get('ratio', '?')}x")

    df_clean = agent.fix(df, strategy={
        "missing": "median",
        "duplicates": "drop",
        "outliers": "clip_iqr",
    })
    comparison = agent.compare(df, df_clean)
    df_clean.to_csv(PATHS["cleaned"], index=False)
    print(f"[Stage 2] Сохранено {len(df_clean)} строк -> {PATHS['cleaned']}")

    try:
        comp_md = comparison.to_markdown(index=False)
    except Exception:
        comp_md = comparison.to_string(index=False)

    write_report(
        PATHS["reports"] / "quality_report.md",
        f"""# Quality Report — {NOW}

## Этап 1: Сбор данных
- Источников: 3 (zeroshot, prithvi1029, Jean-Baptiste)
- Строк собрано: {len(df)}

## Этап 2: Очистка данных

| Параметр | Значение |
|----------|----------|
| Severity | {report['severity']} |
| Дубликаты | {report['issues']['duplicates']['count']} ({report['issues']['duplicates']['pct']}%) |
| Дисбаланс (ratio) | {report['issues']['imbalance'].get('ratio', '?')}x |
| Строк до | {len(df)} |
| Строк после | {len(df_clean)} |
| Изменение | {len(df_clean) - len(df):+d} |

### Стратегия чистки
- missing: median (заполнение медианой)
- duplicates: drop (удаление дубликатов, keep=first)
- outliers: clip_iqr (обрезка по IQR)

### Сравнение до/после

{comp_md}
"""
    )

    hitl_pause(
        f"Этап 2 завершён.\n"
        f"   Файл: {PATHS['cleaned']}\n"
        f"   До: {len(df)} строк -> После: {len(df_clean)} строк\n"
        f"   Отчёт: {PATHS['reports'] / 'quality_report.md'}"
    )
    return df_clean


# ── Stage 3: Авторазметка + HITL ─────────────────────────────────────────────

def stage_3_annotate(df: pd.DataFrame) -> pd.DataFrame:
    print("\n" + "-"*60)
    print("[Stage 3/4] AnnotationAgent — авторазметка + HITL")
    print("-"*60)

    PATHS["labeled"].parent.mkdir(parents=True, exist_ok=True)

    agent = AnnotationAgent(modality="text", confidence_threshold=0.75)

    # Auto-label
    df_labeled = agent.auto_label(df, candidate_labels=LABELS, text_col="text")

    # Export: labelstudio + low_confidence.csv
    agent.export_to_labelstudio(
        df_labeled,
        text_col="text",
        output_path=PATHS["labelstudio"],
        low_confidence_path=PATHS["low_conf"],
    )

    flagged = int(df_labeled["flagged_for_review"].sum())
    total = len(df_labeled)
    print(f"[Stage 3] Размечено: {total} строк. Флаги: {flagged} ({flagged*100//max(total,1)}%)")

    # Generate spec
    agent.generate_spec(
        df_labeled,
        task=TASK,
        candidate_labels=LABELS,
        output_path=PATHS["labeled"].parent / "annotation_spec.md",
    )

    metrics_before = agent.check_quality(df_labeled)

    # ── HITL-точка ────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("HUMAN CHECKPOINT: Авторазметка завершена.")
    print(f"   Флагировано для проверки: {flagged} строк ({flagged*100//max(total,1)}%)")
    print(f"   Откройте файл и исправьте ошибочные метки в колонке 'label':")
    print(f"   -> {PATHS['low_conf'].resolve()}")
    print(f"   Допустимые значения меток: {LABELS}")
    print("="*60)
    input("После правки нажмите Enter (или Enter без правок чтобы продолжить)...\n")

    # Применяем правки из low_confidence.csv (join по тексту)
    applied = 0
    if PATHS["low_conf"].exists():
        corrected = pd.read_csv(PATHS["low_conf"])
        if "text" in corrected.columns and "label" in corrected.columns:
            text_to_label = dict(zip(corrected["text"], corrected["label"]))
            for idx, row in df_labeled.iterrows():
                new_label = text_to_label.get(row["text"])
                if new_label and new_label in LABELS and new_label != row["label"]:
                    df_labeled.at[idx, "label"] = new_label
                    df_labeled.at[idx, "flagged_for_review"] = False
                    applied += 1
    print(f"[Stage 3] Применено HITL-правок: {applied} строк")

    # Финальные метрики и сохранение
    metrics_after = agent.check_quality(df_labeled)
    df_labeled.to_csv(PATHS["labeled"], index=False)

    write_report(
        PATHS["reports"] / "annotation_report.md",
        f"""# Annotation Report — {NOW}

## Авторазметка

| Параметр | Значение |
|----------|----------|
| Модель | cross-encoder/nli-MiniLM2-L6-H768 |
| Порог confidence | 0.75 |
| Строк размечено | {total} |
| Флаги (low-confidence) | {flagged} ({flagged*100//max(total,1)}%) |

## Распределение меток (после HITL)

| Класс | Количество | Доля |
|-------|-----------|------|
{"".join(f"| {k} | {v} | {v/total:.1%} |\n" for k, v in metrics_after.get('label_counts', {}).items())}

## Confidence

| Метрика | До HITL | После HITL |
|---------|---------|------------|
| Mean | {metrics_before.get('confidence_mean', '?')} | {metrics_after.get('confidence_mean', '?')} |
| Std  | {metrics_before.get('confidence_std', '?')} | {metrics_after.get('confidence_std', '?')} |
| Min  | {metrics_before.get('confidence_min', '?')} | {metrics_after.get('confidence_min', '?')} |

## HITL

- Low-confidence файл: `data/labeled/low_confidence.csv`
- Правок применено: **{applied}**
- Допустимые классы: {LABELS}

## Выходные файлы

- `data/labeled/labeled.csv` — финальный размеченный датасет
- `data/labeled/labelstudio_import.json` — импорт для LabelStudio
- `data/labeled/annotation_spec.md` — спецификация разметки
"""
    )

    hitl_pause(
        f"Этап 3 завершён.\n"
        f"   Размечено: {total} строк, HITL-правок: {applied}\n"
        f"   Отчёт: {PATHS['reports'] / 'annotation_report.md'}"
    )
    return df_labeled


# ── Stage 4: Active Learning ──────────────────────────────────────────────────

def stage_4_active_learning(df: pd.DataFrame) -> None:
    print("\n" + "-"*60)
    print("[Stage 4/4] ActiveLearningAgent — entropy sampling + обучение модели")
    print("-"*60)

    PATHS["al_dir"].mkdir(parents=True, exist_ok=True)
    PATHS["models"].mkdir(parents=True, exist_ok=True)

    df = df[df["label"].isin(LABELS)].dropna(subset=["label", "text"]).reset_index(drop=True)

    train_df, test_df = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df["label"]
    )
    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    n_start = min(50, len(train_df) - 20)
    labeled_df = train_df.iloc[:n_start].copy()
    pool_df = train_df.iloc[n_start:].copy().reset_index(drop=True)

    print(f"[Stage 4] Train={len(train_df)}, Test={len(test_df)}, Start={n_start}, Pool={len(pool_df)}")

    # Entropy
    agent = ActiveLearningAgent(model="logreg")
    history = agent.run_cycle(
        labeled_df=labeled_df,
        pool_df=pool_df,
        strategy="entropy",
        n_iterations=5,
        batch_size=20,
        test_df=test_df,
        text_col="text",
        label_col="label",
    )
    (PATHS["al_dir"] / "al_history_entropy.json").write_text(json.dumps(history, indent=2))

    # Random baseline
    agent_rnd = ActiveLearningAgent(model="logreg")
    random_history = agent_rnd.run_cycle(
        labeled_df=labeled_df.copy(),
        pool_df=pool_df.copy(),
        strategy="random",
        n_iterations=5,
        batch_size=20,
        test_df=test_df,
        text_col="text",
        label_col="label",
    )
    (PATHS["al_dir"] / "al_history_random.json").write_text(json.dumps(random_history, indent=2))

    # Learning curve PNG
    agent.report(
        history,
        output_path=PATHS["al_dir"] / "learning_curve.png",
        compare_history=random_history,
        compare_label="random",
    )

    # Savings analysis
    savings = savings_analysis(history, random_history, target_metric="f1")
    (PATHS["al_dir"] / "al_savings.json").write_text(json.dumps(savings, indent=2))

    # ── Сохраняем обученную модель ────────────────────────────────────────────
    joblib.dump(agent.pipeline,      PATHS["models"] / "al_pipeline.joblib")
    joblib.dump(agent.label_encoder, PATHS["models"] / "label_encoder.joblib")
    print(f"[Stage 4] Модель сохранена -> {PATHS['models'] / 'al_pipeline.joblib'}")

    final = history[-1]
    rnd_final = random_history[-1]

    history_rows = "".join(
        f"| {h['iteration']} | {h['n_labeled']} | {h['accuracy']:.4f} | {h['f1']:.4f} |\n"
        for h in history
    )
    random_rows = "".join(
        f"| {h['iteration']} | {h['n_labeled']} | {h['accuracy']:.4f} | {h['f1']:.4f} |\n"
        for h in random_history
    )

    write_report(
        PATHS["reports"] / "al_report.md",
        f"""# Active Learning Report — {NOW}

## Конфигурация

| Параметр | Значение |
|----------|---------|
| Модель | LogisticRegression + TF-IDF (ngram 1-2, max_features=10k) |
| Стратегия | entropy vs random |
| n_start | {n_start} |
| Итераций | 5 |
| Batch size | 20 |
| Train / Test | {len(train_df)} / {len(test_df)} |

## Entropy Strategy

| Итерация | N labeled | Accuracy | F1 |
|---------|-----------|----------|-----|
{history_rows}

## Random Baseline

| Итерация | N labeled | Accuracy | F1 |
|---------|-----------|----------|-----|
{random_rows}

## Экономия меток

| Метрика | Значение |
|---------|---------|
| Random final F1 | {rnd_final['f1']} при {rnd_final['n_labeled']} метках |
| Entropy final F1 | {final['f1']} при {final['n_labeled']} метках |
| Entropy достигает F1 random при | {savings.get('entropy_reached_same_quality_at_n', '?')} метках |
| Сэкономлено меток | {savings.get('labels_saved', '?')} ({savings.get('savings_pct', '?')}%) |

## Сохранённая модель

- `models/al_pipeline.joblib` — TF-IDF + LogisticRegression
- `models/label_encoder.joblib` — LabelEncoder

## График обучения

![learning_curve](../data/al/learning_curve.png)
"""
    )

    print(f"\n[Stage 4] Entropy F1: {final['f1']} | Random F1: {rnd_final['f1']}")
    if savings.get("labels_saved"):
        print(f"[Stage 4] Сэкономлено: {savings['labels_saved']} меток ({savings['savings_pct']}%)")

    hitl_pause(
        f"Этап 4 завершён.\n"
        f"   Модель: {PATHS['models'] / 'al_pipeline.joblib'}\n"
        f"   Отчёт: {PATHS['reports'] / 'al_report.md'}"
    )


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  ML Pipeline: Financial News Sentiment Classification")
    print(f"  {NOW}")
    print("=" * 60)
    print("Этапы: Сбор -> Очистка -> Авторазметка (HITL) -> Active Learning")
    print("На каждом этапе — пауза для проверки.\n")

    df_raw     = stage_1_collect()
    df_clean   = stage_2_quality(df_raw)
    df_labeled = stage_3_annotate(df_clean)
    stage_4_active_learning(df_labeled)

    print("\n" + "="*60)
    print("Pipeline complete!")
    print("="*60)
    print(f"  Репорты:  reports/quality_report.md")
    print(f"            reports/annotation_report.md")
    print(f"            reports/al_report.md")
    print(f"  Модель:   models/al_pipeline.joblib")
    print(f"  Данные:   data/labeled/labeled.csv")
    print(f"            data/labeled/labelstudio_import.json")
    print(f"  График:   data/al/learning_curve.png")


if __name__ == "__main__":
    main()
