# ML Data Pipeline — Claude Code Skills

Набор агентов и скиллов для подготовки ML-датасетов: поиск → чистка → разметка → active learning.
Все этапы объединены в единый пайплайн, запускаемый одной командой.

---

## Демо-видео

Запись демонстрации полного пайплайна: [Google Drive](https://drive.google.com/drive/folders/1lw1AV9gAPa6Vb1bh69g7H-8kpW4edqnU?usp=sharing)

Сценарий видео: [VIDEO_SCRIPT.md](./VIDEO_SCRIPT.md)

---

## Быстрый старт

### 1. Установить зависимости

```bash
uv sync
```

### 2. Настроить API ключи

```bash
cp .env.example .env
# Открыть .env и заполнить ключи
```

```env
OPENROUTER_API_KEY=sk-or-...     # обязательно (LLM фильтрация датасетов)
OPENROUTER_MODEL=qwen/qwen3-30b-a3b
HF_TOKEN=hf_...                  # опционально (повышает rate limit HuggingFace)
KAGGLE_USERNAME=...              # для поиска по Kaggle
KAGGLE_KEY=...
```

### 3. Открыть в Claude Code

```bash
claude .
```

### 4. Запустить пайплайн

```
/data-pipeline "отзывы на товары" --classes "positive,negative,neutral" --task "классификация тональности"
```

---

## Пайплайн

```
┌──────────────────┐    ┌──────────────────┐    ┌──────────────────┐    ┌──────────────────┐
│  1. Collector    │    │  2. Detective    │    │  3. Annotator    │    │  4. ActiveLearn  │
│  Сбор данных     │───▶│  Чистка данных   │───▶│  Авторазметка    │───▶│  Оптимизация     │
└──────────────────┘    └──────────────────┘    └──────────────────┘    └──────────────────┘
        │                       │                       │                       │
        ▼                       ▼                       ▼                       ▼
data/raw/unified.csv   data/cleaned.csv        data/labeled.csv        learning_curve.png
```

На каждом этапе — **human-in-the-loop**: агент останавливается, показывает что нашёл,
и ждёт подтверждения перед тем как двигаться дальше.

---

## Агенты

| Агент | Файл | Задание |
|-------|------|---------|
| DataCollectionAgent | `agents/data_collection_agent.py` | Задание 1 |
| DataQualityAgent | `agents/data_quality_agent.py` | Задание 2 |
| AnnotationAgent | `agents/annotation_agent.py` | Задание 3 |
| ActiveLearningAgent | `agents/al_agent.py` | Задание 4 |

### DataCollectionAgent (Задание 1)

Собирает данные из нескольких источников, приводит к unified schema.

```python
from agents.data_collection_agent import DataCollectionAgent

agent = DataCollectionAgent(config='config_annotation.yaml')
df = agent.run(sources=[
    {'type': 'hf_dataset', 'name': 'imdb'},
    {'type': 'scrape', 'url': 'https://...', 'selector': '.review-text'},
    {'type': 'api', 'endpoint': 'https://api.example.com/data', 'params': {}},
])
# → pd.DataFrame: text, audio, image, label, source, collected_at
```

**Unified schema:**

| Колонка | Описание |
|---------|----------|
| `text` | Текстовый контент |
| `audio` | Путь к аудиофайлу |
| `image` | Путь/URL к изображению |
| `label` | Метка класса |
| `source` | `hf:<name>`, `kaggle:<name>`, `scrape:<url>`, `api:<endpoint>` |
| `collected_at` | ISO timestamp |

### DataQualityAgent (Задание 2)

Детектирует и исправляет проблемы качества: пропуски, дубликаты, выбросы, дисбаланс классов.

```python
from agents.data_quality_agent import DataQualityAgent

agent = DataQualityAgent()
report = agent.detect_issues(df, label_col='label')
# → {'missing': {...}, 'duplicates': N, 'outliers': {...}, 'imbalance': {...}, 'severity': 'medium'}

df_clean = agent.fix(df, strategy={
    'missing': 'median',      # mean | median | mode | ffill | drop_rows | constant
    'duplicates': 'drop',     # drop | keep_last | keep_none
    'outliers': 'clip_iqr',   # clip_iqr | clip_zscore | drop
})

comparison = agent.compare(df, df_clean)
# → DataFrame: metric | before | after | change | change_pct
```

Для детальной аналитики — отдельные скрипты в `scripts/quality/`:

```bash
uv run scripts/quality/detect_missing.py --input data.csv --output missing.json
uv run scripts/quality/detect_outliers.py --input data.csv --method iqr
uv run scripts/quality/fix_duplicates.py --input data.csv --keep first --output clean.csv
uv run scripts/quality/compare.py --before data.csv --after clean.csv
```

### AnnotationAgent (Задание 3)

Автоматически размечает тексты через zero-shot классификацию (NLI). Флагует сомнительные примеры для ручной проверки. Экспортирует в LabelStudio.

```python
from agents.annotation_agent import AnnotationAgent

agent = AnnotationAgent(modality='text', confidence_threshold=0.75)
df_labeled = agent.auto_label(df, candidate_labels=['positive', 'negative', 'neutral'])
# Модель: cross-encoder/nli-MiniLM2-L6-H768 (~120MB, быстрая)

spec = agent.generate_spec(df, task='sentiment_classification')
# → annotation_spec.md: задача, классы, примеры, граничные случаи

metrics = agent.check_quality(df_labeled, reference_col='human_label')
# → {'kappa': 0.72, 'label_dist': {...}, 'confidence_mean': 0.85}

agent.export_to_labelstudio(df_labeled)
# → labelstudio_import.json + low_confidence.csv (human-in-the-loop бонус)
```

```bash
uv run agents/annotation_agent.py \
    --input data/cleaned.csv \
    --labels "positive,negative,neutral" \
    --confidence-threshold 0.75 \
    --output-dir data
```

### ActiveLearningAgent (Задание 4)

Итеративно выбирает наиболее информативные примеры для разметки. Сравнивает стратегии entropy vs random. Показывает экономию меток.

```python
from agents.al_agent import ActiveLearningAgent

agent = ActiveLearningAgent(model='logreg')
history = agent.run_cycle(
    labeled_df=df_labeled_50,
    pool_df=df_unlabeled,
    strategy='entropy',   # entropy | margin | random
    n_iterations=5,
    batch_size=20,
    test_df=df_test,
)
agent.report(history, compare_history=random_history)
# → learning_curve.png: обе кривые на одном графике
```

```bash
uv run agents/al_agent.py \
    --input data/labeled.csv \
    --strategy entropy \
    --n-start 50 --n-iterations 5 --batch-size 20 \
    --compare --output-dir data
```

**Результат на newsgroups (3 класса, 2649 строк):**
- Entropy достигает F1=0.52 при **90 метках**
- Random достигает той же F1 при **150 метках**
- Экономия: **60 меток (40%)**

---

## Скиллы

| Скилл | Файл | Агент |
|-------|------|-------|
| Dataset Search | `skills/dataset_search.md` | DataCollectionAgent |
| Scrape | `skills/scrape.md` | DataCollectionAgent |
| Fetch API | `skills/fetch_api.md` | DataCollectionAgent |
| Merge Sources | `skills/merge_sources.md` | DataCollectionAgent |
| Detect Issues | `skills/detect_issues.md` | DataQualityAgent |
| Fix Data | `skills/fix_data.md` | DataQualityAgent |
| Auto Label | `skills/auto_label.md` | AnnotationAgent |
| Check Quality | `skills/check_quality.md` | AnnotationAgent |
| Export LabelStudio | `skills/export_labelstudio.md` | AnnotationAgent |
| Active Learning | `skills/active_learning.md` | ActiveLearningAgent |
| **Data Pipeline** | `skills/data_pipeline.md` | все агенты |

---

## Структура проекта

```
.
├── agents/
│   ├── data_collection_agent.py   # Задание 1 — сбор данных
│   ├── data_quality_agent.py      # Задание 2 — качество данных
│   ├── annotation_agent.py        # Задание 3 — авторазметка
│   └── al_agent.py                # Задание 4 — active learning
│
├── skills/
│   ├── data_pipeline.md           # Финальный пайплайн (все этапы)
│   ├── dataset_search.md          # Поиск датасетов
│   ├── scrape.md                  # Playwright скрапинг
│   ├── fetch_api.md               # REST API коллектор
│   ├── merge_sources.md           # Объединение источников
│   ├── detect_issues.md           # Детекция проблем
│   ├── fix_data.md                # Чистка данных
│   ├── auto_label.md              # Zero-shot разметка
│   ├── check_quality.md           # Метрики качества
│   ├── export_labelstudio.md      # Экспорт в LabelStudio
│   └── active_learning.md         # AL стратегии
│
├── scripts/
│   ├── search/                    # HF + Kaggle поиск, BM25, LLM фильтр
│   └── quality/                   # Детекторы и фиксеры качества
│
├── notebooks/
│   ├── annotation_eda.ipynb       # EDA разметки
│   └── al_experiment.ipynb        # AL эксперимент
│
├── data/
│   └── raw/                       # Сырые данные
│
├── config_annotation.yaml         # Конфиг источников данных
├── pyproject.toml                 # Зависимости (uv)
└── .env.example                   # Шаблон API ключей
```

---

## Зависимости

Управляются через `uv`. Установка:

```bash
uv sync
```

Основные пакеты: `pandas`, `scikit-learn`, `transformers`, `torch`, `datasets`,
`playwright`, `matplotlib`, `huggingface-hub`, `kaggle`, `python-dotenv`.

---

## Требования

- Python 3.13+
- [uv](https://docs.astral.sh/uv/)
- [Claude Code](https://docs.anthropic.com/claude-code)
- OpenRouter API ключ
