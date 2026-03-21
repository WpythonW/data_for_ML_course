---
name: data-pipeline
description: Полный ML-пайплайн — сбор данных → чистка → разметка → active learning. Запускается одной командой, проходит все этапы автоматически с human-in-the-loop на ключевых точках.
---

# Data Pipeline

Мета-скилл, объединяющий все 4 агента в единый пайплайн подготовки ML-датасета.

## Запуск

```
/data-pipeline <тема> --classes "class1,class2,class3" --task "описание задачи"
```

Примеры:
```
/data-pipeline "отзывы на товары" --classes "positive,negative,neutral" --task "классификация тональности"
/data-pipeline "новости" --classes "политика,спорт,технологии" --task "классификация тематики"
```

---

## Поток данных

```
┌──────────────────┐    ┌──────────────────┐    ┌──────────────────┐    ┌──────────────────┐
│  1. Collector    │    │  2. Detective    │    │  3. Annotator    │    │  4. ActiveLearn  │
│  Сбор данных     │───▶│  Чистка данных   │───▶│  Авторазметка    │───▶│  Оптимизация     │
└──────────────────┘    └──────────────────┘    └──────────────────┘    └──────────────────┘
        │                       │                       │                       │
        ▼                       ▼                       ▼                       ▼
data/raw/unified.csv   data/cleaned.csv        data/labeled.csv        data/learning_curve.png
data/eda/REPORT.md     data/comparison.json    data/labelstudio.json   data/al_savings.json
```

---

## Этап 0 — Setup

Проверить наличие `.env` с ключами:
```bash
uv run scripts/search/check_env.py
```

Создать нужные директории:
```bash
mkdir -p data/raw data/eda
```

---

## Этап 1 — Dataset Collector

**Цель:** собрать данные минимум из 2 источников и объединить в unified schema.

**Действия:**
1. Поиск датасетов на HuggingFace и Kaggle — запустить поисковую волну:
```bash
uv run scripts/search/run_wave.py \
    --wave 1 \
    --queries "<тема>, <тема> dataset, <тема> classification" \
    --keywords "<тема>, label, text" \
    --goal "<описание задачи>"
```
2. Найти второй источник — сайт для скрапинга или публичный API через WebSearch
3. Написать скрапер или API-коллектор (по скиллу `skills/scrape.md` или `skills/fetch_api.md`)
4. Объединить через `skills/merge_sources.md` → `data/raw/unified.csv`
5. Запустить EDA

**⏸ HUMAN CHECKPOINT #1:**
```
## Найденные источники

### Датасеты (HuggingFace / Kaggle):
| # | Название | Размер | Лицензия | Почему подходит |
|---|----------|--------|----------|-----------------|
| 1 | ...      | ...    | ...      | ...             |

### Второй источник:
| # | Тип | URL / Название | ~Строк |
|---|-----|----------------|--------|
| 2 | scrape/api | ... | ... |

Подтверждаешь эти источники? Или заменить что-то? [да / укажи замену]
```

**После подтверждения:**
- Скачать/собрать данные
- Применить unified schema: `text, label, source, collected_at`
- Сохранить `data/raw/unified.csv`
- Запустить EDA → `data/eda/REPORT.md`

---

## Этап 2 — Data Detective

**Цель:** найти и исправить проблемы качества данных.

**Вход:** `data/raw/unified.csv`

**Действия — запустить все детекторы:**
```bash
uv run scripts/quality/profile.py --input data/raw/unified.csv
uv run scripts/quality/detect_missing.py --input data/raw/unified.csv --output data/missing.json
uv run scripts/quality/detect_duplicates.py --input data/raw/unified.csv --output data/duplicates.json
uv run scripts/quality/detect_outliers.py --input data/raw/unified.csv --method iqr --output data/outliers.json
uv run scripts/quality/detect_imbalance.py --input data/raw/unified.csv --label label --output data/imbalance.json
```

**⏸ HUMAN CHECKPOINT #2:**
```
## Найденные проблемы качества

| Проблема | Кол-во | % | Серьёзность |
|----------|--------|---|-------------|
| Пропуски | ...    |   | низкая/средняя/высокая |
| Дубликаты | ...   |   | |
| Выбросы  | ...    |   | |
| Дисбаланс | ...   |   | |

### Доступные стратегии:
- **aggressive** — удалить пропуски, дубли, выбросы (IQR). Меньше данных, чище.
- **conservative** — заполнить пропуски, оставить выбросы. Больше данных.
- **balanced** — удалить дубли, заполнить пропуски медианой, обрезать экстремальные выбросы (z>3). Рекомендуется.

Какую стратегию применить? [aggressive / conservative / balanced]
```

**После подтверждения:**
```bash
# Применить шаг за шагом
uv run scripts/quality/fix_duplicates.py --input data/raw/unified.csv --keep first --output data/step1.csv
uv run scripts/quality/fix_missing.py --input data/step1.csv --strategy <выбор> --output data/step2.csv
uv run scripts/quality/fix_outliers.py --input data/step2.csv --strategy <выбор> --output data/cleaned.csv
uv run scripts/quality/compare.py --before data/raw/unified.csv --after data/cleaned.csv --output data/comparison.json
```
- Показать сравнительный отчёт до/после

---

## Этап 3 — Annotation Agent

**Цель:** автоматически разметить данные, сгенерировать спецификацию, экспортировать в LabelStudio.

**Вход:** `data/cleaned.csv`

**⏸ HUMAN CHECKPOINT #3:**
```
## Настройки авторазметки

- Файл: data/cleaned.csv
- Строк: N
- Классы: <classes>
- Задача: <task>
- Модель: cross-encoder/nli-MiniLM2-L6-H768 (~120MB)
- Порог уверенности: 0.75 (ниже → ручная разметка)

Размечаем? [да / изменить классы]
```

**После подтверждения:**
```bash
uv run agents/annotation_agent.py \
    --input data/cleaned.csv \
    --task "<task>" \
    --labels "<classes>" \
    --confidence-threshold 0.75 \
    --output-dir data
```

- Показать распределение меток и confidence
- Передать `data/annotation_spec.md` однокурснику для ручной разметки выборки
- Экспорт в LabelStudio: `data/labelstudio_import.json`
- Флаги низкой уверенности: `data/low_confidence.csv`

---

## Этап 4 — Active Learner

**Цель:** показать, сколько меток можно сэкономить через entropy vs random.

**Вход:** `data/labeled.csv`

**⏸ HUMAN CHECKPOINT #4:**
```
## Настройки Active Learning

- Стартовый набор: 50 примеров
- Итераций: 5 × 20 примеров = 100 дополнительных меток
- Стратегии: entropy (умный) vs random (baseline)
- Модель: LogisticRegression + TF-IDF

Запускаем? [да / изменить параметры]
```

**После подтверждения:**
```bash
uv run agents/al_agent.py \
    --input data/labeled.csv \
    --strategy entropy \
    --n-start 50 \
    --n-iterations 5 \
    --batch-size 20 \
    --compare \
    --output-dir data
```

- Показать learning curves (entropy vs random на одном графике)
- Показать savings: сколько меток сэкономлено

---

## Итоговый отчёт

После завершения всех этапов показать сводку:

```
## Пайплайн завершён

### Этап 1: Dataset Collector
- Источники: HF:<название> + scrape/api:<url>
- Строк собрано: N
- Файл: data/raw/unified.csv
- EDA: data/eda/REPORT.md

### Этап 2: Data Detective
- Стратегия: <выбранная>
- До: N строк → После: M строк (-X%)
- Файл: data/cleaned.csv

### Этап 3: Annotation Agent
- Размечено: M строк
- Confidence среднее: X
- Флаги для ручной проверки: K строк
- Спецификация: data/annotation_spec.md
- LabelStudio: data/labelstudio_import.json

### Этап 4: Active Learner
- Entropy F1: X при 150 метках
- Random F1: Y при 150 метках
- Сэкономлено: N меток (P%) vs random baseline
- График: data/learning_curve.png

### Готовые файлы для ML:
- data/labeled.csv — размеченный датасет
- data/annotation_spec.md — спецификация классов
- data/learning_curve.png — кривые обучения
```

---

## Правила

1. Идти строго по этапам: 1 → 2 → 3 → 4
2. На каждом checkpoint ждать подтверждения пользователя
3. Передавать данные автоматически: выход N = вход N+1
4. Показывать прогресс: `[Этап X/4] Название...`
5. При ошибке — показать проблему и предложить решение, не прерывать пайплайн
6. Всегда использовать `uv run`, никогда не использовать `python` напрямую
