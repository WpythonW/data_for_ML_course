# Dataset Search Agent

Агент для поиска ML датасетов по HuggingFace (и скоро Kaggle).
Запускается одной командой прямо в Claude Code, результаты — в чате.

## Как это работает

```
/find-datasets <тема>
        ↓
  dataset-searcher agent
        ↓
  HuggingFace bulk search (до 3 волн)
        ↓
  BM25 ranking
        ↓
  OpenRouter: фильтрация + реранкинг
        ↓
  Haiku: целевые вопросы → финальный отбор
        ↓
  Результаты в чате
```

## Быстрый старт

### 1. Клонировать репо

```bash
git clone https://github.com/WpythonW/data_for_ML_course.git
cd data_for_ML_course
```

### 2. Установить зависимости

```bash
uv sync
```

### 3. Настроить API ключи

Скопировать `.env.example` в `.env` и заполнить:

```bash
cp .env.example .env
```

Открыть `.env` и вставить ключи:

```env
# OpenRouter — обязательно (для LLM фильтрации)
# Получить на: https://openrouter.ai/keys
OPENROUTER_API_KEY=sk-or-...

# Модель через OpenRouter (по умолчанию qwen3-30b)
# Не менять — зафиксирована для Haiku-оркестратора
OPENROUTER_MODEL=qwen/qwen3-30b-a3b

# HuggingFace — опционально, повышает rate limit
# Получить на: https://huggingface.co/settings/tokens
HF_TOKEN=hf_...

# Kaggle — для поиска по Kaggle (скоро)
# Получить: kaggle.com/settings/account → API → Create New Token
# Скачается kaggle.json — взять username и key оттуда
KAGGLE_USERNAME=your_username
KAGGLE_KEY=your_key
```

### 4. Открыть проект в Claude Code

```bash
claude .
```

### 5. Искать датасеты

```
/find-datasets car brand image classification
/find-datasets chest X-ray pneumonia detection
/find-datasets sentiment analysis Russian text
```

Агент работает автономно — до 3 волн поиска, результаты прямо в чате.

## Структура проекта

```
dataset-agent/
├── .claude/
│   ├── agents/
│   │   └── dataset-searcher.md   # System prompt агента
│   └── commands/
│       └── find-datasets.md      # Slash-команда /find-datasets
├── skills/
│   └── dataset_search.md         # Skill: описание пайплайна
├── scripts/
│   └── search/
│       ├── hf_bulk_search.py     # Bulk search по HuggingFace
│       ├── semantic_filter.py    # BM25 + OpenRouter + Haiku воронка
│       ├── update_seen_ids.py    # Обновление списка виденных ID
│       ├── merge_results.py      # Слияние сырых результатов
│       └── merge_final.py        # Слияние финальных результатов
├── .env.example                  # Шаблон для ключей
├── .gitignore
├── CLAUDE.md                     # Инструкции для Claude Code
└── pyproject.toml
```

## Пайплайн фильтрации (внутри semantic_filter.py)

| Стадия | Кто | Что |
|--------|-----|-----|
| 1 | BM25 | Широкий отсев по ключевым словам |
| 2 | OpenRouter | Суммаризация + фильтрация нерелевантных |
| 3 | OpenRouter | Реранкинг выживших |
| 3.5 | HF API | Загрузка README карточек для финалистов |
| 4 | Haiku | Выбирает датасеты для уточняющих вопросов |
| 5 | OpenRouter | Лаконично отвечает на вопросы Haiku |
| 6 | Haiku | Финальный узкий отбор |

## Дедупликация

- `data/seen_ids.txt` — все датасеты, которые уже были получены из HF API (не попадут в следующую волну поиска)
- `data/rejected_ids.txt` — все датасеты, отвергнутые OpenRouter или Haiku (не попадут в пайплайн никогда)

Оба файла обновляются автоматически и персистятся между волнами.

## Требования

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) — менеджер пакетов
- [Claude Code](https://claude.ai/claude-code) — CLI
- OpenRouter API ключ (обязательно)
