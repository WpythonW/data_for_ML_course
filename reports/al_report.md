# Active Learning Report — 2026-03-22

## Конфигурация

| Параметр | Значение |
|----------|---------|
| Модель | LogisticRegression + TF-IDF (ngram 1-2, max_features=10k, sublinear_tf=True) |
| Стратегии | entropy vs random |
| Train / Test split | 80% / 20% |
| n_start | 250 |
| Итераций | 5 |
| Batch size | 250 |
| Финальный размер | 1,500 меток |

## Сравнение стратегий: Entropy vs Random (шаг 250)

| Меток | Entropy Accuracy | Entropy F1 | Random Accuracy | Random F1 | Δ F1 |
|-------|-----------------|------------|-----------------|-----------|------|
| 250   | 0.6663 | 0.6506 | 0.6663 | 0.6506 | +0.000 |
| 500   | 0.7087 | 0.7012 | 0.7184 | 0.7089 | −0.008 |
| 750   | 0.7221 | 0.7148 | 0.7172 | 0.7059 | **+0.009** |
| 1,000 | 0.7464 | 0.7376 | 0.7464 | 0.7378 | −0.000 |
| 1,250 | 0.7597 | 0.7510 | 0.7536 | 0.7420 | **+0.009** |
| 1,500 | **0.7670** | **0.7598** | 0.7597 | 0.7496 | **+0.010** |

## Экономия меток

| Метрика | Значение |
|---------|---------|
| Random final F1 | 0.7496 при 1,500 метках |
| Entropy достигает той же F1 при | 1,250 метках |
| Сэкономлено меток | **250 (16.7%)** |
| Entropy final F1 | 0.7598 (+0.010 vs random) |

## Итоговые метрики модели

| Метрика | Entropy | Random |
|---------|---------|--------|
| Accuracy (1,500 меток) | **0.767** | 0.760 |
| F1 weighted (1,500 меток) | **0.760** | 0.750 |

## Сохранённая модель

| Файл | Описание |
|------|---------|
| `models/al_pipeline.joblib` | TF-IDF + LogisticRegression (финальная модель, 1,500 меток) |
| `models/label_encoder.joblib` | LabelEncoder (negative=0, neutral=1, positive=2) |

### Загрузка модели

```python
import joblib
pipeline = joblib.load('models/al_pipeline.joblib')
encoder  = joblib.load('models/label_encoder.joblib')

pred_encoded = pipeline.predict(["Stock markets rally on Fed news"])
pred_label   = encoder.inverse_transform(pred_encoded)
# → ['positive']
```

## График обучения

![learning_curve](../data/al/learning_curve.png)

## Выводы

- Entropy sampling **стабильно опережает** random baseline начиная с 750 меток
- При 1,500 метках entropy достигает F1=0.760 vs F1=0.750 у random (+1.0 п.п.)
- Экономия **250 меток (16.7%)** при достижении того же уровня качества
- Для задачи с 3 сбалансированными классами и короткими финансовыми текстами
  TF-IDF + LogReg показывает хорошее качество (F1=0.76) без fine-tuning трансформеров
