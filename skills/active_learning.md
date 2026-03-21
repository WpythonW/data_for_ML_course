# Skill: active_learning cycle

## Purpose
Iteratively select the most informative unlabeled examples for labeling,
train a model, and track quality improvement vs random baseline.

---

## When to use
- Large unlabeled pool, labeling is expensive
- Want to reach target quality with minimum labeled examples

---

## Query strategies

| Strategy | How it works | Best for |
|----------|-------------|----------|
| `entropy` | Pick examples with highest prediction entropy (most uncertain) | General purpose |
| `margin` | Pick examples with smallest gap between top-2 predicted classes | Binary / few classes |
| `random` | Random selection — baseline | Comparison only |

---

## Entropy formula
```
H(x) = -Σ p(y|x) * log(p(y|x))
```
Higher entropy = model is more confused → most valuable to label.

---

## CLI usage

```bash
# entropy strategy + random baseline comparison
uv run agents/al_agent.py \
    --input data/labeled.csv \
    --text-col text \
    --label-col label \
    --strategy entropy \
    --n-start 50 \
    --n-iterations 5 \
    --batch-size 20 \
    --compare \
    --output-dir data
```

---

## Python API

```python
from agents.al_agent import ActiveLearningAgent

agent = ActiveLearningAgent(model='logreg')
history = agent.run_cycle(
    labeled_df=df_labeled_50,
    pool_df=df_unlabeled,
    strategy='entropy',
    n_iterations=5,
    batch_size=20,
    test_df=df_test,
)
agent.report(history)  # → data/learning_curve.png
```

---

## Output files

| File | Description |
|------|-------------|
| `data/al_history_entropy.json` | Per-iteration metrics for entropy |
| `data/al_history_random.json` | Per-iteration metrics for random |
| `data/learning_curve.png` | Both curves on one plot |
| `data/al_savings.json` | Labels saved vs random baseline |

---

## Rules
- Always compare against random baseline (`--compare`)
- Test set is held out before the AL cycle starts — never touched during training
- Report savings: how many fewer labels needed to reach same F1
