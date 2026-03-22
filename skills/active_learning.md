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
# Recommended: step=10, start=50, run to 2000 labels, print every 200
uv run agents/al_agent.py \
    --input data/labeled.csv \
    --text-col text \
    --label-col label \
    --strategy entropy \
    --n-start 50 \
    --n-iterations 195 \
    --batch-size 10 \
    --compare \
    --output-dir data 2>&1 | grep -E "^\s*(Iter|[-]+|\s*[0-9])" | awk 'NR<=2 || ($2 ~ /^[0-9]+$/ && ($2 == 50 || $2 % 200 == 0))'
```

> **Recommended parameters:**
> - `--n-start 50` — initial seed set (first row of the table)
> - `--batch-size 10` — fine-grained steps, best curve resolution
> - `--n-iterations 195` — covers 50→2000 labels (195 × 10 = 1950 added)
> - Display: first row = 50 labels, then every 200

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

## Results table format

After the run, print a comparison table — every 200 labels:

| N labeled | Entropy F1 | Random F1 | Δ |
|-----------|-----------|-----------|------|
| 200 | ... | ... | ... |
| 400 | ... | ... | ... |
| ... | | | |

Interpret:
- **Δ > 0.04** at 600+ labels = entropy is significantly better
- **Entropy F1 at N** ≈ **Random F1 at 3–4×N** = typical savings range
- Plateau: if Δ stops growing, model (TF-IDF + LogReg) has hit its ceiling — switch to multilingual BERT for further gains

---

## Rules
- Always compare against random baseline (`--compare`)
- Both strategies must start from the **same seed set** — enforced by the agent automatically
- Test set is held out before the AL cycle starts — never touched during training
- Report savings: how many fewer labels needed to reach same F1
- Preferred display: step=10, print every 200 labels (first row = n-start)
