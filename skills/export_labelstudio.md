# Skill: export_to_labelstudio(df)

## Purpose
Export labeled DataFrame to LabelStudio import JSON format.
Includes model predictions as pre-annotations and flags low-confidence rows separately.

---

## When to use
- After `auto_label` to send data for human review in LabelStudio
- To create a review task for a teammate or classmate

---

## Output format (LabelStudio import JSON)

```json
[
  {
    "id": 0,
    "data": {"text": "This movie was amazing!"},
    "meta": {"confidence": 0.99, "flagged_for_review": false},
    "predictions": [
      {
        "model_version": "annotation_agent_v1",
        "score": 0.99,
        "result": [
          {
            "id": "result_0",
            "type": "choices",
            "value": {"choices": ["positive"]},
            "from_name": "sentiment",
            "to_name": "text"
          }
        ]
      }
    ]
  }
]
```

---

## LabelStudio label config (paste when creating project)

```xml
<View>
  <Text name="text" value="$text"/>
  <Choices name="sentiment" toName="text" choice="single">
    <Choice value="positive"/>
    <Choice value="negative"/>
    <Choice value="neutral"/>
  </Choices>
</View>
```

---

## CLI usage

```bash
uv run agents/annotation_agent.py \
    --input data/labeled.csv \
    --output-dir data
# → data/labelstudio_import.json
# → data/low_confidence.csv  (flagged rows for manual review)
```

---

## Human-in-the-loop (BONUS)

Rows with `confidence < threshold` are automatically saved to `data/low_confidence.csv`.
These are the examples the model is uncertain about — send them for priority human review.

---

## Rules
- JSON must be a list of task dicts — do not wrap in another object
- `from_name` and `to_name` must match the label config exactly
- Never include private data in `data` field — only the content column
- Import via LabelStudio UI: Projects → Import → upload JSON file
