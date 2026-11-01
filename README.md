# Project 9: Distillation

Trains a very small student model to imitate the fine-tuned DistilBERT from Project 7.
The student only averages word embeddings and has no attention at all. The question
is how much of the teacher's skill it can pick up.

| Student training | What it learns from |
| --- | --- |
| Hard labels only | The 4,000 labeled Yelp reviews, with no teacher |
| Distilled, temperature 1, 2 or 5 | Those labels plus the teacher's soft probabilities |
| Distilled + 6,000 unlabeled reviews | The same at temperature 2, plus reviews only the teacher labeled |

## Results

**Not available yet.** This experiment hasn't been run, and it needs Project 7's trained
model. When it has run, results will be in `outputs\results.md` and `outputs\comparison.png`.

## Run it

Run Project 7's `experiment.py` first, because this uses its model and data.

```powershell
uv sync
uv run python experiment.py
uv run pytest
```

The teacher's slow predictions are saved to `outputs\teacher_logits.pt`, so later runs
skip that step. Delete the file if you retrain the teacher.

| Option | Default | Meaning |
| --- | --- | --- |
| `--seeds` | `5` | Seeds per student |
| `--epochs` | `20` | Training epochs per student |
| `--teacher` | `../07-finetune-reviews/outputs/best_model` | Teacher model |

## Files

| File | What it does |
| --- | --- |
| `distill.py` | The student, the distillation loss and training |
| `experiment.py` | Runs the teacher once, then compares all the students |
| `tests/` | Includes a check that a student can learn from the teacher with no labels |
