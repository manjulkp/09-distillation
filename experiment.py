"""Compare a tiny student trained alone with students distilled from the teacher."""

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from distill import Student, build_vocab, encode, student_accuracy, train_student

PROJECT_7 = Path("../07-finetune-reviews")
SPLITS = ["yelp_train", "yelp_val", "yelp_unlabeled", "yelp_test", "amazon_test"]

VARIANTS = {
    "Hard labels only": {"distill": False, "extra": False},
    "Distilled, temperature 1": {"distill": True, "extra": False, "temperature": 1.0},
    "Distilled, temperature 2": {"distill": True, "extra": False, "temperature": 2.0},
    "Distilled, temperature 5": {"distill": True, "extra": False, "temperature": 5.0},
    "Distilled + 6,000 unlabeled reviews": {"distill": True, "extra": True, "temperature": 2.0},
}

plt.rcParams.update(
    {
        "font.family": ["Segoe UI", "DejaVu Sans"],
        "figure.facecolor": "#fcfcfb",
        "axes.facecolor": "#fcfcfb",
        "axes.edgecolor": "#c3c2b7",
        "axes.labelcolor": "#52514e",
        "axes.titlelocation": "left",
        "axes.titlesize": 12,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.color": "#52514e",
        "ytick.color": "#52514e",
        "grid.color": "#e1e0d9",
        "grid.linewidth": 0.7,
    }
)


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def teacher_logits_for(rows_by_split, cache_path, teacher_dir, log=print):
    """Run the slow teacher once and save its outputs, so reruns are instant."""
    if cache_path.exists():
        return torch.load(cache_path, weights_only=True)

    # Project 7's helper handles batching by length.
    sys.path.insert(0, str(PROJECT_7.resolve()))
    from finetune import predict_logits

    tokenizer = AutoTokenizer.from_pretrained(teacher_dir)
    model = AutoModelForSequenceClassification.from_pretrained(teacher_dir).eval()
    results = {"parameters": sum(p.numel() for p in model.parameters())}
    for split, rows in rows_by_split.items():
        started = time.perf_counter()
        results[split] = predict_logits(model, tokenizer, rows)
        seconds = time.perf_counter() - started
        log(f"Teacher finished {split}: {len(rows) / seconds:.0f} reviews per second")
        if split == "yelp_test":
            results["reviews_per_second"] = len(rows) / seconds
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(results, cache_path)
    return results


def plot(rows, teacher, path):
    rows = sorted(rows, key=lambda r: r["val_mean"])
    y = range(len(rows))
    fig, ax = plt.subplots(figsize=(9.5, 0.5 * len(rows) + 1.7))
    ax.hlines(y, [r["test_min"] for r in rows], [r["test_max"] for r in rows], color="#2a78d6", lw=1.6, alpha=0.45)
    ax.scatter([r["test_mean"] for r in rows], y, color="#2a78d6", s=46, edgecolors="#fcfcfb", linewidths=1.5, zorder=3)
    ax.axvline(teacher, color="#898781", lw=0.9)
    ax.annotate(f"Teacher: {teacher:.1%}", (teacher, len(rows) - 0.55), xytext=(-5, 0), textcoords="offset points", ha="right", color="#52514e", fontsize=9)
    ax.set_yticks(list(y), [r["name"] for r in rows])
    ax.xaxis.set_major_formatter(lambda value, _: f"{value:.1%}")
    ax.grid(axis="x")
    ax.set_axisbelow(True)
    ax.tick_params(length=0)
    ax.margins(x=0.1, y=0.2)
    ax.set_xlabel("Yelp test accuracy")
    ax.set_title("Student accuracy: dot is the mean, line is the range across seeds")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default=str(PROJECT_7 / "data"))
    parser.add_argument("--teacher", default=str(PROJECT_7 / "outputs" / "best_model"))
    parser.add_argument("--out", default="outputs")
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=20)
    args = parser.parse_args(argv)

    out = Path(args.out)
    if not (Path(args.teacher) / "config.json").exists():
        parser.error(f"{args.teacher} not found. Run Project 7's experiment.py first.")
    rows = {split: read_jsonl(Path(args.data) / f"{split}.jsonl") for split in SPLITS}
    teacher = teacher_logits_for(rows, out / "teacher_logits.pt", args.teacher, log=lambda t: print(t, flush=True))

    labels = {s: torch.tensor([r["label"] for r in rows[s]]) for s in SPLITS}
    teacher_acc = {
        s: (teacher[s].argmax(-1) == labels[s]).float().mean().item() for s in ("yelp_val", "yelp_test", "amazon_test")
    }
    # The vocabulary may include unlabeled reviews, because no labels are involved.
    vocab = build_vocab(rows["yelp_train"] + rows["yelp_unlabeled"])
    ids = {s: encode(rows[s], vocab) for s in SPLITS}

    results = []
    student = None
    for name, settings in VARIANTS.items():
        train_ids, train_labels = ids["yelp_train"], labels["yelp_train"]
        logits = teacher["yelp_train"] if settings["distill"] else None
        if settings["extra"]:
            train_ids = torch.cat([train_ids, ids["yelp_unlabeled"]])
            train_labels = torch.cat([train_labels, torch.full((len(rows["yelp_unlabeled"]),), -1)])
            logits = torch.cat([logits, teacher["yelp_unlabeled"]])

        vals, tests, amazons, agreements = [], [], [], []
        for seed in range(args.seeds):
            torch.manual_seed(seed)
            student = Student(len(vocab) + 2)
            val = train_student(
                student, train_ids, train_labels, ids["yelp_val"], labels["yelp_val"], logits,
                temperature=settings.get("temperature", 2.0), epochs=args.epochs, seed=seed,
            )  # fmt: skip
            vals.append(val)
            tests.append(student_accuracy(student, ids["yelp_test"], labels["yelp_test"]))
            amazons.append(student_accuracy(student, ids["amazon_test"], labels["amazon_test"]))
            agreements.append(student_accuracy(student, ids["yelp_test"], teacher["yelp_test"].argmax(-1)))
        results.append(
            {
                "name": name,
                "val_mean": statistics.mean(vals),
                "test_mean": statistics.mean(tests),
                "test_min": min(tests),
                "test_max": max(tests),
                "amazon_mean": statistics.mean(amazons),
                "agreement_mean": statistics.mean(agreements),
            }
        )
        print(results[-1], flush=True)

    student.eval()
    started = time.perf_counter()
    student_accuracy(student, ids["yelp_test"], labels["yelp_test"])
    student_speed = len(ids["yelp_test"]) / (time.perf_counter() - started)
    student_parameters = sum(p.numel() for p in student.parameters())

    results.sort(key=lambda row: -row["val_mean"])
    lines = [
        "# Distillation results",
        "",
        "| Model | Parameters | Reviews per second |",
        "| --- | --- | --- |",
        f"| Teacher (DistilBERT) | {teacher['parameters']:,} | {teacher['reviews_per_second']:,.0f} |",
        f"| Student | {student_parameters:,} | {student_speed:,.0f} |",
        "",
        f"The teacher scored {teacher_acc['yelp_val']:.1%} on validation, {teacher_acc['yelp_test']:.1%} on Yelp test "
        f"and {teacher_acc['amazon_test']:.1%} on Amazon test.",
        "",
        f"Students, averaged over {args.seeds} seeds and sorted by validation accuracy. "
        "Agreement is how often a student makes the same prediction as the teacher on Yelp test.",
        "",
        "| Student training | Validation | Yelp test | Seed range | Amazon test | Agreement with teacher |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in results:
        lines.append(
            f"| {row['name']} | {row['val_mean']:.1%} | {row['test_mean']:.1%} | {row['test_min']:.1%} to {row['test_max']:.1%} | "
            f"{row['amazon_mean']:.1%} | {row['agreement_mean']:.1%} |"
        )
    (out / "results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (out / "results.json").write_text(json.dumps({"students": results, "teacher": teacher_acc}, indent=1))
    plot(results, teacher_acc["yelp_test"], out / "comparison.png")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
