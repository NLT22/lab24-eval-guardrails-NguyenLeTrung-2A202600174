"""Phase B.3 - Cohen's kappa calibration."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

from sklearn.metrics import cohen_kappa_score


def normalize(label: str) -> str:
    label = (label or "tie").strip().lower()
    if label in {"a", "answer_a"}:
        return "A"
    if label in {"b", "answer_b"}:
        return "B"
    return "tie"


def ensure_human_template(pairwise_path: Path, human_path: Path, n: int = 10) -> None:
    if human_path.exists():
        return
    with pairwise_path.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))[:n]
    human_path.parent.mkdir(parents=True, exist_ok=True)
    with human_path.open("w", encoding="utf-8", newline="") as f:
        fields = ["question_id", "question", "human_winner", "confidence", "notes"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for i, row in enumerate(rows, start=1):
            writer.writerow({
                "question_id": i,
                "question": row["question"],
                "human_winner": "tie",
                "confidence": "low",
                "notes": "Fill manually before final calibration.",
            })


def interpret(kappa: float) -> str:
    if math.isnan(kappa):
        return "Not enough label diversity to compute kappa"
    if kappa < 0:
        return "Worse than chance"
    if kappa < 0.2:
        return "Slight agreement"
    if kappa < 0.4:
        return "Fair agreement"
    if kappa < 0.6:
        return "Moderate agreement"
    if kappa < 0.8:
        return "Substantial agreement"
    return "Almost perfect agreement"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairwise", default="phase-b/pairwise_results.csv")
    parser.add_argument("--human", default="phase-b/human_labels.csv")
    parser.add_argument("--out", default="phase-b/kappa_report.md")
    args = parser.parse_args()

    pairwise_path = Path(args.pairwise)
    human_path = Path(args.human)
    ensure_human_template(pairwise_path, human_path)

    with pairwise_path.open(encoding="utf-8", newline="") as f:
        judge_rows = list(csv.DictReader(f))
    with human_path.open(encoding="utf-8", newline="") as f:
        human_rows = list(csv.DictReader(f))

    n = min(len(judge_rows), len(human_rows))
    human = [normalize(r.get("human_winner", "tie")) for r in human_rows[:n]]
    judge = [normalize(r.get("winner_after_swap", "tie")) for r in judge_rows[:n]]
    labels_seen = set(human) | set(judge)
    kappa = cohen_kappa_score(human, judge, labels=["A", "B", "tie"]) if n and len(labels_seen) > 1 else float("nan")
    label_counts = {label: {"human": human.count(label), "judge": judge.count(label)} for label in ["A", "B", "tie"]}
    text = (
        "# Cohen's Kappa Calibration\n\n"
        f"- Samples: {n}\n"
        f"- Kappa: {'nan' if math.isnan(kappa) else f'{kappa:.3f}'}\n"
        f"- Interpretation: {interpret(kappa)}\n\n"
        "## Label Counts\n\n"
        "| Label | Human | Judge |\n"
        "|---|---:|---:|\n"
        + "\n".join(f"| {label} | {counts['human']} | {counts['judge']} |" for label, counts in label_counts.items())
        + "\n\n"
        "If kappa is below 0.6, inspect length bias, style bias, and mismatched tie policy.\n"
        "If kappa is `nan`, manually label more examples with at least two distinct labels.\n"
    )
    Path(args.out).write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
