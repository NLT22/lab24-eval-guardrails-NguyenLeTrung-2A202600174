"""Phase A.3 - Build failure_analysis.md from ragas_results.csv."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


METRICS = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]


def score(row: dict, metric: str) -> float:
    try:
        return float(row.get(metric, 0.0) or 0.0)
    except ValueError:
        return 0.0


def cluster_for(row: dict) -> tuple[str, str]:
    scores = {metric: score(row, metric) for metric in METRICS}
    worst = min(scores, key=scores.get)
    if worst == "faithfulness":
        return "C1", "Faithfulness / hallucination failures"
    if worst == "answer_relevancy":
        return "C2", "Answer relevancy failures"
    if worst == "context_precision":
        return "C3", "Irrelevant retrieval context"
    return "C4", "Missing context / low recall"


def analyze(input_path: Path, output_path: Path, bottom_n: int) -> None:
    with input_path.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    for row in rows:
        row["avg"] = sum(score(row, metric) for metric in METRICS) / len(METRICS)
        row["cluster"], row["cluster_name"] = cluster_for(row)
    rows.sort(key=lambda item: item["avg"])
    bottom = rows[:bottom_n]

    lines = [
        "# Failure Cluster Analysis",
        "",
        "## Bottom 10 Questions",
        "",
        "| # | Question | F | AR | CP | CR | Avg | Cluster |",
        "|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for i, row in enumerate(bottom, start=1):
        q = row.get("question", "").replace("|", " ")[:90]
        lines.append(
            f"| {i} | {q} | {score(row, 'faithfulness'):.3f} | "
            f"{score(row, 'answer_relevancy'):.3f} | {score(row, 'context_precision'):.3f} | "
            f"{score(row, 'context_recall'):.3f} | {row['avg']:.3f} | {row['cluster']} |"
        )

    clusters = {}
    for row in bottom:
        clusters.setdefault(row["cluster"], {"name": row["cluster_name"], "examples": []})
        clusters[row["cluster"]]["examples"].append(row.get("question", ""))

    lines.extend(["", "## Clusters Identified", ""])
    for cid, data in sorted(clusters.items()):
        lines.extend([
            f"### {cid}: {data['name']}",
            "",
            "**Pattern:** Related metric is the weakest signal in the bottom questions.",
            "",
            "**Examples:**",
        ])
        for example in data["examples"][:2]:
            lines.append(f"- {example}")
        lines.extend([
            "",
            "**Proposed fix:** Tune retrieval/reranking and tighten answer grounding prompts. "
            "For low recall, increase top-k or use parent chunks; for low precision, add metadata filters.",
            "",
        ])

    output_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="phase-a/ragas_results.csv")
    parser.add_argument("--output", default="phase-a/failure_analysis.md")
    parser.add_argument("--bottom-n", type=int, default=10)
    args = parser.parse_args()
    analyze(Path(args.input), Path(args.output), args.bottom_n)


if __name__ == "__main__":
    main()
