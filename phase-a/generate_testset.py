"""Phase A.1 - Generate Lab 24 testset_v1.csv.

Default mode reuses the existing Day 18 `test_set.json` and normalizes it to
the Lab 24 CSV schema. This keeps the lab runnable without spending LLM calls.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def evolution_type(index: int, total: int) -> str:
    ratio = index / max(total, 1)
    if ratio < 0.50:
        return "simple"
    if ratio < 0.75:
        return "reasoning"
    return "multi_context"


def generate(input_path: Path, output_path: Path) -> None:
    data = json.loads(input_path.read_text(encoding="utf-8"))
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["question", "ground_truth", "contexts", "evolution_type"],
        )
        writer.writeheader()
        for i, item in enumerate(data):
            writer.writerow({
                "question": item.get("question", ""),
                "ground_truth": item.get("ground_truth", ""),
                "contexts": "",
                "evolution_type": evolution_type(i, len(data)),
            })

    notes = output_path.with_name("testset_review_notes.md")
    if not notes.exists():
        notes.write_text(
            "# Test Set Review Notes\n\n"
            "- Source: existing Day 18 `test_set.json`.\n"
            "- Schema normalized for Lab 24 Phase A.\n"
            "- Distribution labels: 50% simple, 25% reasoning, 25% multi_context.\n"
            "- Manual edit note: review at least 10 rows before final submission.\n",
            encoding="utf-8",
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="test_set.json")
    parser.add_argument("--output", default="phase-a/testset_v1.csv")
    args = parser.parse_args()
    generate(Path(args.input), Path(args.output))
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
