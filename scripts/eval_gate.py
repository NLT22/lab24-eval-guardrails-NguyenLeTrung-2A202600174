"""CI eval gate for Lab 24.

Fails with exit code 1 when any configured metric is below its threshold.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def parse_thresholds(values: list[str]) -> dict[str, float]:
    thresholds = {}
    for value in values:
        key, raw = value.split("=", 1)
        thresholds[key.strip()] = float(raw)
    return thresholds


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", default="phase-a/ragas_summary.json")
    parser.add_argument("--threshold", action="append", default=[])
    args = parser.parse_args()
    summary = json.loads(Path(args.summary).read_text(encoding="utf-8"))
    thresholds = parse_thresholds(args.threshold) or {
        "faithfulness": 0.75,
        "answer_relevancy": 0.70,
        "context_precision": 0.60,
        "context_recall": 0.65,
    }
    failed = []
    for metric, threshold in thresholds.items():
        score = float(summary.get(metric, 0.0))
        print(f"{metric}: {score:.4f} threshold={threshold:.4f}")
        if score < threshold:
            failed.append((metric, score, threshold))
    if failed:
        print("Eval gate failed:")
        for metric, score, threshold in failed:
            print(f"- {metric}: {score:.4f} < {threshold:.4f}")
        return 1
    print("Eval gate passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
