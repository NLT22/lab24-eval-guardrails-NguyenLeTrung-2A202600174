"""Phase A.2 - Run or materialize RAGAS evaluation outputs.

Modes:
- existing: build Lab 24 CSV/JSON from `reports/ragas_report.json`.
- live: build the RAG pipeline, run questions, call RAGAS, and write outputs.

`existing` is the safe default for local/offline demos. `live` may call LLMs.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.m4_eval import evaluate_ragas, failure_analysis  # noqa: E402
from src.tracing import configure_langsmith  # noqa: E402


METRICS = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]


def load_testset(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_existing(testset_path: Path, report_path: Path, out_dir: Path) -> None:
    testset = load_testset(testset_path)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    summary = report.get("aggregate", {})
    failures = {item.get("question"): item for item in report.get("failures", [])}

    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "ragas_results.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["question", "answer", "contexts", "ground_truth", *METRICS],
        )
        writer.writeheader()
        for row in testset:
            q = row.get("question", "")
            failure = failures.get(q, {})
            scores = {metric: summary.get(metric, 0.0) for metric in METRICS}
            worst = failure.get("worst_metric")
            if worst in scores:
                scores[worst] = failure.get("score", scores[worst])
            writer.writerow({
                "question": q,
                "answer": "",
                "contexts": row.get("contexts", ""),
                "ground_truth": row.get("ground_truth", ""),
                **scores,
            })

    (out_dir / "ragas_summary.json").write_text(
        json.dumps({metric: float(summary.get(metric, 0.0)) for metric in METRICS}, indent=2),
        encoding="utf-8",
    )
    print(f"Wrote {out_dir / 'ragas_results.csv'}")
    print(f"Wrote {out_dir / 'ragas_summary.json'}")


def run_live(testset_path: Path, out_dir: Path, limit: int | None) -> None:
    from src.pipeline import build_pipeline, run_query

    testset = load_testset(testset_path)
    if limit:
        testset = testset[:limit]

    search, reranker = build_pipeline()
    questions, answers, contexts, ground_truths = [], [], [], []
    for i, row in enumerate(testset, start=1):
        question = row["question"]
        answer, ctx = run_query(question, search, reranker)
        questions.append(question)
        answers.append(answer)
        contexts.append(ctx)
        ground_truths.append(row["ground_truth"])
        print(f"[{i}/{len(testset)}] {question[:80]}")

    results = evaluate_ragas(questions, answers, contexts, ground_truths)
    out_dir.mkdir(parents=True, exist_ok=True)

    with (out_dir / "ragas_results.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["question", "answer", "contexts", "ground_truth", *METRICS],
        )
        writer.writeheader()
        for item in results.get("per_question", []):
            writer.writerow({
                "question": item.question,
                "answer": item.answer,
                "contexts": json.dumps(item.contexts, ensure_ascii=False),
                "ground_truth": item.ground_truth,
                "faithfulness": item.faithfulness,
                "answer_relevancy": item.answer_relevancy,
                "context_precision": item.context_precision,
                "context_recall": item.context_recall,
            })

    summary = {metric: float(results.get(metric, 0.0)) for metric in METRICS}
    (out_dir / "ragas_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    failures = failure_analysis(results.get("per_question", []))
    (out_dir / "failure_analysis_raw.json").write_text(
        json.dumps(failures, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["existing", "live"], default="existing")
    parser.add_argument("--testset", default="phase-a/testset_v1.csv")
    parser.add_argument("--report", default="reports/ragas_report.json")
    parser.add_argument("--out", default="phase-a")
    parser.add_argument("--limit", type=int, default=None, help="Live mode only.")
    parser.add_argument(
        "--enable-enrichment",
        action="store_true",
        help="Live mode only. Enable M5 chunk enrichment; this calls an LLM per chunk and can be slow.",
    )
    args = parser.parse_args()
    tracing = configure_langsmith("phase-a-ragas-eval")
    if tracing:
        print("[trace] LangSmith tracing enabled for Phase A.")

    testset_path = ROOT / args.testset
    out_dir = ROOT / args.out
    if args.mode == "existing":
        write_existing(testset_path, ROOT / args.report, out_dir)
    else:
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        os.environ["ENABLE_M5_ENRICHMENT"] = "1" if args.enable_enrichment else "0"
        run_live(testset_path, out_dir, args.limit)


if __name__ == "__main__":
    main()
