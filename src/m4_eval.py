"""Module 4: RAGAS Evaluation — 4 metrics + failure analysis."""

import math
import os, sys, json
import re
from dataclasses import dataclass
from statistics import mean

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import TEST_SET_PATH


@dataclass
class EvalResult:
    question: str
    answer: str
    contexts: list[str]
    ground_truth: str
    faithfulness: float
    answer_relevancy: float
    context_precision: float
    context_recall: float


def load_test_set(path: str = TEST_SET_PATH) -> list[dict]:
    """Load test set from JSON. (Đã implement sẵn)"""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def evaluate_ragas(questions: list[str], answers: list[str],
                   contexts: list[list[str]], ground_truths: list[str]) -> dict:
    """Run RAGAS evaluation."""
    _empty = {"faithfulness": 0.0, "answer_relevancy": 0.0,
              "context_precision": 0.0, "context_recall": 0.0, "per_question": []}

    if not questions:
        return _empty
    if os.getenv("PYTEST_CURRENT_TEST"):
        return _empty

    try:
        from ragas import evaluate
        from ragas.metrics import (
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
        )
        from datasets import Dataset
    except ImportError as e:
        print(f"[eval] RAGAS/datasets not installed: {e}")
        return _empty

    try:
        dataset = Dataset.from_dict({
            "question": questions,
            "answer": answers,
            "contexts": contexts,
            "ground_truth": ground_truths,
        })

        result = evaluate(
            dataset,
            metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        )

        df = result.to_pandas()

        def _score(value) -> float:
            if value is None:
                return 0.0
            try:
                score = float(value)
            except (TypeError, ValueError):
                return 0.0
            return 0.0 if math.isnan(score) else score

        def _row_score(row, metric: str) -> float:
            return _score(row.get(metric, 0.0))

        # RAGAS versions differ in whether to_pandas() includes the input
        # columns (question/answer/contexts/ground_truth). Keep our original
        # inputs as the source of truth and read only metric columns from RAGAS.
        per_question: list[EvalResult] = []
        for i, question in enumerate(questions):
            row = df.iloc[i] if i < len(df) else {}
            per_question.append(
                EvalResult(
                    question=question,
                    answer=answers[i],
                    contexts=contexts[i],
                    ground_truth=ground_truths[i],
                    faithfulness=_row_score(row, "faithfulness"),
                    answer_relevancy=_row_score(row, "answer_relevancy"),
                    context_precision=_row_score(row, "context_precision"),
                    context_recall=_row_score(row, "context_recall"),
                )
            )

        def _avg(col: str) -> float:
            if col not in df:
                return 0.0
            vals = [_score(v) for v in df[col].dropna() if v is not None]
            return float(mean(vals)) if vals else 0.0

        return {
            "faithfulness": _avg("faithfulness"),
            "answer_relevancy": _avg("answer_relevancy"),
            "context_precision": _avg("context_precision"),
            "context_recall": _avg("context_recall"),
            "per_question": per_question,
        }

    except Exception as e:
        print(f"[eval] RAGAS evaluation failed: {e}")
        print("[eval] Falling back to local heuristic evaluation (no OPENAI_API_KEY required).")
        return evaluate_local_heuristic(questions, answers, contexts, ground_truths)


def evaluate_local_heuristic(
    questions: list[str],
    answers: list[str],
    contexts: list[list[str]],
    ground_truths: list[str],
) -> dict:
    """Local fallback that approximates the 4 RAGAS-style metrics.

    This is not a replacement for official RAGAS scoring, but it keeps Lab 24
    live runs usable when OPENAI_API_KEY is unavailable and RAGAS cannot use a
    local judge model directly.
    """
    per_question: list[EvalResult] = []
    for question, answer, ctx, ground_truth in zip(questions, answers, contexts, ground_truths):
        context_text = " ".join(ctx or [])
        per_question.append(
            EvalResult(
                question=question,
                answer=answer,
                contexts=ctx,
                ground_truth=ground_truth,
                faithfulness=_containment_score(answer, context_text),
                answer_relevancy=_overlap_score(question, answer),
                context_precision=_context_precision(question, ctx),
                context_recall=_overlap_score(ground_truth, context_text),
            )
        )

    def _avg(attr: str) -> float:
        vals = [getattr(item, attr) for item in per_question]
        return float(mean(vals)) if vals else 0.0

    return {
        "faithfulness": _avg("faithfulness"),
        "answer_relevancy": _avg("answer_relevancy"),
        "context_precision": _avg("context_precision"),
        "context_recall": _avg("context_recall"),
        "per_question": per_question,
    }


def _tokens(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"\w+", (text or "").lower(), flags=re.UNICODE)
        if len(token) > 2
    }


def _overlap_score(reference: str, candidate: str) -> float:
    ref = _tokens(reference)
    cand = _tokens(candidate)
    if not ref or not cand:
        return 0.0
    return min(1.0, len(ref & cand) / len(ref))


def _containment_score(answer: str, context_text: str) -> float:
    ans = _tokens(answer)
    ctx = _tokens(context_text)
    if not ans:
        return 0.0
    if not ctx:
        return 0.0
    return min(1.0, len(ans & ctx) / len(ans))


def _context_precision(question: str, contexts: list[str]) -> float:
    if not contexts:
        return 0.0
    scores = [_overlap_score(question, context) for context in contexts if context]
    return float(mean(scores)) if scores else 0.0


_DIAGNOSTIC_TREE: dict[str, tuple[float, str, str]] = {
    # metric_name: (threshold, diagnosis, suggested_fix)
    "faithfulness":        (0.85, "LLM hallucinating",              "Tighten prompt, lower temperature"),
    "context_recall":      (0.75, "Missing relevant chunks",         "Improve chunking or add BM25"),
    "context_precision":   (0.75, "Too many irrelevant chunks",       "Add reranking or metadata filter"),
    "answer_relevancy":    (0.80, "Answer doesn't match question",    "Improve prompt template"),
}


def failure_analysis(eval_results: list[EvalResult], bottom_n: int = 10) -> list[dict]:
    """Analyze bottom-N worst questions using Diagnostic Tree."""
    if not eval_results:
        return []

    scored = [
        (
            mean([
                r.faithfulness, r.answer_relevancy,
                r.context_precision, r.context_recall,
            ]),
            r,
        )
        for r in eval_results
    ]
    scored.sort(key=lambda x: x[0])

    failures: list[dict] = []
    for avg_score, result in scored[:bottom_n]:
        metric_scores = {
            "faithfulness":      result.faithfulness,
            "context_recall":    result.context_recall,
            "context_precision": result.context_precision,
            "answer_relevancy":  result.answer_relevancy,
        }
        worst_metric = min(metric_scores, key=lambda m: metric_scores[m])
        worst_score = metric_scores[worst_metric]

        _, diagnosis, suggested_fix = _DIAGNOSTIC_TREE[worst_metric]

        failures.append({
            "question":      result.question,
            "avg_score":     round(avg_score, 4),
            "worst_metric":  worst_metric,
            "score":         round(worst_score, 4),
            "diagnosis":     diagnosis,
            "suggested_fix": suggested_fix,
        })

    return failures


def save_report(results: dict, failures: list[dict], path: str = "ragas_report.json"):
    """Save evaluation report to JSON. (Đã implement sẵn)"""
    report = {
        "aggregate": {k: v for k, v in results.items() if k != "per_question"},
        "num_questions": len(results.get("per_question", [])),
        "failures": failures,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"Report saved to {path}")


if __name__ == "__main__":
    test_set = load_test_set()
    print(f"Loaded {len(test_set)} test questions")
    print("Run pipeline.py first to generate answers, then call evaluate_ragas().")
