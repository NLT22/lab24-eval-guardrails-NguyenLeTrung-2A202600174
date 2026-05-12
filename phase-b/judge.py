"""Phase B - LLM-as-Judge pipeline.

Creates:
- phase-b/pairwise_results.csv
- phase-b/absolute_scores.csv
- phase-b/judge_bias_report.md

The judge uses src.llm_client.chat_completion(), so it works with OpenAI or
LM Studio. If no model is reachable, deterministic local heuristics are used so
the pipeline still produces reviewable artifacts.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.llm_client import chat_completion  # noqa: E402
from src.tracing import configure_langsmith  # noqa: E402


PAIRWISE_PROMPT = """You are an impartial evaluator. Compare two answers to the same question.
Question: {question}
Answer A: {answer_a}
Answer B: {answer_b}
Rate based on factual accuracy, relevance to the question, and conciseness.
Output JSON only:
{{"winner": "A" or "B" or "tie", "reason": "..."}}
"""

ABSOLUTE_PROMPT = """Score the answer on 4 dimensions, each 1-5 scale:
1. Factual accuracy
2. Relevance
3. Conciseness
4. Helpfulness
Question: {question}
Answer: {answer}
Output JSON only:
{{"accuracy": int, "relevance": int, "conciseness": int, "helpfulness": int, "overall": float}}
"""


def parse_judge_output(text: str | None) -> dict:
    if not text:
        return {}
    cleaned = text.replace("```json", "").replace("```", "").strip()
    match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
    if match:
        cleaned = match.group(0)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        return {}


def judge_once(question: str, answer_a: str, answer_b: str) -> dict:
    out = chat_completion(
        [{"role": "user", "content": PAIRWISE_PROMPT.format(question=question, answer_a=answer_a, answer_b=answer_b)}],
        temperature=0.0,
        max_tokens=220,
    )
    parsed = parse_judge_output(out)
    winner = str(parsed.get("winner", "tie")).strip().upper()
    if winner not in {"A", "B", "TIE"}:
        winner = "TIE"
    return {"winner": winner.lower() if winner == "TIE" else winner, "reason": parsed.get("reason", "Parsed/fallback result")}


def heuristic_pairwise(question: str, answer_a: str, answer_b: str) -> dict:
    # Lightweight fallback: prefer an answer that is closer in length to a concise
    # factual response and contains more tokens from the question.
    q_tokens = set(re.findall(r"\w+", question.lower()))

    def score(answer: str) -> float:
        tokens = re.findall(r"\w+", answer.lower())
        overlap = len(q_tokens & set(tokens))
        length_penalty = abs(len(tokens) - 55) / 100
        return overlap - length_penalty

    sa, sb = score(answer_a), score(answer_b)
    if abs(sa - sb) < 0.15:
        return {"winner": "tie", "reason": "Heuristic scores are close."}
    return {"winner": "A" if sa > sb else "B", "reason": "Heuristic fallback based on relevance/length."}


def pairwise_judge_with_swap(question: str, ans1: str, ans2: str) -> dict:
    run1 = judge_once(question, ans1, ans2) or heuristic_pairwise(question, ans1, ans2)
    run2_raw = judge_once(question, ans2, ans1) or heuristic_pairwise(question, ans2, ans1)
    run2 = dict(run2_raw)
    if run2["winner"] == "A":
        run2["winner"] = "B"
    elif run2["winner"] == "B":
        run2["winner"] = "A"

    final = run1["winner"] if run1["winner"] == run2["winner"] else "tie"
    return {
        "run1_winner": run1["winner"],
        "run2_winner": run2["winner"],
        "winner_after_swap": final,
        "run1_reason": run1.get("reason", ""),
        "run2_reason": run2.get("reason", ""),
    }


def absolute_score(question: str, answer: str) -> dict:
    out = chat_completion(
        [{"role": "user", "content": ABSOLUTE_PROMPT.format(question=question, answer=answer)}],
        temperature=0.0,
        max_tokens=220,
    )
    parsed = parse_judge_output(out)
    dims = ["accuracy", "relevance", "conciseness", "helpfulness"]
    if not all(dim in parsed for dim in dims):
        token_count = len(re.findall(r"\w+", answer))
        fallback = {
            "accuracy": 3,
            "relevance": 4 if token_count else 1,
            "conciseness": 5 if token_count <= 90 else 3,
            "helpfulness": 4 if token_count else 1,
        }
        fallback["overall"] = sum(fallback[d] for d in dims) / 4
        return fallback
    scores = {dim: max(1, min(5, int(float(parsed[dim])))) for dim in dims}
    scores["overall"] = float(parsed.get("overall") or sum(scores[d] for d in dims) / 4)
    return scores


def load_rows(path: Path, limit: int) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    return rows[:limit] if limit else rows


def answer_variants(row: dict) -> tuple[str, str]:
    ground_truth = row.get("ground_truth", "")
    answer = row.get("answer", "") or ground_truth
    contexts = row.get("contexts", "")
    if contexts and contexts not in {"[]", ""}:
        baseline = contexts[:800]
    else:
        baseline = "Không tìm thấy thông tin đầy đủ trong context."
    return answer, baseline


def write_pairwise(rows: list[dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["question", "answer_a", "answer_b", "winner_after_swap", "run1_winner", "run2_winner", "run1_reason", "run2_reason"]
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            a, b = answer_variants(row)
            result = pairwise_judge_with_swap(row["question"], a, b)
            writer.writerow({"question": row["question"], "answer_a": a, "answer_b": b, **result})


def write_absolute(rows: list[dict], out_path: Path) -> None:
    fields = ["question", "answer", "accuracy", "relevance", "conciseness", "helpfulness", "overall"]
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            answer, _ = answer_variants(row)
            scores = absolute_score(row["question"], answer)
            writer.writerow({"question": row["question"], "answer": answer, **scores})


def write_bias_report(pairwise_path: Path, out_path: Path) -> None:
    rows = load_rows(pairwise_path, 0)
    total = len(rows) or 1
    run1_a = sum(1 for r in rows if r["run1_winner"] == "A")
    final_a = sum(1 for r in rows if r["winner_after_swap"] == "A")
    final_b = sum(1 for r in rows if r["winner_after_swap"] == "B")
    ties = sum(1 for r in rows if r["winner_after_swap"] == "tie")
    longer_wins = 0
    longer_cases = 0
    for r in rows:
        len_a, len_b = len(r["answer_a"]), len(r["answer_b"])
        if len_a == len_b or r["winner_after_swap"] == "tie":
            continue
        longer_cases += 1
        if (len_a > len_b and r["winner_after_swap"] == "A") or (len_b > len_a and r["winner_after_swap"] == "B"):
            longer_wins += 1
    longer_rate = longer_wins / longer_cases if longer_cases else 0.0

    out_path.write_text(
        "# Judge Bias Report\n\n"
        "## Position Bias\n\n"
        f"- A wins when listed first: {run1_a}/{total} ({run1_a / total:.1%}).\n"
        "- Mitigation used: swap-and-average; final winner becomes tie if the two orders disagree.\n\n"
        "## Length Bias\n\n"
        f"- Longer answer wins: {longer_wins}/{longer_cases or 1} ({longer_rate:.1%}).\n\n"
        "## Final Winner Distribution\n\n"
        "| Winner | Count |\n|---|---:|\n"
        f"| A | {final_a} |\n| B | {final_b} |\n| tie | {ties} |\n\n"
        "## Mitigation Strategy\n\n"
        "- Keep swap-and-average for pairwise comparisons.\n"
        "- Add rubric-based absolute scoring for monitoring.\n"
        "- Manually calibrate with Cohen's kappa before trusting judge scores in CI.\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="phase-a/ragas_results.csv")
    parser.add_argument("--out", default="phase-b")
    parser.add_argument("--limit", type=int, default=30)
    args = parser.parse_args()
    tracing = configure_langsmith("phase-b-llm-judge")
    if tracing:
        print("[trace] LangSmith tracing enabled for Phase B.")

    rows = load_rows(ROOT / args.input, args.limit)
    out_dir = ROOT / args.out
    write_pairwise(rows, out_dir / "pairwise_results.csv")
    write_absolute(rows, out_dir / "absolute_scores.csv")
    write_bias_report(out_dir / "pairwise_results.csv", out_dir / "judge_bias_report.md")
    print(f"Wrote Phase B outputs to {out_dir}")


if __name__ == "__main__":
    main()
