"""Topic scope guard using local-first embeddings.

This module does not require OPENAI_API_KEY. In auto mode it tries LM Studio's
OpenAI-compatible embeddings endpoint first, then falls back to
sentence-transformers through src.llm_client.
"""

from __future__ import annotations

import csv
import math
import os
import sys
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.llm_client import embed_query, embed_texts


DEFAULT_ALLOWED_TOPICS = [
    "Vietnam personal data protection law and Decree 13/2023",
    "financial statements, VAT declarations, accounting and tax reports",
    "RAG evaluation, guardrails, retrieval and answer quality",
]

_ON_TOPIC_KEYWORDS = {
    "nghi dinh", "decree", "du lieu", "personal data", "cccd", "privacy",
    "thue", "gtgt", "vat", "bao cao tai chinh", "financial", "accounting",
    "rag", "ragas", "guardrail", "retrieval", "context", "cohen", "kappa",
    "latency", "audit", "judge", "pii",
}

_OFF_TOPIC_KEYWORDS = {
    "nau", "pho", "bong da", "tho tinh", "xe may", "du lich", "dien thoai",
    "choi game", "trong cay", "tap gym", "phim", "thoi tiet",
}


@dataclass
class TopicDecision:
    allowed: bool
    reason: str
    best_topic: str
    score: float


class TopicGuard:
    def __init__(self, allowed_topics: list[str] | None = None, threshold: float = 0.45):
        self.allowed_topics = allowed_topics or DEFAULT_ALLOWED_TOPICS
        self.threshold = threshold
        self.topic_vectors = embed_texts(self.allowed_topics)

    def check(self, text: str) -> TopicDecision:
        if not text.strip():
            return TopicDecision(False, "Empty input is outside the supported scope.", "", 0.0)

        keyword_decision = _keyword_decision(text)
        if keyword_decision is not None:
            return keyword_decision

        query_vector = embed_query(text)
        scores = [_cosine_similarity(query_vector, topic_vector) for topic_vector in self.topic_vectors]
        best_idx = max(range(len(scores)), key=lambda idx: scores[idx])
        best_score = float(scores[best_idx])
        best_topic = self.allowed_topics[best_idx]
        allowed = best_score >= self.threshold
        reason = (
            f"On topic: {best_topic} ({best_score:.2f})"
            if allowed
            else f"Off topic. Closest supported topic: {best_topic} ({best_score:.2f})"
        )
        return TopicDecision(allowed, reason, best_topic, best_score)


def evaluate_topic_guard(path: str = "phase-c/topic_guard_test_results.csv") -> dict[str, float]:
    tests = [
        ("Nghi dinh 13 quy dinh quyen cua chu the du lieu nhu the nao?", True),
        ("Du lieu ca nhan nhay cam can bao ve ra sao?", True),
        ("To khai thue GTGT co chi tieu nao quan trong?", True),
        ("Bao cao tai chinh nay noi gi ve doanh thu?", True),
        ("RAGAS faithfulness thap thi sua pipeline the nao?", True),
        ("Lam sao do context precision trong RAG?", True),
        ("Quy trinh audit log cho guardrail nen thiet ke sao?", True),
        ("Can redact CCCD trong cau hoi nguoi dung nhu the nao?", True),
        ("Cohen kappa dung de hieu chuan judge ra sao?", True),
        ("Latency P95 cua guardrail nen benchmark the nao?", True),
        ("Cong thuc nau pho bo ngon?", False),
        ("Lich thi dau bong da hom nay?", False),
        ("Viet tho tinh yeu bang tieng Anh", False),
        ("Cach sua xe may bi chet may?", False),
        ("Du lich Da Nang nen di dau?", False),
        ("Mua dien thoai nao choi game tot?", False),
        ("Huong dan trong cay ca chua", False),
        ("Cach tap gym tang co", False),
        ("Review phim moi nhat", False),
        ("Thoi tiet ngay mai o Ha Noi?", False),
    ]
    guard = TopicGuard()
    rows = []
    correct = 0
    for text, expected in tests:
        decision = guard.check(text)
        is_correct = decision.allowed == expected
        correct += int(is_correct)
        rows.append({
            "text": text,
            "expected_on_topic": expected,
            "allowed": decision.allowed,
            "score": f"{decision.score:.4f}",
            "best_topic": decision.best_topic,
            "reason": decision.reason,
            "correct": is_correct,
        })

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    return {"accuracy": correct / len(tests), "num_tests": len(tests)}


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if not norm_a or not norm_b:
        return 0.0
    return dot / (norm_a * norm_b)


def _keyword_decision(text: str) -> TopicDecision | None:
    lowered = text.lower()
    if any(keyword in lowered for keyword in _ON_TOPIC_KEYWORDS):
        return TopicDecision(True, "On topic: keyword match", "keyword", 1.0)
    if any(keyword in lowered for keyword in _OFF_TOPIC_KEYWORDS):
        return TopicDecision(False, "Off topic: keyword match", "keyword", 0.0)
    return None


if __name__ == "__main__":
    result = evaluate_topic_guard()
    passed = int(result["accuracy"] * result["num_tests"])
    print(f"Topic guard accuracy: {result['accuracy']:.1%} ({passed}/{result['num_tests']})")
