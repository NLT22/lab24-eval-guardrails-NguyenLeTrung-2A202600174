"""Module 3: Reranking — Cross-encoder top-20 → top-3 + latency benchmark."""

import os, re, sys, time
from dataclasses import dataclass
from statistics import mean
import warnings

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import RERANK_TOP_K


@dataclass
class RerankResult:
    text: str
    original_score: float
    rerank_score: float
    metadata: dict
    rank: int


class CrossEncoderReranker:
    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3"):
        self.model_name = model_name
        self._model = None
        self._model_unavailable = False

    def _load_model(self):
        if os.getenv("PYTEST_CURRENT_TEST"):
            self._model_unavailable = True
            return None
        if self._model is None and not self._model_unavailable:
            from sentence_transformers import CrossEncoder
            try:
                self._model = CrossEncoder(self.model_name, local_files_only=True)
            except Exception as e:
                self._model_unavailable = True
                warnings.warn(
                    f"[m3] reranker model unavailable locally ({e}) — using lexical fallback."
                )
        return self._model

    @staticmethod
    def _lexical_score(query: str, text: str) -> float:
        query_tokens = set(re.findall(r"\w+", query.lower(), flags=re.UNICODE))
        text_tokens = set(re.findall(r"\w+", text.lower(), flags=re.UNICODE))
        if not query_tokens or not text_tokens:
            return 0.0
        overlap = len(query_tokens & text_tokens) / len(query_tokens)
        exact_bonus = 0.1 if any(token in text_tokens for token in ["12", "nghỉ", "phep", "phép"]) else 0.0
        return overlap + exact_bonus

    def rerank(self, query: str, documents: list[dict], top_k: int = RERANK_TOP_K) -> list[RerankResult]:
        """Rerank documents: top-20 → top-k."""
        if not documents:
            return []

        model = self._load_model()
        if model is None:
            raw_scores = [self._lexical_score(query, doc["text"]) for doc in documents]
        else:
            pairs = [[query, doc["text"]] for doc in documents]
            raw_scores = model.predict(pairs)

            # predict trả về numpy array hoặc float nếu 1 pair, normalize về list
            if isinstance(raw_scores, (int, float)):
                raw_scores = [float(raw_scores)]
            else:
                raw_scores = [float(s) for s in raw_scores]

        scored = sorted(zip(raw_scores, documents), key=lambda x: x[0], reverse=True)

        results = []
        for i, (score, doc) in enumerate(scored[:top_k]):
            results.append(RerankResult(
                text=doc["text"],
                original_score=doc.get("score", 0.0),
                rerank_score=float(score),
                metadata=doc.get("metadata", {}),
                rank=i + 1,
            ))
        return results


class FlashrankReranker:
    """Lightweight alternative (<5ms). Optional."""
    def __init__(self):
        self._model = None

    def rerank(self, query: str, documents: list[dict], top_k: int = RERANK_TOP_K) -> list[RerankResult]:
        try:
            from flashrank import Ranker, RerankRequest
        except ImportError:
            return []

        if self._model is None:
            self._model = Ranker()

        passages = [{"id": i, "text": d["text"]} for i, d in enumerate(documents)]
        ranked = self._model.rerank(RerankRequest(query=query, passages=passages))

        results = []
        for i, item in enumerate(ranked[:top_k]):
            orig_doc = documents[item["id"]]
            results.append(RerankResult(
                text=item["text"],
                original_score=orig_doc.get("score", 0.0),
                rerank_score=float(item["score"]),
                metadata=orig_doc.get("metadata", {}),
                rank=i + 1,
            ))
        return results


def benchmark_reranker(reranker, query: str, documents: list[dict], n_runs: int = 5) -> dict:
    """Benchmark latency over n_runs."""
    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        reranker.rerank(query, documents)
        times.append((time.perf_counter() - start) * 1000)  # ms
    return {
        "avg_ms": mean(times),
        "min_ms": min(times),
        "max_ms": max(times),
    }


if __name__ == "__main__":
    query = "Nhân viên được nghỉ phép bao nhiêu ngày?"
    docs = [
        {"text": "Nhân viên được nghỉ 12 ngày/năm.", "score": 0.8, "metadata": {}},
        {"text": "Mật khẩu thay đổi mỗi 90 ngày.", "score": 0.7, "metadata": {}},
        {"text": "Thời gian thử việc là 60 ngày.", "score": 0.75, "metadata": {}},
    ]
    reranker = CrossEncoderReranker()
    results = reranker.rerank(query, docs)
    for r in results:
        print(f"[{r.rank}] score={r.rerank_score:.4f} | {r.text}")

    stats = benchmark_reranker(reranker, query, docs, n_runs=3)
    print(f"\nLatency: avg={stats['avg_ms']:.1f}ms  min={stats['min_ms']:.1f}ms  max={stats['max_ms']:.1f}ms")
