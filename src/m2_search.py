"""Module 2: Hybrid Search - BM25 (Vietnamese) + Dense + RRF."""

import math
import os
import re
import sys
import warnings
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (  # noqa: E402
    BM25_TOP_K,
    COLLECTION_NAME,
    DENSE_TOP_K,
    EMBEDDING_DIM,
    HYBRID_TOP_K,
    QDRANT_HOST,
    QDRANT_PORT,
)
from src.llm_client import embed_query, embed_texts


@dataclass
class SearchResult:
    text: str
    score: float
    metadata: dict
    method: str  # "bm25", "dense", "hybrid"


def segment_vietnamese(text: str) -> str:
    """Segment Vietnamese text into words for lexical retrieval."""
    text = (text or "").strip()
    if not text:
        return ""

    try:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r"builtin type .* has no __module__ attribute",
                category=DeprecationWarning,
            )
            from underthesea import word_tokenize

            segmented = word_tokenize(text, format="text")
        return segmented if isinstance(segmented, str) else " ".join(segmented)
    except Exception:
        return " ".join(re.findall(r"\w+", text.lower(), flags=re.UNICODE))


def _tokenize(text: str) -> list[str]:
    return [token.lower() for token in segment_vietnamese(text).split() if token.strip()]


class _SimpleBM25:
    """Small BM25 fallback used when rank_bm25 is unavailable."""

    def __init__(self, corpus_tokens: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.corpus_tokens = corpus_tokens
        self.k1 = k1
        self.b = b
        self.doc_len = [len(doc) for doc in corpus_tokens]
        self.avgdl = sum(self.doc_len) / len(self.doc_len) if self.doc_len else 0.0
        self.doc_freqs: list[dict[str, int]] = []
        self.idf: dict[str, float] = {}
        self._initialize()

    def _initialize(self) -> None:
        document_frequency: dict[str, int] = {}
        for doc in self.corpus_tokens:
            frequencies: dict[str, int] = {}
            for token in doc:
                frequencies[token] = frequencies.get(token, 0) + 1
            self.doc_freqs.append(frequencies)
            for token in frequencies:
                document_frequency[token] = document_frequency.get(token, 0) + 1

        n_docs = len(self.corpus_tokens)
        for token, freq in document_frequency.items():
            self.idf[token] = math.log(1 + (n_docs - freq + 0.5) / (freq + 0.5))

    def get_scores(self, query_tokens: list[str]) -> list[float]:
        scores = [0.0] * len(self.corpus_tokens)
        if not query_tokens or not self.avgdl:
            return scores

        for i, frequencies in enumerate(self.doc_freqs):
            doc_len = self.doc_len[i]
            for token in query_tokens:
                tf = frequencies.get(token, 0)
                if not tf:
                    continue
                denominator = tf + self.k1 * (1 - self.b + self.b * doc_len / self.avgdl)
                scores[i] += self.idf.get(token, 0.0) * tf * (self.k1 + 1) / denominator
        return scores


class BM25Search:
    def __init__(self):
        self.corpus_tokens = []
        self.documents = []
        self.bm25 = None

    def index(self, chunks: list[dict]) -> None:
        """Build a BM25 index from chunk dictionaries."""
        self.documents = list(chunks or [])
        self.corpus_tokens = [_tokenize(chunk.get("text", "")) for chunk in self.documents]

        if not self.corpus_tokens:
            self.bm25 = None
            return

        try:
            from rank_bm25 import BM25Okapi

            self.bm25 = BM25Okapi(self.corpus_tokens)
        except Exception:
            self.bm25 = _SimpleBM25(self.corpus_tokens)

    def search(self, query: str, top_k: int = BM25_TOP_K) -> list[SearchResult]:
        """Search the indexed chunks using BM25."""
        if self.bm25 is None or not self.documents or top_k <= 0:
            return []

        tokenized_query = _tokenize(query)
        if not tokenized_query:
            return []

        scores = list(self.bm25.get_scores(tokenized_query))
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

        return [
            SearchResult(
                text=self.documents[idx].get("text", ""),
                score=float(scores[idx]),
                metadata=dict(self.documents[idx].get("metadata", {})),
                method="bm25",
            )
            for idx in top_indices
        ]


class DenseSearch:
    """Dense vector search with Qdrant first and in-memory fallback."""

    def __init__(self):
        self.client = None
        self._local_documents: list[dict] = []
        self._local_vectors: list[list[float]] = []

        try:
            from qdrant_client import QdrantClient

            self.client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
        except Exception:
            self.client = None

    def index(self, chunks: list[dict], collection: str = COLLECTION_NAME) -> None:
        """Index chunks into Qdrant, falling back to local vectors if needed."""
        chunks = list(chunks or [])
        self._local_documents = chunks
        self._local_vectors = []
        if not chunks:
            return

        texts = [chunk.get("text", "") for chunk in chunks]
        vector_lists = embed_texts(texts)

        if self.client is None:
            self._local_vectors = vector_lists
            return

        try:
            from qdrant_client.models import Distance, PointStruct, VectorParams

            self.client.recreate_collection(
                collection_name=collection,
                vectors_config=VectorParams(size=len(vector_lists[0]) if vector_lists else EMBEDDING_DIM, distance=Distance.COSINE),
            )
            points = [
                PointStruct(
                    id=i,
                    vector=vector,
                    payload={**chunk.get("metadata", {}), "text": chunk.get("text", "")},
                )
                for i, (chunk, vector) in enumerate(zip(chunks, vector_lists))
            ]
            self.client.upsert(collection_name=collection, points=points)
        except Exception:
            self._local_vectors = vector_lists

    def search(
        self,
        query: str,
        top_k: int = DENSE_TOP_K,
        collection: str = COLLECTION_NAME,
    ) -> list[SearchResult]:
        """Search using dense vectors."""
        if top_k <= 0:
            return []

        query_vector = embed_query(query)

        if self.client is not None:
            try:
                hits = self.client.search(
                    collection_name=collection,
                    query_vector=query_vector,
                    limit=top_k,
                )
                return [
                    SearchResult(
                        text=(hit.payload or {}).get("text", ""),
                        score=float(hit.score),
                        metadata=dict(hit.payload or {}),
                        method="dense",
                    )
                    for hit in hits
                ]
            except Exception:
                pass

        if not self._local_documents or not self._local_vectors:
            return []

        scored = [
            (_cosine_similarity(query_vector, vector), doc)
            for doc, vector in zip(self._local_documents, self._local_vectors)
        ]
        scored.sort(key=lambda item: item[0], reverse=True)

        return [
            SearchResult(
                text=doc.get("text", ""),
                score=float(score),
                metadata=dict(doc.get("metadata", {})),
                method="dense",
            )
            for score, doc in scored[:top_k]
        ]


def reciprocal_rank_fusion(
    results_list: list[list[SearchResult]],
    k: int = 60,
    top_k: int = HYBRID_TOP_K,
) -> list[SearchResult]:
    """Merge ranked lists using RRF: score(d) = sum(1 / (k + rank))."""
    fused: dict[str, dict] = {}

    for results in results_list:
        for rank, result in enumerate(results or [], start=1):
            if not result.text:
                continue
            if result.text not in fused:
                fused[result.text] = {"score": 0.0, "best": result}
            fused[result.text]["score"] += 1.0 / (k + rank)
            if result.score > fused[result.text]["best"].score:
                fused[result.text]["best"] = result

    ranked = sorted(fused.values(), key=lambda item: item["score"], reverse=True)[:top_k]
    return [
        SearchResult(
            text=item["best"].text,
            score=float(item["score"]),
            metadata=dict(item["best"].metadata),
            method="hybrid",
        )
        for item in ranked
    ]


class HybridSearch:
    """Combines BM25 + Dense search using reciprocal rank fusion."""

    def __init__(self):
        self.bm25 = BM25Search()
        self.dense = DenseSearch()

    def index(self, chunks: list[dict]) -> None:
        self.bm25.index(chunks)
        self.dense.index(chunks)

    def search(self, query: str, top_k: int = HYBRID_TOP_K) -> list[SearchResult]:
        bm25_results = self.bm25.search(query, top_k=BM25_TOP_K)
        dense_results = self.dense.search(query, top_k=DENSE_TOP_K)
        return reciprocal_rank_fusion([bm25_results, dense_results], top_k=top_k)


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if not norm_a or not norm_b:
        return 0.0
    return dot / (norm_a * norm_b)


if __name__ == "__main__":
    sample = "Nhan vien duoc nghi phep nam"
    print(f"Original:  {sample}")
    print(f"Segmented: {segment_vietnamese(sample)}")
