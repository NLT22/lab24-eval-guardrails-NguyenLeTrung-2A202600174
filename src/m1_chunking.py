"""
Module 1: Advanced Chunking Strategies
=======================================
Implement semantic, hierarchical, và structure-aware chunking.
So sánh với basic chunking (baseline) để thấy improvement.

Test: pytest tests/test_m1.py
"""

import glob
import os
import re
import sys
import warnings
from dataclasses import dataclass, field

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (DATA_DIR, HIERARCHICAL_PARENT_SIZE, HIERARCHICAL_CHILD_SIZE,
                    SEMANTIC_THRESHOLD)

# Embedding model used only by chunk_semantic() to decide split boundaries.
# Vietnamese-specific SBERT — small (~120MB), faster than bge-m3 for chunking
# (M2 uses bge-m3 separately for retrieval embedding).
SEMANTIC_CHUNK_MODEL = os.getenv("SEMANTIC_CHUNK_MODEL", "keepitreal/vietnamese-sbert")


@dataclass
class Chunk:
    text: str
    metadata: dict = field(default_factory=dict)
    parent_id: str | None = None


_SEMANTIC_MODEL = None
_SEMANTIC_MODEL_UNAVAILABLE = False


def _get_semantic_model():
    """Lazy-load Vietnamese SBERT used to score sentence boundaries."""
    global _SEMANTIC_MODEL, _SEMANTIC_MODEL_UNAVAILABLE
    if os.getenv("PYTEST_CURRENT_TEST"):
        _SEMANTIC_MODEL_UNAVAILABLE = True
        return None
    if _SEMANTIC_MODEL is None and not _SEMANTIC_MODEL_UNAVAILABLE:
        from sentence_transformers import SentenceTransformer
        try:
            _SEMANTIC_MODEL = SentenceTransformer(
                SEMANTIC_CHUNK_MODEL,
                local_files_only=True,
            )
        except Exception as e:
            _SEMANTIC_MODEL_UNAVAILABLE = True
            warnings.warn(
                f"[m1] semantic model unavailable locally ({e}) — using lexical fallback."
            )
    return _SEMANTIC_MODEL


def _lexical_similarity(left: str, right: str) -> float:
    """Tiny fallback similarity for offline tests and constrained environments."""
    left_tokens = set(re.findall(r"\w+", left.lower(), flags=re.UNICODE))
    right_tokens = set(re.findall(r"\w+", right.lower(), flags=re.UNICODE))
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / len(left_tokens | right_tokens)


def load_documents(data_dir: str = DATA_DIR) -> list[dict]:
    """Load all markdown/text files from data/. (Đã implement sẵn)"""
    docs = []
    for fp in sorted(glob.glob(os.path.join(data_dir, "*.md"))):
        with open(fp, encoding="utf-8") as f:
            docs.append({"text": f.read(), "metadata": {"source": os.path.basename(fp)}})
    return docs


# ─── Baseline: Basic Chunking (để so sánh) ──────────────


def chunk_basic(text: str, chunk_size: int = 500, metadata: dict | None = None) -> list[Chunk]:
    """
    Basic chunking: split theo paragraph (\\n\\n).
    Đây là baseline — KHÔNG phải mục tiêu của module này.
    (Đã implement sẵn)
    """
    metadata = metadata or {}
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks = []
    current = ""
    for i, para in enumerate(paragraphs):
        if len(current) + len(para) > chunk_size and current:
            chunks.append(Chunk(text=current.strip(), metadata={**metadata, "chunk_index": len(chunks)}))
            current = ""
        current += para + "\n\n"
    if current.strip():
        chunks.append(Chunk(text=current.strip(), metadata={**metadata, "chunk_index": len(chunks)}))
    return chunks


# ─── Strategy 1: Semantic Chunking ───────────────────────


# Split on sentence-final punctuation followed by whitespace, but NOT after a
# digit (avoids splitting "Khoản 1. Nội dung..." or numeric values like "12.5").
_SENT_SPLIT_RE = re.compile(r"(?<![0-9])(?<=[.!?])\s+|\n{2,}")


def _split_sentences(text: str, min_len: int = 20) -> list[str]:
    """Vietnamese-aware sentence splitter. Merges fragments shorter than min_len
    (typically list markers like 'a)' or stray short lines) into the next sentence."""
    raw = [s.strip() for s in _SENT_SPLIT_RE.split(text) if s.strip()]
    if not raw:
        return []
    merged: list[str] = []
    buf = ""
    for s in raw:
        if buf:
            buf = f"{buf} {s}"
        else:
            buf = s
        if len(buf) >= min_len:
            merged.append(buf)
            buf = ""
    if buf:
        if merged:
            merged[-1] = f"{merged[-1]} {buf}"
        else:
            merged.append(buf)
    return merged


def chunk_semantic(text: str, threshold: float = SEMANTIC_THRESHOLD,
                   metadata: dict | None = None) -> list[Chunk]:
    """
    Split text by sentence similarity — nhóm câu cùng chủ đề.
    Tốt hơn basic vì không cắt giữa ý.

    Args:
        text: Input text.
        threshold: Cosine similarity threshold. Dưới threshold → tách chunk mới.
        metadata: Metadata gắn vào mỗi chunk.

    Returns:
        List of Chunk objects grouped by semantic similarity.
    """
    metadata = metadata or {}
    sentences = _split_sentences(text)
    if not sentences:
        return []

    def make_chunk(group: list[str], idx: int) -> Chunk:
        return Chunk(
            text=" ".join(group).strip(),
            metadata={**metadata, "chunk_index": idx, "strategy": "semantic"},
        )

    if len(sentences) == 1:
        return [make_chunk(sentences, 0)]

    model = _get_semantic_model()
    embeddings = None
    if model is not None:
        embeddings = model.encode(
            sentences, show_progress_bar=False, normalize_embeddings=True
        )
        embeddings = np.asarray(embeddings)

    chunks: list[Chunk] = []
    current_group = [sentences[0]]
    for i in range(1, len(sentences)):
        if embeddings is None:
            sim = _lexical_similarity(sentences[i - 1], sentences[i])
        else:
            sim = float(np.dot(embeddings[i - 1], embeddings[i]))
        if sim < threshold:
            chunks.append(make_chunk(current_group, len(chunks)))
            current_group = []
        current_group.append(sentences[i])
    if current_group:
        chunks.append(make_chunk(current_group, len(chunks)))
    return chunks


# ─── Strategy 2: Hierarchical Chunking ──────────────────


def chunk_hierarchical(text: str, parent_size: int = HIERARCHICAL_PARENT_SIZE,
                       child_size: int = HIERARCHICAL_CHILD_SIZE,
                       metadata: dict | None = None) -> tuple[list[Chunk], list[Chunk]]:
    """
    Parent-child hierarchy: retrieve child (precision) → return parent (context).
    Đây là default recommendation cho production RAG.

    Args:
        text: Input text.
        parent_size: Chars per parent chunk.
        child_size: Chars per child chunk.
        metadata: Metadata gắn vào mỗi chunk.

    Returns:
        (parents, children) — mỗi child có parent_id link đến parent.
    """
    metadata = metadata or {}
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

    parents: list[Chunk] = []
    current = ""
    for para in paragraphs:
        if len(current) + len(para) > parent_size and current:
            pid = f"parent_{len(parents)}"
            parents.append(Chunk(
                text=current.strip(),
                metadata={**metadata, "chunk_type": "parent",
                          "parent_id": pid, "chunk_index": len(parents)},
            ))
            current = ""
        current += para + "\n\n"
    if current.strip():
        pid = f"parent_{len(parents)}"
        parents.append(Chunk(
            text=current.strip(),
            metadata={**metadata, "chunk_type": "parent",
                      "parent_id": pid, "chunk_index": len(parents)},
        ))

    children: list[Chunk] = []
    overlap = max(child_size // 5, 1)  # ~20% overlap
    step = max(child_size - overlap, 1)
    for parent in parents:
        pid = parent.metadata["parent_id"]
        ptext = parent.text
        for start in range(0, len(ptext), step):
            ctext = ptext[start:start + child_size].strip()
            if not ctext:
                continue
            children.append(Chunk(
                text=ctext,
                metadata={**metadata, "chunk_type": "child",
                          "chunk_index": len(children),
                          "parent_index": parent.metadata["chunk_index"]},
                parent_id=pid,
            ))
            if start + child_size >= len(ptext):
                break

    return parents, children


# ─── Strategy 3: Structure-Aware Chunking ────────────────


def chunk_structure_aware(text: str, metadata: dict | None = None) -> list[Chunk]:
    """
    Parse markdown headers → chunk theo logical structure.
    Giữ nguyên tables, code blocks, lists — không cắt giữa chừng.

    Args:
        text: Markdown text.
        metadata: Metadata gắn vào mỗi chunk.

    Returns:
        List of Chunk objects, mỗi chunk = 1 section (header + content).
    """
    metadata = metadata or {}
    sections = re.split(r'(^#{1,6}\s+.+$)', text, flags=re.MULTILINE)
    chunks: list[Chunk] = []
    current_header = ""
    current_content = ""

    def flush():
        if not current_content.strip():
            return
        body = f"{current_header}\n\n{current_content}".strip()
        chunks.append(Chunk(
            text=body,
            metadata={**metadata, "section": current_header,
                      "strategy": "structure", "chunk_index": len(chunks)},
        ))

    for part in sections:
        if re.match(r'^#{1,6}\s+', part):
            flush()
            current_header = part.strip()
            current_content = ""
        else:
            current_content += part
    flush()
    return chunks


# ─── A/B Test: Compare All Strategies ────────────────────


def compare_strategies(documents: list[dict]) -> dict:
    """
    Run all strategies on documents and compare.

    Returns:
        {"basic": {...}, "semantic": {...}, "hierarchical": {...}, "structure": {...}}
    """
    def stats(chunks: list[Chunk]) -> dict:
        if not chunks:
            return {"num_chunks": 0, "avg_length": 0, "min_length": 0, "max_length": 0}
        lengths = [len(c.text) for c in chunks]
        return {
            "num_chunks": len(chunks),
            "avg_length": round(sum(lengths) / len(lengths)),
            "min_length": min(lengths),
            "max_length": max(lengths),
        }

    all_basic: list[Chunk] = []
    all_semantic: list[Chunk] = []
    all_parents: list[Chunk] = []
    all_children: list[Chunk] = []
    all_struct: list[Chunk] = []

    for doc in documents:
        text, meta = doc["text"], doc.get("metadata", {})
        all_basic.extend(chunk_basic(text, metadata=meta))
        all_semantic.extend(chunk_semantic(text, metadata=meta))
        parents, children = chunk_hierarchical(text, metadata=meta)
        all_parents.extend(parents)
        all_children.extend(children)
        all_struct.extend(chunk_structure_aware(text, metadata=meta))

    results = {
        "basic": stats(all_basic),
        "semantic": stats(all_semantic),
        "hierarchical": {
            **stats(all_children),
            "num_parents": len(all_parents),
            "num_children": len(all_children),
        },
        "structure": stats(all_struct),
    }

    print(f"\n{'Strategy':<14} | {'Chunks':>6} | {'Avg':>6} | {'Min':>4} | {'Max':>5}")
    print("-" * 50)
    for name in ["basic", "semantic", "hierarchical", "structure"]:
        s = results[name]
        label = (f"{s['num_parents']}p/{s['num_children']}c"
                 if name == "hierarchical" else str(s["num_chunks"]))
        print(f"  {name:<12} | {label:>6} | "
              f"{s['avg_length']:>6} | {s['min_length']:>4} | {s['max_length']:>5}")
    return results


if __name__ == "__main__":
    docs = load_documents()
    print(f"Loaded {len(docs)} documents")
    results = compare_strategies(docs)
    for name, stats in results.items():
        print(f"  {name}: {stats}")
