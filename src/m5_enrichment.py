"""
Module 5: Enrichment Pipeline
==============================
Làm giàu chunks TRƯỚC khi embed: Summarize, HyQA, Contextual Prepend, Auto Metadata.

Test: pytest tests/test_m5.py
"""

import json
import os
import sys
import warnings
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import LLM_PROVIDER, LMSTUDIO_BASE_URL, LMSTUDIO_MODEL, OPENAI_API_KEY


def _get_openai_client():
    """Return an OpenAI client if API key is available, else None."""
    if os.getenv("PYTEST_CURRENT_TEST"):
        return None
    try:
        from openai import OpenAI
        if OPENAI_API_KEY and LLM_PROVIDER in {"auto", "openai", ""}:
            return OpenAI(api_key=OPENAI_API_KEY, timeout=5.0, max_retries=0)
        if LLM_PROVIDER in {"auto", "lmstudio", ""}:
            return OpenAI(
                api_key="lm-studio",
                base_url=LMSTUDIO_BASE_URL,
                timeout=5.0,
                max_retries=0,
            )
        return None
    except ImportError:
        warnings.warn("[m5] openai package not installed — LLM enrichment disabled.")
        return None

def _chat_model() -> str:
    if OPENAI_API_KEY and LLM_PROVIDER in {"auto", "openai", ""}:
        return os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    return LMSTUDIO_MODEL


@dataclass
class EnrichedChunk:
    """Chunk đã được làm giàu."""
    original_text: str
    enriched_text: str
    summary: str
    hypothesis_questions: list[str]
    auto_metadata: dict
    method: str  # "contextual", "summary", "hyqa", "full"


# ─── Technique 1: Chunk Summarization ────────────────────


def summarize_chunk(text: str) -> str:
    """
    Tạo summary ngắn cho chunk.
    Embed summary thay vì (hoặc cùng với) raw chunk → giảm noise.

    Args:
        text: Raw chunk text.

    Returns:
        Summary string (2-3 câu). Trả về extractive summary nếu không có API key.
    """
    client = _get_openai_client()
    if client:
        try:
            resp = client.chat.completions.create(
                model=_chat_model(),
                messages=[
                    {
                        "role": "system",
                        "content": "Tóm tắt đoạn văn sau trong 2-3 câu ngắn gọn bằng tiếng Việt. Chỉ trả về phần tóm tắt.",
                    },
                    {"role": "user", "content": text},
                ],
                max_tokens=150,
                temperature=0.3,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            warnings.warn(f"[m5] summarize_chunk failed: {e}")

    # Fallback: extractive — lấy 2 câu đầu
    sentences = [s.strip() for s in text.replace("\n", " ").split(". ") if s.strip()]
    if not sentences:
        return text
    extract = ". ".join(sentences[:2])
    if not extract.endswith("."):
        extract += "."
    return extract


# ─── Technique 2: Hypothesis Question-Answer (HyQA) ─────


def generate_hypothesis_questions(text: str, n_questions: int = 3) -> list[str]:
    """
    Generate câu hỏi mà chunk có thể trả lời.
    Index cả questions lẫn chunk → query match tốt hơn (bridge vocabulary gap).

    Args:
        text: Raw chunk text.
        n_questions: Số câu hỏi cần generate.

    Returns:
        List of question strings. Trả về [] nếu không có API key.
    """
    client = _get_openai_client()
    if not client:
        return []

    try:
        resp = client.chat.completions.create(
            model=_chat_model(),
            messages=[
                {
                    "role": "system",
                    "content": (
                        f"Dựa trên đoạn văn, tạo đúng {n_questions} câu hỏi bằng tiếng Việt "
                        f"mà đoạn văn có thể trả lời. "
                        f"Trả về mỗi câu hỏi trên 1 dòng, không đánh số."
                    ),
                },
                {"role": "user", "content": text},
            ],
            max_tokens=200,
            temperature=0.5,
        )
        raw = resp.choices[0].message.content.strip().split("\n")
        questions = [
            q.strip().lstrip("0123456789.-) ").strip()
            for q in raw
            if q.strip()
        ]
        return questions[:n_questions]
    except Exception as e:
        warnings.warn(f"[m5] generate_hypothesis_questions failed: {e}")
        return []


# ─── Technique 3: Contextual Prepend (Anthropic style) ──


def contextual_prepend(text: str, document_title: str = "") -> str:
    """
    Prepend context giải thích chunk nằm ở đâu trong document.
    Anthropic benchmark: giảm 49% retrieval failure (alone).

    Args:
        text: Raw chunk text.
        document_title: Tên document gốc.

    Returns:
        Text với context prepended. Trả về text gốc nếu không có API key.
    """
    client = _get_openai_client()
    if not client:
        return text

    try:
        doc_hint = f"Tài liệu: {document_title}\n\n" if document_title else ""
        resp = client.chat.completions.create(
            model=_chat_model(),
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Viết 1 câu ngắn bằng tiếng Việt mô tả đoạn văn này nằm ở đâu "
                        "trong tài liệu và nói về chủ đề gì. Chỉ trả về đúng 1 câu."
                    ),
                },
                {"role": "user", "content": f"{doc_hint}Đoạn văn:\n{text}"},
            ],
            max_tokens=80,
            temperature=0.3,
        )
        context_sentence = resp.choices[0].message.content.strip()
        return f"{context_sentence}\n\n{text}"
    except Exception as e:
        warnings.warn(f"[m5] contextual_prepend failed: {e}")
        return text


# ─── Technique 4: Auto Metadata Extraction ──────────────


def extract_metadata(text: str) -> dict:
    """
    LLM extract metadata tự động: topic, entities, date_range, category.

    Args:
        text: Raw chunk text.

    Returns:
        Dict with extracted metadata fields. Trả về {} nếu không có API key.
    """
    client = _get_openai_client()
    if not client:
        return {}

    try:
        resp = client.chat.completions.create(
            model=_chat_model(),
            messages=[
                {
                    "role": "system",
                    "content": (
                        'Trích xuất metadata từ đoạn văn. '
                        'Trả về JSON hợp lệ với các trường: '
                        '{"topic": "string", "entities": ["string"], '
                        '"date_range": "string hoặc null nếu không có", '
                        '"category": "legal|finance|policy|hr|it|other"}. '
                        'Chỉ trả về JSON, không giải thích thêm.'
                    ),
                },
                {"role": "user", "content": text},
            ],
            max_tokens=150,
            temperature=0.1,
        )
        raw = resp.choices[0].message.content.strip()
        # Strip markdown code fences if present
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        return json.loads(raw.strip())
    except Exception as e:
        warnings.warn(f"[m5] extract_metadata failed: {e}")
        return {}


# ─── Full Enrichment Pipeline ────────────────────────────


def enrich_chunks(
    chunks: list[dict],
    methods: list[str] | None = None,
) -> list[EnrichedChunk]:
    """
    Chạy enrichment pipeline trên danh sách chunks.

    Args:
        chunks: List of {"text": str, "metadata": dict}
        methods: List of methods to apply. Default: ["contextual", "hyqa", "metadata"]
                 Options: "summary", "hyqa", "contextual", "metadata", "full"

    Returns:
        List of EnrichedChunk objects.
    """
    if os.getenv("ENABLE_M5_ENRICHMENT", "0") != "1":
        return [
            EnrichedChunk(
                original_text=chunk["text"],
                enriched_text=chunk["text"],
                summary="",
                hypothesis_questions=[],
                auto_metadata=dict(chunk.get("metadata", {})),
                method="raw",
            )
            for chunk in chunks
        ]

    if methods is None:
        methods = ["contextual", "hyqa", "metadata"]

    use_summary = "summary" in methods or "full" in methods
    use_hyqa = "hyqa" in methods or "full" in methods
    use_contextual = "contextual" in methods or "full" in methods
    use_metadata = "metadata" in methods or "full" in methods

    enriched: list[EnrichedChunk] = []

    for chunk in chunks:
        text: str = chunk["text"]
        meta: dict = chunk.get("metadata", {})
        source: str = meta.get("source", "")

        # 1. Summary (optional, not in default methods)
        summary = summarize_chunk(text) if use_summary else ""

        # 2. Contextual prepend: thêm 1 câu context vào đầu chunk
        enriched_text = contextual_prepend(text, source) if use_contextual else text

        # 3. HyQA: generate câu hỏi và nối vào cuối enriched_text để index
        questions: list[str] = []
        if use_hyqa:
            questions = generate_hypothesis_questions(text)
            if questions:
                enriched_text = enriched_text + "\n\n" + "\n".join(questions)

        # 4. Auto metadata extraction
        auto_meta = {**meta}
        if use_metadata:
            extracted = extract_metadata(text)
            if extracted:
                auto_meta = {**meta, **extracted}

        enriched.append(EnrichedChunk(
            original_text=text,
            enriched_text=enriched_text,
            summary=summary,
            hypothesis_questions=questions,
            auto_metadata=auto_meta,
            method="+".join(methods),
        ))

    return enriched


# ─── Main ────────────────────────────────────────────────

if __name__ == "__main__":
    sample = "Nhân viên chính thức được nghỉ phép năm 12 ngày làm việc mỗi năm. Số ngày nghỉ phép tăng thêm 1 ngày cho mỗi 5 năm thâm niên công tác."

    print("=== Enrichment Pipeline Demo ===\n")
    print(f"Original: {sample}\n")

    s = summarize_chunk(sample)
    print(f"Summary: {s}\n")

    qs = generate_hypothesis_questions(sample)
    print(f"HyQA questions: {qs}\n")

    ctx = contextual_prepend(sample, "Sổ tay nhân viên VinUni 2024")
    print(f"Contextual: {ctx}\n")

    meta = extract_metadata(sample)
    print(f"Auto metadata: {meta}\n")

    result = enrich_chunks(
        [{"text": sample, "metadata": {"source": "sample.md"}}],
        methods=["contextual", "hyqa", "metadata"],
    )
    if result:
        print(f"EnrichedChunk:\n  enriched_text: {result[0].enriched_text[:200]}")
        print(f"  auto_metadata: {result[0].auto_metadata}")
