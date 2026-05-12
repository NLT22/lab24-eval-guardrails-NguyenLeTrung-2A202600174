"""OpenAI-compatible chat and embedding clients with local fallbacks."""

from __future__ import annotations

import hashlib
import os
import warnings
from functools import lru_cache
from typing import Iterable

from config import (
    EMBEDDING_MODEL,
    EMBEDDING_PROVIDER,
    LMSTUDIO_BASE_URL,
    LMSTUDIO_CHAT_TIMEOUT,
    LMSTUDIO_EMBEDDING_MODEL,
    LMSTUDIO_EMBEDDING_TIMEOUT,
    LMSTUDIO_MODEL,
    LLM_PROVIDER,
    OPENAI_API_KEY,
)


def _provider(value: str, allowed: set[str], default: str = "auto") -> str:
    value = (value or default).strip().lower()
    return value if value in allowed else default


def _openai_client(*, base_url: str | None = None, api_key: str | None = None, timeout: float = 30.0):
    from openai import OpenAI

    kwargs = {"api_key": api_key or OPENAI_API_KEY or "lm-studio", "timeout": timeout, "max_retries": 0}
    if base_url:
        kwargs["base_url"] = base_url
        if "localhost" in base_url or "127.0.0.1" in base_url:
            try:
                import httpx

                kwargs["http_client"] = httpx.Client(trust_env=False, timeout=timeout)
            except Exception:
                pass
    return OpenAI(**kwargs)


def _chat_provider_order(provider: str | None = None) -> list[str]:
    selected = _provider(provider or LLM_PROVIDER, {"auto", "openai", "lmstudio"})
    if selected == "openai":
        return ["openai"]
    if selected == "lmstudio":
        return ["lmstudio"]
    return ["openai", "lmstudio"] if OPENAI_API_KEY else ["lmstudio"]


def _embedding_provider_order(provider: str | None = None) -> list[str]:
    selected = _provider(
        provider or EMBEDDING_PROVIDER,
        {"auto", "openai", "lmstudio", "sentence_transformers"},
    )
    if selected != "auto":
        return [selected]
    return ["openai", "lmstudio", "sentence_transformers"] if OPENAI_API_KEY else ["lmstudio", "sentence_transformers"]


def chat_completion(
    messages: list[dict[str, str]],
    *,
    model: str | None = None,
    provider: str | None = None,
    temperature: float = 0.1,
    max_tokens: int = 512,
) -> str | None:
    """Return assistant text from OpenAI or LM Studio, or None if unavailable."""
    if os.getenv("PYTEST_CURRENT_TEST"):
        return None

    for candidate in _chat_provider_order(provider):
        try:
            if candidate == "openai":
                if not OPENAI_API_KEY:
                    continue
                client = _openai_client(timeout=30.0)
                chosen_model = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
            elif candidate == "lmstudio":
                client = _openai_client(base_url=LMSTUDIO_BASE_URL, api_key="lm-studio", timeout=LMSTUDIO_CHAT_TIMEOUT)
                chosen_model = model or LMSTUDIO_MODEL
            else:
                continue

            resp = client.chat.completions.create(
                model=chosen_model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            content = resp.choices[0].message.content
            return content.strip() if content else ""
        except Exception as exc:
            warnings.warn(f"[llm_client] chat provider {candidate} failed: {exc}")
    return None


def embed_texts(
    texts: str | Iterable[str],
    *,
    model: str | None = None,
    provider: str | None = None,
) -> list[list[float]]:
    """Embed one or more texts using OpenAI, LM Studio, or sentence-transformers."""
    input_texts = [texts] if isinstance(texts, str) else list(texts)
    if not input_texts:
        return []
    if os.getenv("PYTEST_CURRENT_TEST") and _provider(
        provider or EMBEDDING_PROVIDER,
        {"auto", "openai", "lmstudio", "sentence_transformers"},
    ) in {"auto", "sentence_transformers"}:
        return _hash_embeddings(input_texts)

    errors: list[str] = []
    for candidate in _embedding_provider_order(provider):
        try:
            if candidate == "openai":
                if not OPENAI_API_KEY:
                    continue
                return _embed_openai_compatible(
                    input_texts,
                    model=model or os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"),
                )
            if candidate == "lmstudio":
                return _embed_openai_compatible(
                    input_texts,
                    model=model or LMSTUDIO_EMBEDDING_MODEL,
                    base_url=LMSTUDIO_BASE_URL,
                    api_key="lm-studio",
                    timeout=LMSTUDIO_EMBEDDING_TIMEOUT,
                )
            if candidate == "sentence_transformers":
                return _embed_sentence_transformers(input_texts, model=model or EMBEDDING_MODEL)
        except Exception as exc:
            errors.append(f"{candidate}: {exc}")
            warnings.warn(f"[llm_client] embedding provider {candidate} failed: {exc}")

    if os.getenv("PYTEST_CURRENT_TEST"):
        return _hash_embeddings(input_texts)
    raise RuntimeError("No embedding provider available. Tried: " + "; ".join(errors))


def embed_query(text: str, **kwargs) -> list[float]:
    vectors = embed_texts([text], **kwargs)
    return vectors[0] if vectors else []


def _embed_openai_compatible(
    texts: list[str],
    *,
    model: str,
    base_url: str | None = None,
    api_key: str | None = None,
    timeout: float = 15.0,
) -> list[list[float]]:
    client = _openai_client(base_url=base_url, api_key=api_key, timeout=timeout)
    resp = client.embeddings.create(model=model, input=texts)
    ordered = sorted(resp.data, key=lambda item: item.index)
    return [list(item.embedding) for item in ordered]


@lru_cache(maxsize=4)
def _sentence_transformer(model: str):
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model)


def _embed_sentence_transformers(texts: list[str], *, model: str) -> list[list[float]]:
    if os.getenv("PYTEST_CURRENT_TEST"):
        return _hash_embeddings(texts)
    vectors = _sentence_transformer(model).encode(texts, show_progress_bar=False)
    return [vector.tolist() if hasattr(vector, "tolist") else list(vector) for vector in vectors]


def _hash_embeddings(texts: list[str], dim: int = 16) -> list[list[float]]:
    """Deterministic tiny vectors for offline unit tests only."""
    vectors: list[list[float]] = []
    for text in texts:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        values = [((digest[i % len(digest)] / 255.0) * 2.0) - 1.0 for i in range(dim)]
        vectors.append(values)
    return vectors
