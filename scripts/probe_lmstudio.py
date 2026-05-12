"""Probe LM Studio chat and embedding endpoints.

Use this when LM Studio UI says a model is loaded but the app reports
"No models loaded" or embedding failures.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config import LMSTUDIO_BASE_URL, LMSTUDIO_EMBEDDING_MODEL, LMSTUDIO_MODEL  # noqa: E402
from src.llm_client import chat_completion, embed_texts  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default=LMSTUDIO_BASE_URL)
    parser.add_argument("--chat-model", default=LMSTUDIO_MODEL)
    parser.add_argument("--embedding-model", default=LMSTUDIO_EMBEDDING_MODEL)
    args = parser.parse_args()

    os.environ["LMSTUDIO_BASE_URL"] = args.base_url
    print(f"LM Studio base URL: {args.base_url}")
    print(f"Chat model: {args.chat_model}")
    print(f"Embedding model: {args.embedding_model}")

    try:
        import requests

        session = requests.Session()
        session.trust_env = False
        r = session.get(f"{args.base_url.rstrip('/')}/models", timeout=10)
        print("\nLoaded models:")
        print(json.dumps(r.json(), indent=2)[:2000])
    except Exception as exc:
        print(f"\nCould not list models: {exc}")

    print("\nEmbedding probe:")
    try:
        vectors = embed_texts(["hello", "xin chao"], model=args.embedding_model, provider="lmstudio")
        print(f"OK: {len(vectors)} vectors, dim={len(vectors[0]) if vectors else 0}")
    except Exception as exc:
        print(f"FAILED: {exc}")

    print("\nChat probe:")
    try:
        text = chat_completion(
            [{"role": "user", "content": "Reply with exactly: ok"}],
            model=args.chat_model,
            provider="lmstudio",
            max_tokens=10,
            temperature=0,
        )
        print(f"OK: {text!r}")
    except Exception as exc:
        print(f"FAILED: {exc}")


if __name__ == "__main__":
    main()
