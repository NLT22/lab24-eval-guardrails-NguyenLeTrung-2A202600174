"""Shared configuration for Lab 18/24."""

import os
from dotenv import load_dotenv

load_dotenv()

# --- Model providers ---
# OPEN_API_KEY is accepted as a compatibility alias for common typos, but the
# canonical variable is OPENAI_API_KEY.
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "") or os.getenv("OPEN_API_KEY", "")
COHERE_API_KEY = os.getenv("COHERE_API_KEY", "")
HF_TOKEN = os.getenv("HF_TOKEN", "") or os.getenv("HUGGINGFACEHUB_API_TOKEN", "")
LANGSMITH_TRACING = os.getenv("LANGSMITH_TRACING", os.getenv("LANGCHAIN_TRACING_V2", "false")).strip().lower()
LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY", os.getenv("LANGCHAIN_API_KEY", ""))
LANGSMITH_PROJECT = os.getenv("LANGSMITH_PROJECT", os.getenv("LANGCHAIN_PROJECT", "lab24-eval-guardrails"))
LANGSMITH_ENDPOINT = os.getenv("LANGSMITH_ENDPOINT", os.getenv("LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com"))
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "auto").strip().lower()
EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "auto").strip().lower()
LMSTUDIO_BASE_URL = os.getenv("LMSTUDIO_BASE_URL", "http://localhost:1234/v1")
LMSTUDIO_MODEL = os.getenv("LMSTUDIO_MODEL", "local-model")
LMSTUDIO_EMBEDDING_MODEL = os.getenv("LMSTUDIO_EMBEDDING_MODEL", "local-embedding-model")
LMSTUDIO_GUARD_MODEL = os.getenv("LMSTUDIO_GUARD_MODEL", "local-llama-guard-model")
LMSTUDIO_CHAT_TIMEOUT = float(os.getenv("LMSTUDIO_CHAT_TIMEOUT", "120"))
LMSTUDIO_EMBEDDING_TIMEOUT = float(os.getenv("LMSTUDIO_EMBEDDING_TIMEOUT", "60"))
OUTPUT_GUARD_PROVIDER = os.getenv("OUTPUT_GUARD_PROVIDER", "auto").strip().lower()
LLAMA_GUARD_MODEL = os.getenv("LLAMA_GUARD_MODEL", "meta-llama/Llama-Guard-3-8B")

# --- Qdrant ---
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
COLLECTION_NAME = "lab18_production"
NAIVE_COLLECTION = "lab18_naive"

# --- Embedding ---
EMBEDDING_MODEL = "BAAI/bge-m3"
EMBEDDING_DIM = 1024

# --- Chunking ---
HIERARCHICAL_PARENT_SIZE = 2048
HIERARCHICAL_CHILD_SIZE = 256
SEMANTIC_THRESHOLD = 0.85

# --- Search ---
BM25_TOP_K = 20
DENSE_TOP_K = 20
HYBRID_TOP_K = 20
RERANK_TOP_K = 3

# --- Paths ---
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
TEST_SET_PATH = os.path.join(os.path.dirname(__file__), "test_set.json")
