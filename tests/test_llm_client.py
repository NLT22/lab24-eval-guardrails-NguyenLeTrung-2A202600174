from src.llm_client import embed_query, embed_texts


def test_embed_texts_returns_vectors(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("EMBEDDING_PROVIDER", "sentence_transformers")
    monkeypatch.setenv("PYTEST_CURRENT_TEST", "1")

    vectors = embed_texts(["hello", "xin chao"], provider="sentence_transformers")

    assert len(vectors) == 2
    assert all(isinstance(v, list) and v for v in vectors)
    assert all(isinstance(x, float) for x in vectors[0])


def test_embed_query_returns_single_vector(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("EMBEDDING_PROVIDER", "sentence_transformers")
    monkeypatch.setenv("PYTEST_CURRENT_TEST", "1")

    vector = embed_query("hello", provider="sentence_transformers")

    assert isinstance(vector, list)
    assert len(vector) > 0
