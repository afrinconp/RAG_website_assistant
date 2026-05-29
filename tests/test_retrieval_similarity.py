from app.rag.vector_store import ChromaVectorStore


def test_search_returns_ranked_chunks():
    """Smoke test: after the index is built, search should return ranked chunks."""
    store = ChromaVectorStore()
    docs = store.search("tarjeta de crédito", k=3)
    assert isinstance(docs, list)
    for doc in docs:
        assert "text" in doc
        assert "similarity_score" in doc
