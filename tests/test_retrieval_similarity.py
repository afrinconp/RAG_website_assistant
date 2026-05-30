from app.rag.vector_store import ChromaVectorStore


def test_search_returns_a_list() -> None:
    """Verify that search returns a list."""
    vector_store = ChromaVectorStore()

    results = vector_store.search(
        query="tarjeta de crédito",
        k=3,
    )

    assert isinstance(results, list)


def test_search_results_contain_text_field() -> None:
    """Verify that retrieved documents contain text."""
    vector_store = ChromaVectorStore()

    results = vector_store.search(
        query="tarjeta de crédito",
        k=3,
    )

    for document in results:
        assert "text" in document


def test_search_results_contain_similarity_score() -> None:
    """Verify that retrieved documents contain similarity scores."""
    vector_store = ChromaVectorStore()

    results = vector_store.search(
        query="tarjeta de crédito",
        k=3,
    )

    for document in results:
        assert "similarity_score" in document