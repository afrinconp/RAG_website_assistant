from app.rag.retrieval import (
    SemanticRetrievalWithOptionalReranker,
)


def test_retriever_exposes_retrieve_method() -> None:
    """
    Verify that the retriever exposes a callable
    retrieval method.
    """
    # Arrange
    retriever = (
        SemanticRetrievalWithOptionalReranker()
    )

    # Assert
    assert callable(
        getattr(
            retriever,
            "retrieve",
            None,
        )
    )