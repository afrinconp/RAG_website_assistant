from app.rag.retrieval import SemanticRetrievalWithOptionalReranker


def test_retriever_can_initialize_without_crashing():
    retriever = SemanticRetrievalWithOptionalReranker()
    assert hasattr(retriever, "retrieve")
