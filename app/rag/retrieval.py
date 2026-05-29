from abc import ABC, abstractmethod
from typing import Dict, List, Optional

from sentence_transformers import CrossEncoder

from app.config.settings import get_settings
from app.rag.vector_store import ChromaVectorStore

Document = Dict


class RetrievalStrategy(ABC):
    """
    Abstract retrieval strategy.

    Different retrieval implementations can inherit from
    this interface and provide their own retrieval logic.
    """

    @abstractmethod
    def retrieve(
        self,
        query: str,
    ) -> List[Document]:
        """
        Retrieve the chunks that best match the query.

        Args:
            query: User question or search query.

        Returns:
            List of retrieved documents.
        """
        raise NotImplementedError


class SemanticRetrievalWithOptionalReranker(
    RetrievalStrategy
):
    """
    Semantic retrieval with optional CrossEncoder reranking.

    Retrieval flow:

    1. Embed the user query.
    2. Search Chroma for nearest vectors.
    3. Optionally rerank results with a CrossEncoder.
    4. Return the best documents.

    If the reranker fails, the system falls back to
    Chroma similarity ranking instead of crashing.
    """

    def __init__(self) -> None:
        """Initialize retriever and optional reranker."""
        self.settings = get_settings()

        self.store = ChromaVectorStore()

        self.reranker: Optional[
            CrossEncoder
        ] = None

        self.reranker_error: Optional[
            str
        ] = None

        self._initialize_reranker()

    def _initialize_reranker(self) -> None:
        """
        Initialize the CrossEncoder reranker.

        The reranker runs on CPU to avoid common
        Docker, CUDA, and meta-tensor issues.
        """
        if not self.settings.use_reranker:
            return

        try:
            self.reranker = CrossEncoder(
                self.settings.reranker_model,
                device="cpu",
            )

        except Exception as exc:
            self.reranker = None
            self.reranker_error = (
                f"{type(exc).__name__}: {exc}"
            )

    def retrieve(
        self,
        query: str,
    ) -> List[Document]:
        """
        Retrieve documents using vector similarity.

        Args:
            query: User query.

        Returns:
            Top ranked documents.
        """
        documents = self.store.search(
            query=query,
            k=self.settings.top_k,
        )

        if not documents:
            return []

        if self.reranker:
            documents = self._rerank_documents(
                query=query,
                documents=documents,
            )

        return documents[
            : self.settings.final_k
        ]

    def _rerank_documents(
        self,
        query: str,
        documents: List[Document],
    ) -> List[Document]:
        """
        Rerank retrieved documents using a CrossEncoder.

        Args:
            query: User query.
            documents: Retrieved documents.

        Returns:
            Reranked documents.

        Notes:
            If reranking fails, the original
            vector similarity ranking is returned.
        """
        try:
            query_document_pairs = [
                (query, document["text"])
                for document in documents
            ]

            scores = self.reranker.predict(
                query_document_pairs
            )

            for document, score in zip(
                documents,
                scores,
            ):
                document["rerank_score"] = float(
                    score
                )

            return sorted(
                documents,
                key=lambda document: document.get(
                    "rerank_score",
                    0.0,
                ),
                reverse=True,
            )

        except Exception as exc:
            self.reranker = None

            self.reranker_error = (
                f"{type(exc).__name__}: {exc}"
            )

            for document in documents:
                document[
                    "rerank_error"
                ] = self.reranker_error

            return documents