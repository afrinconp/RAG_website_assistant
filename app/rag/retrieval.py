from abc import ABC, abstractmethod
from typing import Dict, List

from sentence_transformers import CrossEncoder

from app.config.settings import get_settings
from app.rag.vector_store import ChromaVectorStore


class RetrievalStrategy(ABC):
    """Strategy Pattern: different retrieval algorithms can implement this interface."""

    @abstractmethod
    def retrieve(self, query: str) -> List[Dict]:
        """Retrieve the chunks that best match the user prompt."""
        raise NotImplementedError


class SemanticRetrievalWithOptionalReranker(RetrievalStrategy):
    """Semantic retrieval over Chroma plus optional safe CrossEncoder reranking.

    The core retrieval is always Chroma vector similarity.

    The reranker is only a bonus. If the CrossEncoder fails because of Torch,
    Transformers, device, or meta-tensor issues, the app falls back to the
    vector-similarity ranking instead of crashing.
    """

    def __init__(self):
        self.settings = get_settings()
        self.store = ChromaVectorStore()
        self.reranker = None
        self.reranker_error = None

        if self.settings.use_reranker:
            try:
                # Force CPU to avoid many Docker/GPU/meta-tensor issues.
                self.reranker = CrossEncoder(
                    self.settings.reranker_model,
                    device="cpu",
                )
            except Exception as exc:
                self.reranker = None
                self.reranker_error = f"{type(exc).__name__}: {exc}"

    def retrieve(self, query: str) -> List[Dict]:
        # Main retrieval: prompt embedding -> Chroma nearest vectors.
        docs = self.store.search(query=query, k=self.settings.top_k)

        if not docs:
            return []

        if self.reranker:
            try:
                pairs = [(query, doc["text"]) for doc in docs]
                scores = self.reranker.predict(pairs)

                for doc, score in zip(docs, scores):
                    doc["rerank_score"] = float(score)

                docs = sorted(
                    docs,
                    key=lambda item: item.get("rerank_score", 0.0),
                    reverse=True,
                )

            except Exception as exc:
                # Do not fail the RAG system if the bonus reranker fails.
                # Keep the Chroma similarity order.
                self.reranker = None
                self.reranker_error = f"{type(exc).__name__}: {exc}"
                for doc in docs:
                    doc["rerank_error"] = self.reranker_error

        return docs[: self.settings.final_k]
