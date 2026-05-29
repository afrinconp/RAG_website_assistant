from typing import Dict, List, Optional

import chromadb
from sentence_transformers import SentenceTransformer

from app.config.settings import get_settings
from app.utils.jsonl import read_jsonl

Document = Dict

DEFAULT_SIMILARITY_SCORE = 0.0


class ChromaVectorStore:
    """
    Singleton vector store backed by ChromaDB.

    Responsibilities:

    1. Generate embeddings from document chunks.
    2. Persist embeddings in ChromaDB.
    3. Embed user queries.
    4. Retrieve semantically similar chunks.
    """

    _instance = None

    def __new__(cls) -> "ChromaVectorStore":
        """
        Create a singleton instance.

        Returns:
            ChromaVectorStore instance.
        """
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False

        return cls._instance

    def __init__(self) -> None:
        """Initialize ChromaDB client and embedding model."""
        if self._initialized:
            return

        self.settings = get_settings()

        self.client = chromadb.PersistentClient(
            path=self.settings.chroma_path
        )

        self.collection = (
            self.client.get_or_create_collection(
                self.settings.collection_name
            )
        )

        self.model = SentenceTransformer(
            self.settings.embedding_model
        )

        self._initialized = True

    def build_index(self) -> int:
        """
        Create embeddings and store them in ChromaDB.

        Returns:
            Number of indexed chunks.
        """
        chunks = read_jsonl(
            self.settings.processed_data_path
        )

        if not chunks:
            return 0

        ids = [
            chunk["id"]
            for chunk in chunks
        ]

        texts = [
            chunk["text"]
            for chunk in chunks
        ]

        embeddings = self.model.encode(
            texts,
            normalize_embeddings=True,
        ).tolist()

        metadatas = self._build_metadata(
            chunks
        )

        self.collection.upsert(
            ids=ids,
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas,
        )

        return len(chunks)

    def _build_metadata(
        self,
        chunks: List[Document],
    ) -> List[Document]:
        """
        Create metadata records for ChromaDB.

        Args:
            chunks: Processed chunks.

        Returns:
            Metadata list.
        """
        return [
            {
                "url": chunk["url"],
                "title": chunk["title"],
                "chunk_index": chunk["chunk_index"],
            }
            for chunk in chunks
        ]

    def _calculate_similarity_score(
        self,
        distance: Optional[float],
    ) -> Optional[float]:
        """
        Convert distance into a similarity score.

        Formula:

            similarity = 1 / (1 + distance)

        Args:
            distance: Chroma distance.

        Returns:
            Similarity score.
        """
        if distance is None:
            return None

        return 1.0 / (1.0 + distance)

    def search(
        self,
        query: str,
        k: int = 5,
    ) -> List[Document]:
        """
        Retrieve the most similar chunks.

        Args:
            query: User prompt.
            k: Number of results.

        Returns:
            Ranked documents ordered by similarity.
        """
        query_embedding = self.model.encode(
            [query],
            normalize_embeddings=True,
        ).tolist()[0]

        results = self.collection.query(
            query_embeddings=[
                query_embedding
            ],
            n_results=k,
            include=[
                "documents",
                "metadatas",
                "distances",
            ],
        )

        documents = results.get(
            "documents",
            [[]],
        )[0]

        metadatas = results.get(
            "metadatas",
            [[]],
        )[0]

        distances = results.get(
            "distances",
            [[]],
        )[0]

        ids = results.get(
            "ids",
            [[]],
        )[0]

        return self._format_results(
            documents=documents,
            metadatas=metadatas,
            distances=distances,
            ids=ids,
        )

    def _format_results(
        self,
        documents: List[str],
        metadatas: List[Dict],
        distances: List[float],
        ids: List[str],
    ) -> List[Document]:
        """
        Convert Chroma results into application format.

        Args:
            documents: Retrieved chunk texts.
            metadatas: Associated metadata.
            distances: Vector distances.
            ids: Chunk identifiers.

        Returns:
            Sorted document list.
        """
        results: List[Document] = []

        for index, text in enumerate(
            documents
        ):
            distance = (
                float(distances[index])
                if distances
                else None
            )

            similarity_score = (
                self._calculate_similarity_score(
                    distance
                )
            )

            results.append(
                {
                    "id": (
                        ids[index]
                        if ids
                        else ""
                    ),
                    "text": text,
                    "metadata": (
                        metadatas[index]
                        if metadatas
                        else {}
                    ),
                    "distance": distance,
                    "similarity_score": (
                        similarity_score
                    ),
                }
            )

        return sorted(
            results,
            key=lambda document: (
                document.get(
                    "similarity_score"
                )
                or DEFAULT_SIMILARITY_SCORE
            ),
            reverse=True,
        )