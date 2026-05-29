from typing import Dict, List

import chromadb
from sentence_transformers import SentenceTransformer

from app.config.settings import get_settings
from app.utils.jsonl import read_jsonl


class ChromaVectorStore:
    """Singleton Pattern: keeps one Chroma client and embedding model instance.

    This class is responsible for:
    1. converting chunks into embeddings,
    2. storing them in Chroma,
    3. embedding the user prompt,
    4. retrieving the most semantically similar chunks.
    """

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self.settings = get_settings()
        self.client = chromadb.PersistentClient(path=self.settings.chroma_path)
        self.collection = self.client.get_or_create_collection(self.settings.collection_name)
        self.model = SentenceTransformer(self.settings.embedding_model)
        self._initialized = True

    def build_index(self) -> int:
        """Embed processed chunks and upsert them into Chroma."""
        chunks = read_jsonl(self.settings.processed_data_path)
        if not chunks:
            return 0

        ids = [row["id"] for row in chunks]
        texts = [row["text"] for row in chunks]
        embeddings = self.model.encode(texts, normalize_embeddings=True).tolist()

        metadatas: List[Dict] = [
            {
                "url": row["url"],
                "title": row["title"],
                "chunk_index": row["chunk_index"],
            }
            for row in chunks
        ]

        self.collection.upsert(
            ids=ids,
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas,
        )
        return len(chunks)

    def search(self, query: str, k: int = 5) -> List[Dict]:
        """Return the top-k chunks most similar to the user prompt.

        Chroma returns a distance. Because embeddings are normalized, lower distance
        means greater semantic similarity. To make this easier to understand in the UI,
        we also expose a `similarity_score` where higher is better:

            similarity_score = 1 / (1 + distance)
        """
        query_embedding = self.model.encode(
            [query],
            normalize_embeddings=True,
        ).tolist()[0]

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=k,
            include=["documents", "metadatas", "distances"],
        )

        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]
        ids = results.get("ids", [[]])[0]

        docs: List[Dict] = []
        for i, text in enumerate(documents):
            distance = float(distances[i]) if distances else None
            similarity_score = None if distance is None else 1.0 / (1.0 + distance)

            docs.append(
                {
                    "id": ids[i] if ids else "",
                    "text": text,
                    "metadata": metadatas[i] if metadatas else {},
                    "distance": distance,
                    "similarity_score": similarity_score,
                }
            )

        return sorted(
            docs,
            key=lambda item: item.get("similarity_score") or 0.0,
            reverse=True,
        )
