from typing import Dict, List

from app.config.settings import get_settings
from app.memory.database import SQLiteConversationDB
from app.rag.generator import generate_answer
from app.rag.retrieval import (
    SemanticRetrievalWithOptionalReranker,
)

USER_ROLE = "user"
ASSISTANT_ROLE = "assistant"


class ConversationService:
    """
    Manage RAG conversations with persistent memory.

    This service:

    - Stores messages in SQLite.
    - Retrieves previous conversation history.
    - Builds contextual retrieval queries.
    - Retrieves relevant chunks from the vector database.
    - Generates answers using Gemini.
    - Persists assistant responses and sources.
    """

    def __init__(self) -> None:
        """Initialize dependencies."""
        self.settings = get_settings()
        self.db = SQLiteConversationDB()
        self.retrieval = (
            SemanticRetrievalWithOptionalReranker()
        )

    def _build_contextual_retrieval_query(
        self,
        question: str,
        history: List[Dict],
    ) -> str:
        """
        Build a retrieval query using recent conversation history.

        This helps the retriever understand follow-up questions
        that depend on previous user interactions.

        Args:
            question: Current user question.
            history: Previous conversation messages.

        Returns:
            Context-enhanced retrieval query.
        """
        if not history:
            return question

        recent_context = "\n".join(
            (
                f"{message['role']}: "
                f"{message['content']}"
            )
            for message in history[
                -self.settings.n_history_messages :
            ]
        )

        return (
            "Conversación reciente:\n"
            f"{recent_context}\n\n"
            "Pregunta actual:\n"
            f"{question}"
        )

    def ask(
        self,
        session_id: str,
        question: str,
    ) -> Dict:
        """
        Process a user question through the RAG pipeline.

        Args:
            session_id: Conversation identifier.
            question: User question.

        Returns:
            Dictionary containing:
            - answer
            - sources
            - used_history_messages
            - contextual_retrieval_query
        """
        history = self.db.get_recent_messages(
            session_id,
            self.settings.n_history_messages,
        )

        self.db.add_message(
            session_id=session_id,
            role=USER_ROLE,
            content=question,
        )

        contextual_query = (
            self._build_contextual_retrieval_query(
                question=question,
                history=history,
            )
        )

        documents = self.retrieval.retrieve(
            contextual_query
        )

        answer, sources = generate_answer(
            question=question,
            retrieved_docs=documents,
            history=history,
        )

        self.db.add_message(
            session_id=session_id,
            role=ASSISTANT_ROLE,
            content=answer,
            sources=sources,
        )

        return {
            "answer": answer,
            "sources": sources,
            "used_history_messages": len(history),
            "contextual_retrieval_query": contextual_query,
        }

    def get_session_messages(
        self,
        session_id: str,
    ) -> List[Dict]:
        """
        Return all messages for a session.

        Falls back to recent messages if the database
        implementation does not expose a full-history method.
        """
        if hasattr(self.db, "get_session_messages"):
            return self.db.get_session_messages(
                session_id
            )

        return self.db.get_recent_messages(
            session_id,
            self.settings.n_history_messages,
        )

    def clear_session(
        self,
        session_id: str,
    ) -> None:
        """
        Remove all stored messages for a session.

        Args:
            session_id: Conversation identifier.
        """
        if hasattr(self.db, "clear_session"):
            self.db.clear_session(session_id)