from typing import Dict, List

from app.config.settings import get_settings
from app.memory.database import SQLiteConversationDB
from app.rag.generator import generate_answer
from app.rag.retrieval import SemanticRetrievalWithOptionalReranker


class ConversationService:
    """Conversation service with persistent memory.

    This service stores each user/assistant message in SQLite by session_id.
    It also uses the last N messages to build a contextual retrieval query,
    so follow-up questions such as "¿y los requisitos?" can still use the
    previous conversation context.
    """

    def __init__(self):
        self.settings = get_settings()
        self.db = SQLiteConversationDB()
        self.retrieval = SemanticRetrievalWithOptionalReranker()

    def _build_contextual_retrieval_query(self, question: str, history: List[Dict]) -> str:
        """Combine recent chat history with the current question for retrieval.

        The LLM receives the full history separately, but the vector search also
        benefits from previous turns. This improves retrieval for follow-up
        questions that depend on context.
        """
        if not history:
            return question

        recent_context = "\n".join(
            f"{message['role']}: {message['content']}"
            for message in history[-self.settings.n_history_messages :]
        )

        return f"""
Conversación reciente:
{recent_context}

Pregunta actual:
{question}
""".strip()

    def ask(self, session_id: str, question: str) -> Dict:
        # Load previous messages before adding the current user question.
        history = self.db.get_recent_messages(session_id, self.settings.n_history_messages)

        # Persist current user message.
        self.db.add_message(session_id, "user", question)

        # Retrieve chunks using the current question plus previous context.
        contextual_query = self._build_contextual_retrieval_query(question, history)
        docs = self.retrieval.retrieve(contextual_query)

        # Generate final grounded answer using Gemini and conversation memory.
        answer, sources = generate_answer(question, docs, history)

        # Persist assistant answer and retrieved sources.
        self.db.add_message(session_id, "assistant", answer, sources=sources)

        return {
            "answer": answer,
            "sources": sources,
            "used_history_messages": len(history),
            "contextual_retrieval_query": contextual_query,
        }

    def get_session_messages(self, session_id: str) -> List[Dict]:
        if hasattr(self.db, "get_session_messages"):
            return self.db.get_session_messages(session_id)
        return self.db.get_recent_messages(session_id, self.settings.n_history_messages)

    def clear_session(self, session_id: str):
        if hasattr(self.db, "clear_session"):
            self.db.clear_session(session_id)
