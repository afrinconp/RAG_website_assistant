import os
from functools import lru_cache
from typing import Dict, List, Tuple

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

from app.config.settings import get_settings


def get_gemini_llm() -> ChatGoogleGenerativeAI:
    """Create the Gemini chat model through LangChain.

    The API key can come from either:
    1. the .env file, or
    2. the Streamlit sidebar, which writes GOOGLE_API_KEY into os.environ.
    """
    settings = get_settings()
    api_key = os.getenv("GOOGLE_API_KEY") or settings.google_api_key

    if not api_key:
        raise ValueError(
            "GOOGLE_API_KEY is missing. Add it in the Streamlit sidebar or in your .env file."
        )

    return ChatGoogleGenerativeAI(
        model=settings.gemini_model,
        google_api_key=api_key,
        temperature=settings.gemini_temperature,
        max_output_tokens=settings.max_new_tokens,
    )


def format_retrieved_context(retrieved_docs: List[Dict]) -> str:
    """Convert retrieved chunks into the context sent to Gemini."""
    blocks = []
    for i, doc in enumerate(retrieved_docs, start=1):
        metadata = doc.get("metadata", {})
        title = metadata.get("title", "Untitled")
        url = metadata.get("url", "")
        text = doc.get("text", "")

        blocks.append(
            f"""[Fuente {i}]
Título: {title}
URL: {url}
Contenido:
{text}
"""
        )

    return "\n\n".join(blocks)


def format_history(history: List[Dict]) -> str:
    """Convert previous messages into a compact conversation memory string."""
    if not history:
        return "No hay historial previo."

    return "\n".join(
        f"{message['role']}: {message['content']}"
        for message in history
    )


def format_sources(retrieved_docs: List[Dict]) -> List[Dict]:
    """Prepare source metadata for the Streamlit UI and SQLite persistence."""
    sources = []

    for i, doc in enumerate(retrieved_docs, start=1):
        metadata = doc.get("metadata", {})
        sources.append(
            {
                "rank": i,
                "title": metadata.get("title", "Fuente"),
                "url": metadata.get("url", ""),
                "chunk_index": metadata.get("chunk_index", ""),
                "distance": doc.get("distance"),
                "similarity_score": doc.get("similarity_score"),
                "rerank_score": doc.get("rerank_score"),
                "rerank_error": doc.get("rerank_error"),
                "text_preview": doc.get("text", "")[:700],
            }
        )

    return sources


def build_rag_chain():
    """Build a LangChain RAG chain.

    The retriever is implemented separately in app/rag/retrieval.py.
    This chain receives:
    - question
    - retrieved_context
    - history

    and asks Gemini to answer grounded only in the retrieved context.
    """
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """Eres un asistente RAG especializado en información bancaria.

Reglas:
- Responde en español.
- Usa el historial de conversación para entender referencias, seguimiento y contexto.
- Usa únicamente el contexto recuperado como fuente factual principal.
- Si el contexto recuperado no contiene la respuesta, responde exactamente:
  "No encontré información suficiente en las fuentes indexadas."
- No inventes datos.
- Sé claro, directo y profesional.
- Cuando sea útil, menciona que la respuesta se basa en las fuentes recuperadas.""",
            ),
            (
                "human",
                """Historial de conversación:
{history}

Contexto recuperado:
{retrieved_context}

Pregunta del usuario:
{question}

Respuesta:""",
            ),
        ]
    )

    return prompt | get_gemini_llm() | StrOutputParser()


def generate_answer(
    question: str,
    retrieved_docs: List[Dict],
    history: List[Dict],
) -> Tuple[str, List[Dict]]:
    """Generate the final RAG answer with Gemini through LangChain.

    RAG flow:
        question
        -> app/rag/retrieval.py retrieves the most similar chunks
        -> retrieved chunks are formatted as context
        -> LangChain sends context + question + history to Gemini
        -> Gemini returns a grounded answer
    """
    sources = format_sources(retrieved_docs)

    if not retrieved_docs:
        return (
            "No encontré información suficiente en las páginas indexadas para responder con confianza.",
            sources,
        )

    try:
        chain = build_rag_chain()
        answer = chain.invoke(
            {
                "question": question,
                "retrieved_context": format_retrieved_context(retrieved_docs),
                "history": format_history(history),
            }
        )
        return answer.strip(), sources

    except Exception as exc:
        context_preview = "\n\n".join(
            f"- {doc.get('text', '')[:700]}" for doc in retrieved_docs[:3]
        )
        fallback = (
            "No pude generar la respuesta con Gemini. "
            "Aun así, estos son los fragmentos más relevantes recuperados:\n\n"
            f"{context_preview}\n\n"
            f"Detalle técnico: {type(exc).__name__}: {exc}"
        )
        return fallback, sources
