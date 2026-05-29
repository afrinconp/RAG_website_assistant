import os
from functools import lru_cache
from typing import Dict, List, Tuple

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

from app.config.settings import get_settings

NO_HISTORY_MESSAGE = "No hay historial previo."
NO_CONTEXT_MESSAGE = (
    "No encontré información suficiente en las páginas "
    "indexadas para responder con confianza."
)

SOURCE_PREVIEW_LENGTH = 700

SourceList = List[Dict]


@lru_cache(maxsize=1)
def get_gemini_llm() -> ChatGoogleGenerativeAI:
    """
    Create and cache the Gemini client.

    The API key can come from:
    1. Environment variables.
    2. The Streamlit sidebar.
    3. The .env file.
    """
    settings = get_settings()

    api_key = (
        os.getenv("GOOGLE_API_KEY")
        or settings.google_api_key
    )

    if not api_key:
        raise ValueError(
            "GOOGLE_API_KEY is missing. "
            "Configure it in Streamlit or in the .env file."
        )

    return ChatGoogleGenerativeAI(
        model=settings.gemini_model,
        google_api_key=api_key,
        temperature=settings.gemini_temperature,
        max_output_tokens=settings.max_new_tokens,
    )


def format_retrieved_context(
    retrieved_docs: List[Dict],
) -> str:
    """
    Convert retrieved chunks into a context block.

    Args:
        retrieved_docs: Retrieved RAG documents.

    Returns:
        Formatted context string.
    """
    blocks = []

    for index, document in enumerate(
        retrieved_docs,
        start=1,
    ):
        metadata = document.get("metadata", {})

        blocks.append(
            (
                f"[Fuente {index}]\n"
                f"Título: {metadata.get('title', 'Untitled')}\n"
                f"URL: {metadata.get('url', '')}\n"
                "Contenido:\n"
                f"{document.get('text', '')}"
            )
        )

    return "\n\n".join(blocks)


def format_history(
    history: List[Dict],
) -> str:
    """
    Convert conversation history into text.

    Args:
        history: Previous conversation messages.

    Returns:
        Formatted history string.
    """
    if not history:
        return NO_HISTORY_MESSAGE

    return "\n".join(
        (
            f"{message['role']}: "
            f"{message['content']}"
        )
        for message in history
    )


def format_sources(
    retrieved_docs: List[Dict],
) -> SourceList:
    """
    Prepare source metadata for persistence and UI display.

    Args:
        retrieved_docs: Retrieved documents.

    Returns:
        List of source metadata dictionaries.
    """
    sources = []

    for index, document in enumerate(
        retrieved_docs,
        start=1,
    ):
        metadata = document.get("metadata", {})

        sources.append(
            {
                "rank": index,
                "title": metadata.get(
                    "title",
                    "Fuente",
                ),
                "url": metadata.get("url", ""),
                "chunk_index": metadata.get(
                    "chunk_index",
                    "",
                ),
                "distance": document.get("distance"),
                "similarity_score": document.get(
                    "similarity_score"
                ),
                "rerank_score": document.get(
                    "rerank_score"
                ),
                "rerank_error": document.get(
                    "rerank_error"
                ),
                "text_preview": document.get(
                    "text",
                    "",
                )[:SOURCE_PREVIEW_LENGTH],
            }
        )

    return sources


def build_rag_chain():
    """
    Build the LangChain RAG pipeline.

    Returns:
        Runnable LangChain chain.
    """
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
Eres un asistente RAG especializado
en información bancaria.

Reglas:
- Responde en español.
- Usa el historial para comprender contexto.
- Usa únicamente el contexto recuperado
  como fuente factual principal.
- Si no existe información suficiente,
  responde exactamente:
  "No encontré información suficiente
   en las fuentes indexadas."
- No inventes datos.
- Sé claro y profesional.
                """,
            ),
            (
                "human",
                """
Historial:
{history}

Contexto:
{retrieved_context}

Pregunta:
{question}

Respuesta:
                """,
            ),
        ]
    )

    return (
        prompt
        | get_gemini_llm()
        | StrOutputParser()
    )


def generate_answer(
    question: str,
    retrieved_docs: List[Dict],
    history: List[Dict],
) -> Tuple[str, SourceList]:
    """
    Generate a grounded answer using Gemini.

    Args:
        question: User question.
        retrieved_docs: Retrieved chunks.
        history: Conversation history.

    Returns:
        Tuple containing:
        - Generated answer
        - Source metadata
    """
    sources = format_sources(retrieved_docs)

    if not retrieved_docs:
        return NO_CONTEXT_MESSAGE, sources

    try:
        chain = build_rag_chain()

        answer = chain.invoke(
            {
                "question": question,
                "retrieved_context": (
                    format_retrieved_context(
                        retrieved_docs
                    )
                ),
                "history": format_history(
                    history
                ),
            }
        )

        return answer.strip(), sources

    except Exception as exc:
        context_preview = "\n\n".join(
            (
                f"- {document.get('text', '')[:SOURCE_PREVIEW_LENGTH]}"
            )
            for document in retrieved_docs[:3]
        )

        fallback_message = (
            "No pude generar la respuesta "
            "con Gemini.\n\n"
            "Fragmentos recuperados:\n\n"
            f"{context_preview}\n\n"
            f"Detalle técnico: "
            f"{type(exc).__name__}: {exc}"
        )

        return fallback_message, sources