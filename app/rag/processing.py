import re
from typing import Dict, List

from app.config.settings import get_settings
from app.utils.jsonl import read_jsonl, write_jsonl

COOKIE_PATTERNS = (
    r"Aceptar cookies",
    r"Política de cookies",
)

Document = Dict


def clean_text(text: str) -> str:
    """
    Clean and normalize raw document text.

    Operations:
    - Remove excessive whitespace.
    - Remove cookie-related banners.

    Args:
        text: Raw text extracted from a webpage.

    Returns:
        Cleaned text.
    """
    cleaned_text = re.sub(
        r"\s+",
        " ",
        text or "",
    ).strip()

    cookie_pattern = "|".join(COOKIE_PATTERNS)

    cleaned_text = re.sub(
        cookie_pattern,
        "",
        cleaned_text,
        flags=re.IGNORECASE,
    )

    return cleaned_text


def chunk_text(
    text: str,
    chunk_size: int,
    overlap: int,
) -> List[str]:
    """
    Split text into overlapping chunks.

    Args:
        text: Input text.
        chunk_size: Maximum chunk size.
        overlap: Number of overlapping characters.

    Returns:
        List of text chunks.
    """
    if not text:
        return []

    chunks: List[str] = []
    start_index = 0

    while start_index < len(text):
        end_index = min(
            start_index + chunk_size,
            len(text),
        )

        chunk = text[start_index:end_index].strip()

        if chunk:
            chunks.append(chunk)

        if end_index == len(text):
            break

        start_index = max(
            0,
            end_index - overlap,
        )

    return chunks


def process_documents() -> int:
    """
    Process scraped documents and generate chunks.

    Workflow:
    1. Read raw documents.
    2. Clean document content.
    3. Split content into chunks.
    4. Save processed chunks.

    Returns:
        Number of generated chunks.
    """
    settings = get_settings()

    raw_documents = read_jsonl(
        settings.raw_data_path
    )

    processed_documents: List[Document] = []

    for document in raw_documents:

        if (
            document.get("error")
            or not document.get("content")
        ):
            continue

        cleaned_text = clean_text(
            document["content"]
        )

        chunks = chunk_text(
            text=cleaned_text,
            chunk_size=settings.chunk_size,
            overlap=settings.chunk_overlap,
        )

        for chunk_index, chunk in enumerate(chunks):

            processed_documents.append(
                {
                    "id": (
                        f"{document['url']}"
                        f"#chunk-{chunk_index}"
                    ),
                    "url": document["url"],
                    "title": document.get(
                        "title",
                        "",
                    ),
                    "chunk_index": chunk_index,
                    "text": chunk,
                }
            )

    write_jsonl(
        settings.processed_data_path,
        processed_documents,
    )

    return len(processed_documents)