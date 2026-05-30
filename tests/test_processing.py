from app.rag.processing import (
    chunk_text,
    clean_text,
)


def test_clean_text_collapses_spaces() -> None:
    """
    Verify that multiple spaces are replaced
    by a single space.
    """
    # Arrange
    raw_text = "hola     mundo"

    # Act
    cleaned_text = clean_text(raw_text)

    # Assert
    assert cleaned_text == "hola mundo"


def test_chunk_text_returns_chunks() -> None:
    """
    Verify that text is split into multiple chunks
    and that no chunk exceeds the configured size.
    """
    # Arrange
    text = "a" * 100
    chunk_size = 30
    overlap = 5

    # Act
    chunks = chunk_text(
        text=text,
        chunk_size=chunk_size,
        overlap=overlap,
    )

    # Assert
    assert len(chunks) > 1

    assert all(
        len(chunk) <= chunk_size
        for chunk in chunks
    )