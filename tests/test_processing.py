from app.rag.processing import clean_text, chunk_text


def test_clean_text_collapses_spaces():
    assert clean_text("hola     mundo") == "hola mundo"


def test_chunk_text_returns_chunks():
    chunks = chunk_text("a" * 100, chunk_size=30, overlap=5)
    assert len(chunks) > 1
    assert all(len(c) <= 30 for c in chunks)
