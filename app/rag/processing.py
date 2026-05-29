import re
from typing import Dict, List

from app.config.settings import get_settings
from app.utils.jsonl import read_jsonl, write_jsonl


def clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text or "").strip()
    text = re.sub(r"(Aceptar cookies|Política de cookies)", "", text, flags=re.IGNORECASE)
    return text


def chunk_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    if not text:
        return []
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == len(text):
            break
        start = max(0, end - overlap)
    return chunks


def process_documents() -> int:
    settings = get_settings()
    raw_docs = read_jsonl(settings.raw_data_path)
    processed: List[Dict] = []

    for doc in raw_docs:
        if doc.get("error") or not doc.get("content"):
            continue
        text = clean_text(doc["content"])
        for idx, chunk in enumerate(chunk_text(text, settings.chunk_size, settings.chunk_overlap)):
            processed.append({
                "id": f"{doc['url']}#chunk-{idx}",
                "url": doc["url"],
                "title": doc.get("title", ""),
                "chunk_index": idx,
                "text": chunk,
            })

    write_jsonl(settings.processed_data_path, processed)
    return len(processed)
