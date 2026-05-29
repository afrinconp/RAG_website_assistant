from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # Scraping configuration
    bank_name: str = Field(
        default="bbva",
        alias="BANK_NAME",
    )

    start_url: str = Field(
        default="https://www.bbva.com.co/",
        alias="START_URL",
    )

    max_pages: int = Field(
        default=20,
        alias="MAX_PAGES",
    )

    request_timeout: int = Field(
        default=15,
        alias="REQUEST_TIMEOUT",
    )

    # Embeddings and vector database
    embedding_model: str = Field(
        default="intfloat/multilingual-e5-small",
        alias="EMBEDDING_MODEL",
    )

    collection_name: str = Field(
        default="bank_knowledge",
        alias="COLLECTION_NAME",
    )

    chunk_size: int = Field(
        default=900,
        alias="CHUNK_SIZE",
    )

    chunk_overlap: int = Field(
        default=150,
        alias="CHUNK_OVERLAP",
    )

    # Retrieval configuration
    top_k: int = Field(
        default=8,
        alias="TOP_K",
    )

    final_k: int = Field(
        default=4,
        alias="FINAL_K",
    )

    use_reranker: bool = Field(
        default=False,
        alias="USE_RERANKER",
    )

    reranker_model: str = Field(
        default="cross-encoder/ms-marco-MiniLM-L-6-v2",
        alias="RERANKER_MODEL",
    )

    # Gemini / LangChain configuration
    google_api_key: str = Field(
        default="",
        alias="GOOGLE_API_KEY",
    )

    gemini_model: str = Field(
        default="gemini-3.1-flash-lite",
        alias="GEMINI_MODEL",
    )

    gemini_temperature: float = Field(
        default=0.2,
        alias="GEMINI_TEMPERATURE",
    )

    max_new_tokens: int = Field(
        default=500,
        alias="MAX_NEW_TOKENS",
    )

    # Conversation memory
    n_history_messages: int = Field(
        default=6,
        alias="N_HISTORY_MESSAGES",
    )

    sqlite_path: str = Field(
        default="data/conversations.db",
        alias="SQLITE_PATH",
    )

    # Data paths
    raw_data_path: str = Field(
        default="data/raw/pages.jsonl",
        alias="RAW_DATA_PATH",
    )

    processed_data_path: str = Field(
        default="data/processed/chunks.jsonl",
        alias="PROCESSED_DATA_PATH",
    )

    chroma_path: str = Field(
        default="data/chroma",
        alias="CHROMA_PATH",
    )

    class Config:
        """Pydantic configuration."""

        env_file = ".env"
        extra = "ignore"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    Return a cached Settings instance.

    Using lru_cache prevents reloading environment variables
    every time the settings object is requested.
    """
    return Settings()