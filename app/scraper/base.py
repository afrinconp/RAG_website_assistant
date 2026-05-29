"""Abstract scraper interface."""

from abc import ABC, abstractmethod
from typing import Dict, List

Document = Dict


class BaseScraper(ABC):
    """
    Abstract base class for all scraper implementations.

    Concrete scraper classes must implement the `scrape`
    method and return a collection of documents that can
    later be processed and indexed by the RAG pipeline.
    """

    @abstractmethod
    def scrape(self) -> List[Document]:
        """
        Scrape content from a source.

        Returns:
            List of document dictionaries.

        Example:
            [
                {
                    "url": "https://example.com",
                    "title": "Example Page",
                    "content": "Page content...",
                    "error": None,
                    "scraped_at": "2026-01-01T12:00:00Z",
                }
            ]
        """
        raise NotImplementedError