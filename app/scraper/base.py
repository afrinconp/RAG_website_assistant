from abc import ABC, abstractmethod
from typing import List, Dict


class BaseScraper(ABC):
    @abstractmethod
    def scrape(self) -> List[Dict]:
        """Return scraped documents as dictionaries."""
