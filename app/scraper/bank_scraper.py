from collections import deque
from datetime import datetime, timezone
from typing import Deque, Dict, List, Set
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup

from app.scraper.base import BaseScraper

USER_AGENT = (
    "Mozilla/5.0 "
    "RAG Technical Assessment Bot; educational use"
)

REMOVABLE_TAGS = (
    "script",
    "style",
    "noscript",
    "svg",
)


class BankScraper(BaseScraper):
    """
    Scraper implementation for banking websites.

    Responsibilities:
    - Crawl pages within the same domain.
    - Extract visible text.
    - Store page metadata.
    - Handle request failures gracefully.
    """

    def __init__(
        self,
        start_url: str,
        max_pages: int = 20,
        timeout: int = 15,
    ) -> None:
        """
        Initialize scraper configuration.

        Args:
            start_url: Initial URL to crawl.
            max_pages: Maximum number of pages to visit.
            timeout: HTTP request timeout in seconds.
        """
        self.start_url = start_url
        self.max_pages = max_pages
        self.timeout = timeout

        self.domain = urlparse(start_url).netloc

        self.headers = {
            "User-Agent": USER_AGENT,
        }

    def scrape(self) -> List[Dict]:
        """
        Crawl and extract content from pages.

        Returns:
            List of scraped documents.
        """
        visited: Set[str] = set()

        queue: Deque[str] = deque(
            [self.start_url]
        )

        documents: List[Dict] = []

        while (
            queue
            and len(visited) < self.max_pages
        ):
            current_url = queue.popleft()

            if current_url in visited:
                continue

            visited.add(current_url)

            response = self._fetch_page(
                current_url
            )

            if response is None:
                documents.append(
                    self._build_error_document(
                        current_url
                    )
                )
                continue

            soup = BeautifulSoup(
                response.text,
                "lxml",
            )

            documents.append(
                self._extract_document(
                    current_url,
                    soup,
                )
            )

            self._enqueue_links(
                current_url,
                soup,
                queue,
                visited,
            )

        return documents

    def _fetch_page(
        self,
        url: str,
    ) -> requests.Response | None:
        """
        Fetch a webpage.

        Args:
            url: URL to retrieve.

        Returns:
            Response object or None.
        """
        try:
            response = requests.get(
                url,
                headers=self.headers,
                timeout=self.timeout,
            )

            response.raise_for_status()

            return response

        except requests.RequestException:
            return None

    def _extract_document(
        self,
        url: str,
        soup: BeautifulSoup,
    ) -> Dict:
        """
        Extract text content from a page.

        Args:
            url: Page URL.
            soup: Parsed HTML document.

        Returns:
            Document dictionary.
        """
        title = (
            soup.title.get_text(strip=True)
            if soup.title
            else url
        )

        for tag in soup(REMOVABLE_TAGS):
            tag.decompose()

        text = " ".join(
            soup.get_text(" ").split()
        )

        return {
            "url": url,
            "title": title,
            "content": text,
            "error": None,
            "scraped_at": (
                datetime.now(
                    timezone.utc
                ).isoformat()
            ),
        }

    def _build_error_document(
        self,
        url: str,
    ) -> Dict:
        """
        Create a fallback document when a page fails.

        Args:
            url: Failed URL.

        Returns:
            Error document.
        """
        return {
            "url": url,
            "title": "ERROR",
            "content": "",
            "error": "Request failed",
            "scraped_at": (
                datetime.now(
                    timezone.utc
                ).isoformat()
            ),
        }

    def _enqueue_links(
        self,
        current_url: str,
        soup: BeautifulSoup,
        queue: Deque[str],
        visited: Set[str],
    ) -> None:
        """
        Discover and enqueue new links.

        Args:
            current_url: Current page URL.
            soup: Parsed HTML document.
            queue: Crawl queue.
            visited: Visited URLs.
        """
        for link in soup.find_all(
            "a",
            href=True,
        ):
            next_url = (
                urljoin(
                    current_url,
                    link["href"],
                )
                .split("#")[0]
            )

            parsed_url = urlparse(next_url)

            is_same_domain = (
                parsed_url.scheme
                in {"http", "https"}
                and parsed_url.netloc
                == self.domain
            )

            crawl_limit_ok = (
                len(visited) + len(queue)
                < self.max_pages * 3
            )

            if (
                is_same_domain
                and next_url not in visited
                and crawl_limit_ok
            ):
                queue.append(next_url)