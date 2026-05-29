from collections import deque
from datetime import datetime, timezone
from typing import Dict, List, Set
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup

from app.scraper.base import BaseScraper


class BankScraper(BaseScraper):
    def __init__(self, start_url: str, max_pages: int = 20, timeout: int = 15):
        self.start_url = start_url
        self.max_pages = max_pages
        self.timeout = timeout
        self.domain = urlparse(start_url).netloc
        self.headers = {
            "User-Agent": "Mozilla/5.0 RAG Technical Assessment Bot; educational use"
        }

    def scrape(self) -> List[Dict]:
        visited: Set[str] = set()
        queue = deque([self.start_url])
        documents: List[Dict] = []

        while queue and len(visited) < self.max_pages:
            url = queue.popleft()
            if url in visited:
                continue
            visited.add(url)

            try:
                response = requests.get(url, headers=self.headers, timeout=self.timeout)
                response.raise_for_status()
            except requests.RequestException as exc:
                documents.append({
                    "url": url,
                    "title": "ERROR",
                    "content": "",
                    "error": str(exc),
                    "scraped_at": datetime.now(timezone.utc).isoformat(),
                })
                continue

            soup = BeautifulSoup(response.text, "lxml")
            title = soup.title.get_text(strip=True) if soup.title else url

            for tag in soup(["script", "style", "noscript", "svg"]):
                tag.decompose()

            text = " ".join(soup.get_text(" ").split())
            documents.append({
                "url": url,
                "title": title,
                "content": text,
                "error": None,
                "scraped_at": datetime.now(timezone.utc).isoformat(),
            })

            for link in soup.find_all("a", href=True):
                next_url = urljoin(url, link["href"]).split("#")[0]
                parsed = urlparse(next_url)
                if parsed.scheme in {"http", "https"} and parsed.netloc == self.domain:
                    if next_url not in visited and len(visited) + len(queue) < self.max_pages * 3:
                        queue.append(next_url)

        return documents
