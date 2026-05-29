from app.scraper.bank_scraper import BankScraper
from app.scraper.base import BaseScraper


class ScraperFactory:
    """
    Factory Pattern implementation for scraper creation.

    This factory is responsible for instantiating the
    appropriate scraper implementation based on the
    provided bank name.
    """

    @staticmethod
    def create(
        bank_name: str,
        start_url: str,
        max_pages: int,
        timeout: int,
    ) -> BaseScraper:
        """
        Create a scraper instance.

        Args:
            bank_name: Bank identifier.
            start_url: Initial URL to crawl.
            max_pages: Maximum number of pages to scrape.
            timeout: HTTP request timeout in seconds.

        Returns:
            BaseScraper: Configured scraper instance.

        Raises:
            ValueError: If the bank is not supported.
        """
        normalized_bank = (
            bank_name
            .lower()
            .strip()
        )

        if normalized_bank in {
            "bbva",
            "bank",
            "generic",
        }:
            return BankScraper(
                start_url=start_url,
                max_pages=max_pages,
                timeout=timeout,
            )

        raise ValueError(
            f"Unsupported bank scraper: {bank_name}"
        )