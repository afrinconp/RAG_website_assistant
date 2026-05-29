from app.scraper.bank_scraper import BankScraper


class ScraperFactory:
    """Factory Pattern: creates scraper implementations based on a bank name."""

    @staticmethod
    def create(bank_name: str, start_url: str, max_pages: int, timeout: int):
        normalized = bank_name.lower().strip()
        if normalized in {"bbva", "bank", "generic"}:
            return BankScraper(start_url=start_url, max_pages=max_pages, timeout=timeout)
        raise ValueError(f"Unsupported bank scraper: {bank_name}")
