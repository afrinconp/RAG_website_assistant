from app.config.settings import get_settings
from app.scraper.factory import ScraperFactory
from app.utils.jsonl import write_jsonl


def run_scraping_pipeline() -> int:
    settings = get_settings()
    scraper = ScraperFactory.create(
        bank_name=settings.bank_name,
        start_url=settings.start_url,
        max_pages=settings.max_pages,
        timeout=settings.request_timeout,
    )
    documents = scraper.scrape()
    write_jsonl(settings.raw_data_path, documents)
    return len(documents)
