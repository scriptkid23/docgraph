import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from docgraph.config import Config
from docgraph.ingest.crawler import UrlCrawler, _markdown_from_result, _title_from_result


def test_markdown_from_result_string():
    result = MagicMock()
    result.markdown = "# Hello"
    assert _markdown_from_result(result) == "# Hello"


def test_markdown_from_result_fit_markdown():
    result = MagicMock()
    md = MagicMock()
    md.fit_markdown = "fit content"
    md.raw_markdown = "raw content"
    result.markdown = md
    assert _markdown_from_result(result) == "fit content"


def test_title_from_result_metadata():
    result = MagicMock()
    result.metadata = {"title": "My Page"}
    assert _title_from_result(result, "https://example.com") == "My Page"


def test_title_from_result_fallback():
    result = MagicMock()
    result.metadata = {}
    title = _title_from_result(result, "https://example.com/docs")
    assert "example.com" in title


@pytest.mark.asyncio
async def test_url_crawler_crawl(tmp_data_dir):
    cfg = Config(data_dir=tmp_data_dir)
    mock_result = MagicMock()
    mock_result.success = True
    mock_result.markdown = "# Crawled"
    mock_result.metadata = {"title": "Crawled Title"}

    mock_crawler = AsyncMock()
    mock_crawler.arun = AsyncMock(return_value=mock_result)
    mock_crawler.__aenter__ = AsyncMock(return_value=mock_crawler)
    mock_crawler.__aexit__ = AsyncMock(return_value=None)

    fake_crawl4ai = MagicMock()
    fake_crawl4ai.AsyncWebCrawler = MagicMock(return_value=mock_crawler)
    fake_crawl4ai.BrowserConfig = MagicMock()
    fake_crawl4ai.CrawlerRunConfig = MagicMock()

    with patch.dict(sys.modules, {"crawl4ai": fake_crawl4ai}):
        async with UrlCrawler(cfg) as crawler:
            md, title = await crawler.crawl("https://example.com/page")

    assert md == "# Crawled"
    assert title == "Crawled Title"
    mock_crawler.arun.assert_awaited_once()
