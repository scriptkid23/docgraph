from __future__ import annotations

from docgraph.config import Config


def _markdown_from_result(result) -> str:
    md = result.markdown
    if md is None:
        return ""
    if isinstance(md, str):
        return md
    for attr in ("fit_markdown", "raw_markdown"):
        if hasattr(md, attr):
            value = getattr(md, attr)
            if value and str(value).strip():
                return str(value)
    return str(md)


def _title_from_result(result, url: str) -> str:
    metadata = getattr(result, "metadata", None) or {}
    if isinstance(metadata, dict):
        title = metadata.get("title") or metadata.get("og:title") or ""
        if title and str(title).strip():
            return str(title).strip()[:200]
    from docgraph.ingest.urls import url_display_name

    return url_display_name(url)


class UrlCrawler:
    """Async context manager wrapping crawl4ai for single-page fetches."""

    def __init__(self, cfg: Config) -> None:
        self._cfg = cfg
        self._crawler = None

    async def __aenter__(self) -> UrlCrawler:
        try:
            from crawl4ai import AsyncWebCrawler, BrowserConfig
        except ImportError as exc:
            raise RuntimeError(
                "crawl4ai not installed. Run: poetry install -E crawl && crawl4ai-setup"
            ) from exc

        browser_config = BrowserConfig(headless=True)
        self._crawler = AsyncWebCrawler(config=browser_config)
        await self._crawler.__aenter__()
        return self

    async def __aexit__(self, *args) -> None:
        if self._crawler is not None:
            await self._crawler.__aexit__(*args)
            self._crawler = None

    async def crawl(self, url: str) -> tuple[str, str]:
        """Fetch one URL and return (markdown, title)."""
        if self._crawler is None:
            raise RuntimeError("UrlCrawler is not started; use async with")

        from crawl4ai import CrawlerRunConfig

        run_config = CrawlerRunConfig(
            page_timeout=self._cfg.crawl_timeout_sec * 1000,
        )
        result = await self._crawler.arun(url=url, config=run_config)

        success = getattr(result, "success", True)
        if not success:
            msg = getattr(result, "error_message", None) or "crawl failed"
            raise ValueError(str(msg))

        markdown = _markdown_from_result(result)
        if not markdown.strip():
            raise ValueError("empty markdown from crawl")

        return markdown, _title_from_result(result, url)
