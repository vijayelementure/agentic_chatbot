from __future__ import annotations
from collections import deque
from dataclasses import dataclass
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup
from playwright.async_api import async_playwright
import asyncio
from agentic_rag.logging_config import get_logger
from agentic_rag.settings import Settings, get_settings

logger = get_logger(__name__)

SKIP_TAGS = ("script", "style", "nav", "footer", "noscript", "svg")


@dataclass(frozen=True)
class CrawledPage:
    url: str
    text: str
    source: str


class WebsiteCrawler:
    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()
        self.base_url = self.settings.WEBSITE_URL
        self.max_pages = self.settings.MAX_CRAWL_PAGES
        self.timeout = self.settings.CRAWL_REQUEST_TIMEOUT_SECONDS
        self.domain = urlparse(self.base_url).netloc

    def is_same_domain(self, url: str) -> bool:
        return urlparse(url).netloc in ("", self.domain)

    @staticmethod
    def clean_text(html: str) -> str:
        soup = BeautifulSoup(html, "lxml")
        parts = []

        title_tag = soup.find("title")
        if title_tag and title_tag.string:
            parts.append(f"Title: {title_tag.string.strip()}")

        meta_desc = soup.find("meta", attrs={"name": "description"})
        if meta_desc and meta_desc.get("content"):
            parts.append(f"Description: {meta_desc['content'].strip()}")

        og_title = soup.find("meta", attrs={"property": "og:title"})
        if og_title and og_title.get("content"):
            parts.append(f"OG Title: {og_title['content'].strip()}")

        og_desc = soup.find("meta", attrs={"property": "og:description"})
        if og_desc and og_desc.get("content"):
            parts.append(f"OG Description: {og_desc['content'].strip()}")

        for tag in soup(SKIP_TAGS):
            tag.decompose()

        body_text = soup.get_text(separator="\n")
        lines = [ln.strip() for ln in body_text.splitlines()]
        cleaned_body = "\n".join(ln for ln in lines if ln)
        if cleaned_body:
            parts.append(cleaned_body)

        return "\n\n".join(parts)

    async def fetch_page(self, url: str, page) -> str:
        await page.goto(url, wait_until="domcontentloaded", timeout=self.timeout * 1000)
        await asyncio.sleep(1)  # Wait for JS to render a bit
        html = await page.content()
        return html

    async def crawl(self) -> list[CrawledPage]:
        visited: set[str] = set()
        queue = deque([self.base_url])
        pages: list[CrawledPage] = []

        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            page = await browser.new_page()
            page.set_default_timeout(self.timeout * 1000)

            while queue and len(visited) < self.max_pages:
                url = queue.popleft()
                norm_url = url.split("#")[0].rstrip("/")
                if norm_url in visited:
                    continue
                visited.add(norm_url)

                try:
                    html = await self.fetch_page(url, page)
                except Exception as e:
                    logger.warning("Failed to fetch %s: %s", url, e)
                    continue

                text = self.clean_text(html)
                if text.strip():
                    pages.append(CrawledPage(url=url, text=text, source=url))
                soup = BeautifulSoup(html, "lxml")
                for a in soup.find_all("a", href=True):
                    next_url = urljoin(url, a["href"]).split("#")[0]
                    if (
                        next_url.startswith("http")
                        and self.is_same_domain(next_url)
                        and next_url.rstrip("/") not in visited
                    ):
                        queue.append(next_url)

            await browser.close()

        logger.info("Crawled %d pages from %s", len(pages), self.base_url)
        return pages

    def crawl_sync(self) -> list[CrawledPage]:
        return asyncio.run(self.crawl())
