
from agentic_rag.settings import Settings
from agentic_rag.web_crawler import WebsiteCrawler


class FakeResponse:
    def __init__(self, html: str, content_type: str = "text/html"):
        self.text = html
        self.headers = {"Content-Type": content_type}

    def raise_for_status(self):
        pass


def test_clean_text_strips_scripts_and_nav():
    html = """
    <html><body>
      <nav>Menu</nav>
      <script>var x = 1;</script>
      <h1>Welcome</h1>
      <p>Some real content.</p>
      <footer>Footer text</footer>
    </body></html>
    """
    cleaned = WebsiteCrawler._clean_text(html)
    assert "Welcome" in cleaned
    assert "Some real content." in cleaned
    assert "Menu" not in cleaned
    assert "Footer text" not in cleaned


def test_crawl_stays_within_domain_and_page_limit(monkeypatch):
    settings = Settings(gemini_api_key="x", website_url="https://example.com/", max_crawl_pages=2)
    crawler = WebsiteCrawler(settings)

    pages = {
        "https://example.com/": (
            '<html><body><p>Home</p>'
            '<a href="/about">About</a>'
            '<a href="https://external.com/x">External</a>'
            "</body></html>"
        ),
        "https://example.com/about": '<html><body><p>About us</p></body></html>',
    }

    def fake_fetch(self, url):
        key = url.rstrip("/")
        if key == "https://example.com":
            key = "https://example.com/"
        return FakeResponse(pages[key])

    monkeypatch.setattr(WebsiteCrawler, "_fetch", fake_fetch)

    result = crawler.crawl()
    urls = [p.url for p in result]
    assert "https://example.com/" in urls
    assert all("external.com" not in u for u in urls)
    assert len(result) <= 2
