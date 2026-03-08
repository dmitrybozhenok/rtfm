"""Tests for web_loader: HTML→markdown conversion, URL discovery, and loading."""

from unittest.mock import MagicMock, patch

import pytest

from rtfm.ingest.web_loader import (
    _convert_table,
    _extract_main_content,
    _html_to_markdown,
    _inline_text,
    _url_to_source_name,
    discover_urls,
    load_url,
)

try:
    from bs4 import BeautifulSoup, Tag
except ImportError:
    pytest.skip("beautifulsoup4 not installed", allow_module_level=True)


# --- HTML to Markdown conversion ---

class TestHtmlToMarkdown:
    def _soup(self, html: str) -> Tag:
        return BeautifulSoup(html, "html.parser").body or BeautifulSoup(html, "html.parser")

    def test_headings(self):
        html = "<div><h1>Title</h1><h2>Subtitle</h2><h3>Section</h3></div>"
        md = _html_to_markdown(self._soup(html))
        assert "# Title" in md
        assert "## Subtitle" in md
        assert "### Section" in md

    def test_paragraphs(self):
        html = "<div><p>Hello world.</p><p>Second paragraph.</p></div>"
        md = _html_to_markdown(self._soup(html))
        assert "Hello world." in md
        assert "Second paragraph." in md

    def test_code_blocks(self):
        html = '<div><pre><code class="language-python">print("hello")</code></pre></div>'
        md = _html_to_markdown(self._soup(html))
        assert "```python" in md
        assert 'print("hello")' in md
        assert "```" in md

    def test_code_block_no_language(self):
        html = "<div><pre><code>some code</code></pre></div>"
        md = _html_to_markdown(self._soup(html))
        assert "```\nsome code\n```" in md

    def test_unordered_list(self):
        html = "<div><ul><li>Item 1</li><li>Item 2</li></ul></div>"
        md = _html_to_markdown(self._soup(html))
        assert "- Item 1" in md
        assert "- Item 2" in md

    def test_ordered_list(self):
        html = "<div><ol><li>First</li><li>Second</li></ol></div>"
        md = _html_to_markdown(self._soup(html))
        assert "1. First" in md
        assert "2. Second" in md

    def test_bold_italic_inline_code(self):
        html = "<div><p><strong>bold</strong> and <em>italic</em> and <code>code</code></p></div>"
        md = _html_to_markdown(self._soup(html))
        assert "**bold**" in md
        assert "*italic*" in md
        assert "`code`" in md

    def test_blockquote(self):
        html = "<div><blockquote><p>Quoted text</p></blockquote></div>"
        md = _html_to_markdown(self._soup(html))
        assert "> " in md
        assert "Quoted text" in md

    def test_table(self):
        html = """<div><table>
        <tr><th>Name</th><th>Age</th></tr>
        <tr><td>Alice</td><td>30</td></tr>
        </table></div>"""
        md = _html_to_markdown(self._soup(html))
        assert "| Name | Age |" in md
        assert "| --- | --- |" in md
        assert "| Alice | 30 |" in md

    def test_image_alt_text(self):
        html = '<div><img alt="A diagram" src="img.png"></div>'
        md = _html_to_markdown(self._soup(html))
        assert "![A diagram]" in md

    def test_strips_nav_footer(self):
        html = "<div><nav>Navigation</nav><main><p>Content</p></main><footer>Footer</footer></div>"
        soup = BeautifulSoup(html, "html.parser")
        content = _extract_main_content(soup)
        md = _html_to_markdown(content)
        assert "Content" in md
        assert "Navigation" not in md
        assert "Footer" not in md


# --- Content extraction ---

class TestExtractMainContent:
    def test_extracts_article(self):
        html = "<html><body><div class='sidebar'>Nav</div><article><p>Main content here that is long enough to pass the 100 character threshold for content extraction testing.</p></article></body></html>"
        soup = BeautifulSoup(html, "html.parser")
        content = _extract_main_content(soup)
        text = content.get_text()
        assert "Main content" in text

    def test_extracts_main_tag(self):
        html = "<html><body><main><p>This is the main content of the page which should be long enough for extraction threshold checks.</p></main></body></html>"
        soup = BeautifulSoup(html, "html.parser")
        content = _extract_main_content(soup)
        assert "main content" in content.get_text()

    def test_strips_sidebar_classes(self):
        html = '<html><body><div class="sidebar">Side</div><div class="content"><p>Real content that is sufficiently long to pass the hundred character threshold for content extraction.</p></div></body></html>'
        soup = BeautifulSoup(html, "html.parser")
        content = _extract_main_content(soup)
        text = content.get_text()
        assert "Side" not in text
        assert "Real content" in text


# --- URL helpers ---

class TestUrlToSourceName:
    def test_simple_path(self):
        assert _url_to_source_name("https://example.com/docs/guide") == "docs/guide"

    def test_deep_path(self):
        name = _url_to_source_name("https://git-scm.com/book/en/v2/Git-Basics-Getting-a-Repository")
        assert "v2/Git-Basics-Getting-a-Repository" == name

    def test_root_url(self):
        assert _url_to_source_name("https://example.com/") == "example.com"
        assert _url_to_source_name("https://example.com") == "example.com"


# --- URL discovery ---

class TestDiscoverUrls:
    @patch("rtfm.ingest.web_loader.httpx.get")
    def test_discovers_same_domain_links(self, mock_get):
        html = """<html><body>
        <a href="/book/en/v2/Chapter-1">Ch1</a>
        <a href="/book/en/v2/Chapter-2">Ch2</a>
        <a href="https://other.com/page">External</a>
        </body></html>"""
        mock_resp = MagicMock()
        mock_resp.text = html
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        urls = discover_urls("https://git-scm.com/book/en/v2")

        assert len(urls) == 2
        assert "https://git-scm.com/book/en/v2/Chapter-1" in urls
        assert "https://git-scm.com/book/en/v2/Chapter-2" in urls
        # External link should NOT be included
        assert not any("other.com" in u for u in urls)

    @patch("rtfm.ingest.web_loader.httpx.get")
    def test_deduplicates_urls(self, mock_get):
        html = """<html><body>
        <a href="/book/en/v2/Page">Link 1</a>
        <a href="/book/en/v2/Page">Link 2</a>
        <a href="/book/en/v2/Page#section">Link 3</a>
        </body></html>"""
        mock_resp = MagicMock()
        mock_resp.text = html
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        urls = discover_urls("https://git-scm.com/book/en/v2")
        assert len(urls) == 1

    @patch("rtfm.ingest.web_loader.httpx.get")
    def test_excludes_base_url(self, mock_get):
        html = """<html><body>
        <a href="/book/en/v2">Same page</a>
        <a href="/book/en/v2/Other">Other</a>
        </body></html>"""
        mock_resp = MagicMock()
        mock_resp.text = html
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        urls = discover_urls("https://git-scm.com/book/en/v2")
        assert len(urls) == 1
        assert "Other" in urls[0]


# --- load_url ---

class TestLoadUrl:
    @patch("rtfm.ingest.web_loader.httpx.get")
    def test_load_url_returns_markdown_and_metadata(self, mock_get):
        html = """<html><head><title>Git Basics</title></head>
        <body><main><h1>Getting Started</h1>
        <p>This is the introduction to Git basics which provides enough content for the extraction threshold.</p>
        </main></body></html>"""
        mock_resp = MagicMock()
        mock_resp.text = html
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        text, metadata = load_url("https://git-scm.com/book/en/v2/Git-Basics")

        assert "# Getting Started" in text
        assert "introduction to Git" in text
        assert metadata["source_url"] == "https://git-scm.com/book/en/v2/Git-Basics"
        assert "Git-Basics" in metadata["source_file"]
        assert metadata["title"] == "Git Basics"

    @patch("rtfm.ingest.web_loader.httpx.get")
    def test_load_url_strips_scripts_and_nav(self, mock_get):
        html = """<html><body>
        <nav>Menu items</nav>
        <script>alert('xss')</script>
        <main><p>Real content that should appear in the output and be long enough for extraction.</p></main>
        <footer>Copyright</footer>
        </body></html>"""
        mock_resp = MagicMock()
        mock_resp.text = html
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        text, _ = load_url("https://example.com/page")

        assert "Real content" in text
        assert "Menu items" not in text
        assert "alert" not in text
        assert "Copyright" not in text
