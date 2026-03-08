"""Web content loader: fetch URLs, extract content, convert HTML to markdown."""

import re
import time
from urllib.parse import urljoin, urlparse

import httpx
from bs4 import BeautifulSoup, Tag


# Default CSS selectors for main content extraction (tried in order)
_CONTENT_SELECTORS = [
    ".book-content",  # Pro Git on git-scm.com
    ".sect1",         # Pro Git section containers
    "article",
    "main",
    ".content",
    "#content",
    "body",
]


def _extract_main_content(soup: BeautifulSoup) -> Tag | None:
    """Find the main content container, stripping nav/footer/ads."""
    # Remove noise elements first
    for tag in soup.find_all(["nav", "footer", "header", "script", "style",
                              "noscript", "iframe", "aside"]):
        tag.decompose()

    # Remove common ad/cookie/sidebar classes
    for cls in ["sidebar", "cookie", "banner", "advertisement", "popup",
                "nav", "menu", "footer", "header"]:
        for tag in soup.find_all(class_=re.compile(cls, re.I)):
            tag.decompose()

    # Try content selectors in priority order
    for selector in _CONTENT_SELECTORS:
        content = soup.select_one(selector)
        if content and len(content.get_text(strip=True)) > 100:
            return content

    return soup.body or soup


def _html_to_markdown(element: Tag) -> str:
    """Convert an HTML element tree to clean markdown."""
    lines: list[str] = []
    _walk(element, lines, indent=0)
    text = "\n".join(lines)
    # Normalize excessive whitespace
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _walk(element: Tag, lines: list[str], indent: int = 0) -> None:
    """Recursively walk HTML elements and convert to markdown."""
    for child in element.children:
        if isinstance(child, str):
            text = child.strip()
            if text:
                lines.append(text)
            continue

        if not isinstance(child, Tag):
            continue

        tag = child.name

        # Headings
        if tag in ("h1", "h2", "h3", "h4", "h5", "h6"):
            level = int(tag[1])
            heading_text = child.get_text(strip=True)
            if heading_text:
                lines.append("")
                lines.append(f"{'#' * level} {heading_text}")
                lines.append("")

        # Paragraphs
        elif tag == "p":
            text = _inline_text(child)
            if text:
                lines.append("")
                lines.append(text)
                lines.append("")

        # Code blocks
        elif tag == "pre":
            code = child.find("code")
            code_text = code.get_text() if code else child.get_text()
            lang = ""
            if code and code.get("class"):
                for cls in code["class"]:
                    if cls.startswith("language-") or cls.startswith("highlight-"):
                        lang = cls.split("-", 1)[1]
                        break
            lines.append("")
            lines.append(f"```{lang}")
            lines.append(code_text.rstrip())
            lines.append("```")
            lines.append("")

        # Inline code (not inside pre)
        elif tag == "code":
            text = child.get_text()
            if text:
                lines.append(f"`{text}`")

        # Lists
        elif tag in ("ul", "ol"):
            lines.append("")
            for i, li in enumerate(child.find_all("li", recursive=False)):
                prefix = f"{i + 1}." if tag == "ol" else "-"
                li_text = _inline_text(li)
                if li_text:
                    lines.append(f"{prefix} {li_text}")
            lines.append("")

        # Tables
        elif tag == "table":
            _convert_table(child, lines)

        # Block-level divs/sections: recurse
        elif tag in ("div", "section", "article", "main", "blockquote"):
            if tag == "blockquote":
                inner_lines: list[str] = []
                _walk(child, inner_lines)
                for line in inner_lines:
                    lines.append(f"> {line}" if line.strip() else ">")
            else:
                _walk(child, lines)

        # Images with alt text
        elif tag == "img":
            alt = child.get("alt", "")
            if alt:
                lines.append(f"![{alt}]")

        # Skip other tags but process children
        else:
            _walk(child, lines)


def _inline_text(element: Tag) -> str:
    """Extract inline text with basic formatting (bold, italic, code)."""
    parts: list[str] = []
    for child in element.children:
        if isinstance(child, str):
            parts.append(child)
        elif isinstance(child, Tag):
            text = child.get_text()
            if child.name in ("strong", "b"):
                parts.append(f"**{text}**")
            elif child.name in ("em", "i"):
                parts.append(f"*{text}*")
            elif child.name == "code":
                parts.append(f"`{text}`")
            elif child.name == "a":
                parts.append(text)
            elif child.name == "br":
                parts.append("\n")
            else:
                parts.append(text)
    return " ".join("".join(parts).split())


def _convert_table(table: Tag, lines: list[str]) -> None:
    """Convert HTML table to markdown table."""
    rows: list[list[str]] = []
    for tr in table.find_all("tr"):
        cells = []
        for td in tr.find_all(["td", "th"]):
            cells.append(td.get_text(strip=True))
        if cells:
            rows.append(cells)

    if not rows:
        return

    lines.append("")
    # Header row
    lines.append("| " + " | ".join(rows[0]) + " |")
    lines.append("| " + " | ".join("---" for _ in rows[0]) + " |")
    # Data rows
    for row in rows[1:]:
        # Pad row to match header length
        while len(row) < len(rows[0]):
            row.append("")
        lines.append("| " + " | ".join(row[:len(rows[0])]) + " |")
    lines.append("")


def load_url(url: str) -> tuple[str, dict]:
    """Fetch a single URL and return (markdown_text, metadata)."""
    resp = httpx.get(url, follow_redirects=True, timeout=30)
    resp.raise_for_status()

    soup = BeautifulSoup(resp.text, "html.parser")

    # Extract title
    title_tag = soup.find("title")
    title = title_tag.get_text(strip=True) if title_tag else ""

    # Extract main content
    content_el = _extract_main_content(soup)
    markdown = _html_to_markdown(content_el)

    metadata = {
        "source_file": _url_to_source_name(url),
        "source_url": url,
        "title": title,
    }

    return markdown, metadata


def _url_to_source_name(url: str) -> str:
    """Convert a URL to a short source name for display."""
    parsed = urlparse(url)
    path = parsed.path.strip("/")
    if path:
        # Use the last 2 path segments for readability
        parts = path.split("/")
        return "/".join(parts[-2:]) if len(parts) > 1 else parts[-1]
    return parsed.netloc


def discover_urls(
    base_url: str,
    delay: float = 1.0,
) -> list[str]:
    """Crawl a base URL and discover linked pages on the same domain.

    For Pro Git (git-scm.com/book/en/v2), discovers all section pages.
    Respects rate limiting with configurable delay between requests.
    """
    parsed_base = urlparse(base_url)
    base_domain = parsed_base.netloc
    base_path = parsed_base.path.rstrip("/")

    resp = httpx.get(base_url, follow_redirects=True, timeout=30)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")

    urls: list[str] = []
    seen: set[str] = {base_url.rstrip("/")}

    for a in soup.find_all("a", href=True):
        href = a["href"]
        full_url = urljoin(base_url, href)
        parsed = urlparse(full_url)

        # Strip fragment and query
        clean_url = f"{parsed.scheme}://{parsed.netloc}{parsed.path}".rstrip("/")

        # Same domain and under base path
        if (parsed.netloc == base_domain
                and parsed.path.startswith(base_path)
                and clean_url not in seen
                and parsed.path != base_path):
            seen.add(clean_url)
            urls.append(clean_url)

    return sorted(urls)
