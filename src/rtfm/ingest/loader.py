"""File loaders for different document types."""

import re
from pathlib import Path


def _clean_pdf_text(text: str) -> str:
    """Post-process PDF-extracted text to remove artifacts."""
    lines = text.split("\n")
    cleaned = []
    for line in lines:
        stripped = line.strip()
        # Skip lines that are just page numbers (standalone digits)
        if re.fullmatch(r"\d+", stripped):
            continue
        # Skip common header/footer patterns like "Page 3 of 10"
        if re.fullmatch(r"[Pp]age\s+\d+(\s+of\s+\d+)?", stripped):
            continue
        cleaned.append(line)

    text = "\n".join(cleaned)
    # Normalize excessive whitespace: 3+ newlines → 2
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Fix broken dotted/hyphenated identifiers from PDF extraction:
    # "unique .chat .user -message .created" → "unique.chat.user-message.created"
    # Triggered when 2+ space-separated dot/hyphen segments appear (avoids false positives)
    def _collapse_identifier(m: re.Match) -> str:
        return re.sub(r" ([.\-])", r"\1", m.group(0))
    text = re.sub(r"\w(?:\s[.\-]\w+){2,}", _collapse_identifier, text)
    return text


def load_file(path: Path) -> tuple[str, dict]:
    """Load a file and return (text_content, metadata).

    Supports .md, .txt, and .pdf files.
    """
    suffix = path.suffix.lower()
    metadata = {"source_file": path.name}

    if suffix in (".md", ".txt"):
        text = path.read_text(encoding="utf-8")
    elif suffix == ".pdf":
        import pymupdf4llm

        text = pymupdf4llm.to_markdown(str(path), page_chunks=False)
        text = _clean_pdf_text(text)
    else:
        raise ValueError(f"Unsupported file type: {suffix}")

    return text, metadata


def discover_files(path: Path) -> list[Path]:
    """Discover all supported files under a path (file or directory)."""
    supported = {".md", ".txt", ".pdf"}
    if path.is_file():
        if path.suffix.lower() in supported:
            return [path]
        return []
    return [f for f in path.rglob("*") if f.suffix.lower() in supported]
