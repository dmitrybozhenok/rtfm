"""File loaders for different document types."""

import re
from pathlib import Path


def _strip_bold(text: str) -> str:
    """Remove markdown bold markers from text."""
    return text.replace("**", "").strip()


def _is_page_footer(stripped: str) -> bool:
    """Detect page footer/header lines from PDF extraction.

    Matches patterns like:
      **RAG** **|** **259**
      **260** **|** **Chapter 6: RAG and Agents**
      **49**
    """
    # Bold page number only: **123**
    if re.fullmatch(r"\*\*\d+\*\*", stripped):
        return True
    # Footer with pipe separator: **Text** **|** **digits** or **digits** **|** **Text**
    if re.search(r"\*\*\|\*\*", stripped) and re.search(r"\*\*\d+\*\*", stripped):
        return True
    return False


def _promote_pdf_headings(text: str) -> str:
    """Convert bold-only lines from PDF extraction into markdown headings.

    pymupdf4llm often renders PDF section headings as bold text rather than
    markdown headings. This detects standalone bold lines and promotes them:
      **CHAPTER N** + #### **Title**  →  # Chapter N: Title
      **Section Name**               →  ## Section Name
    """
    lines = text.split("\n")
    result = []
    i = 0
    while i < len(lines):
        stripped = lines[i].strip()

        # Pattern: **CHAPTER N** followed by #### **Title...**
        chapter_match = re.fullmatch(r"\*\*CHAPTER\s+(\d+)\*\*", stripped)
        if chapter_match:
            chapter_num = chapter_match.group(1)
            # Look ahead for the #### title line
            if i + 1 < len(lines):
                next_stripped = lines[i + 1].strip()
                title_match = re.match(r"#{1,6}\s+(.+)", next_stripped)
                if title_match:
                    title = _strip_bold(title_match.group(1))
                    result.append(f"# Chapter {chapter_num}: {title}")
                    i += 2
                    continue
            # Standalone chapter line without title
            result.append(f"# Chapter {chapter_num}")
            i += 1
            continue

        # Standalone bold line → section heading
        # Must be a single bold group (not multiple like table headers)
        # e.g. **Embedding-based retrieval** but NOT **Term** **Document count**
        bold_heading = re.fullmatch(r"\*\*([^*]+)\*\*", stripped)
        if bold_heading:
            heading_text = bold_heading.group(1).strip()
            # Skip page numbers, single characters, icon chars (Unicode PUA)
            if re.fullmatch(r"\d+", heading_text):
                i += 1
                continue
            if len(heading_text) < 3:
                i += 1
                continue
            # Skip Unicode private-use-area characters (icon fonts)
            if all(0xE000 <= ord(c) <= 0xF8FF or c.isspace() for c in heading_text):
                i += 1
                continue
            # Promote to ## heading
            result.append(f"## {heading_text}")
            i += 1
            continue

        result.append(lines[i])
        i += 1

    return "\n".join(result)


def _strip_toc(text: str) -> str:
    """Remove Table of Contents pages from PDF text.

    TOC lines are characterized by dotted leaders (. . . . .) connecting
    section names to page numbers.
    """
    lines = text.split("\n")
    cleaned = []
    in_toc = False
    # Count consecutive TOC-like lines to detect TOC region
    toc_line_count = 0

    for line in lines:
        stripped = line.strip()
        # Detect TOC header
        if re.search(r"Table of Contents", stripped, re.IGNORECASE):
            in_toc = True
            toc_line_count = 0
            continue

        # TOC lines have dotted leaders: "Section Name. . . . . . . . 42"
        # or just indented section names with page numbers
        is_toc_line = bool(re.search(r"\.(\s*\.){4,}", stripped))

        if in_toc:
            if is_toc_line:
                toc_line_count += 1
                continue
            # Allow blank lines within TOC
            if not stripped:
                continue
            # Exit TOC after seeing enough TOC lines followed by non-TOC content
            if toc_line_count > 5:
                in_toc = False
                # This line is real content, keep it
                cleaned.append(line)
                continue
            # Still might be in TOC (indented section names without dots)
            # Check if it's a short line that could be a TOC entry
            if len(stripped) < 80 and re.search(r"\d+$", stripped):
                continue
            in_toc = False

        if is_toc_line:
            continue

        cleaned.append(line)

    return "\n".join(cleaned)


def _clean_pdf_text(text: str) -> str:
    """Post-process PDF-extracted text to remove artifacts and fix headings."""
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
        # Skip PDF page footers like **RAG** **|** **259**
        if _is_page_footer(stripped):
            continue
        # Skip lines that are only Unicode PUA icon characters (icon fonts)
        text_only = stripped.lstrip("#").strip()
        if text_only and all(
            0xE000 <= ord(c) <= 0xF8FF or c.isspace() for c in text_only
        ):
            continue
        cleaned.append(line)

    text = "\n".join(cleaned)
    # Strip Table of Contents
    text = _strip_toc(text)
    # Promote bold-only lines to markdown headings
    text = _promote_pdf_headings(text)
    # Strip residual bold markers from existing markdown headings
    # e.g. #### **Title** → #### Title
    text = re.sub(
        r"^(#{1,6})\s+\*\*(.+?)\*\*\s*$",
        lambda m: f"{m.group(1)} {m.group(2)}",
        text,
        flags=re.MULTILINE,
    )
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
