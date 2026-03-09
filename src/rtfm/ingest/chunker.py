"""Paragraph-aware text chunking with overlap."""

import re
from dataclasses import dataclass

from rtfm.config import settings


@dataclass
class Chunk:
    text: str
    chunk_index: int
    section: str
    metadata: dict


def _clean_overlap(overlap_text: str) -> str:
    """Remove content from previous sections in overlap text.

    If the overlap text contains a markdown heading, keep only the text
    from the last heading onward. This prevents content from a previous
    section from polluting the next chunk.
    """
    # Find the last heading in the overlap
    heading_pattern = re.compile(r"^#{1,6}\s+", re.MULTILINE)
    matches = list(heading_pattern.finditer(overlap_text))
    if matches:
        # Keep from the last heading onward
        last_heading_pos = matches[-1].start()
        return overlap_text[last_heading_pos:]
    return overlap_text


def chunk_text(
    text: str,
    metadata: dict,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> list[Chunk]:
    """Split text into paragraph-aware chunks with token-approximate sizing.

    Uses whitespace token approximation (~1 token per 4 chars).
    """
    chunk_size = chunk_size or settings.chunk_size
    chunk_overlap = chunk_overlap or settings.chunk_overlap

    # Approximate chars from tokens (rough: 1 token ≈ 4 chars)
    max_chars = chunk_size * 4
    overlap_chars = chunk_overlap * 4

    paragraphs = text.split("\n\n")
    chunks: list[Chunk] = []
    current_text = ""
    current_section = ""

    def _finalize_chunk(text_body: str, section: str) -> None:
        """Create a chunk from accumulated text."""
        chunk_body = text_body.strip()
        if not chunk_body:
            return
        # Prepend section heading if available and not already present
        if section and not chunk_body.startswith(f"## {section}"):
            chunk_body = f"## {section}\n\n{chunk_body}"
        chunks.append(
            Chunk(
                text=chunk_body,
                chunk_index=len(chunks),
                section=section,
                metadata=metadata,
            )
        )

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        is_heading = para.startswith("#")

        # When we hit a new section heading, finalize the current chunk
        # so that previous-section content doesn't leak into the new section
        if is_heading and current_text.strip():
            _finalize_chunk(current_text, current_section)
            # Overlap within same section only — don't carry across headings
            current_text = ""

        # Track markdown headings as section names
        if is_heading:
            first_line = para.split("\n")[0]
            current_section = first_line.lstrip("# ").strip()

        # If adding this paragraph would exceed chunk_size, finalize current chunk
        if current_text and len(current_text) + len(para) + 2 > max_chars:
            _finalize_chunk(current_text, current_section)
            # Keep overlap from end of current chunk, but clean cross-section content
            if overlap_chars > 0 and len(current_text) > overlap_chars:
                current_text = _clean_overlap(current_text[-overlap_chars:])
            else:
                current_text = ""

        current_text += para + "\n\n"

    # Final chunk
    _finalize_chunk(current_text, current_section)

    return chunks
