"""Paragraph-aware text chunking with overlap."""

from dataclasses import dataclass

from rtfm.config import settings


@dataclass
class Chunk:
    text: str
    chunk_index: int
    section: str
    metadata: dict


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

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        # Track markdown headings as section names
        if para.startswith("#"):
            first_line = para.split("\n")[0]
            current_section = first_line.lstrip("# ").strip()

        # If adding this paragraph would exceed chunk_size, finalize current chunk
        if current_text and len(current_text) + len(para) + 2 > max_chars:
            chunk_body = current_text.strip()
            # Prepend section heading if available and not already present
            if current_section and not chunk_body.startswith(f"## {current_section}"):
                chunk_body = f"## {current_section}\n\n{chunk_body}"
            chunks.append(
                Chunk(
                    text=chunk_body,
                    chunk_index=len(chunks),
                    section=current_section,
                    metadata=metadata,
                )
            )
            # Keep overlap from end of current chunk
            if overlap_chars > 0 and len(current_text) > overlap_chars:
                current_text = current_text[-overlap_chars:]
            else:
                current_text = ""

        current_text += para + "\n\n"

    # Final chunk
    if current_text.strip():
        chunk_body = current_text.strip()
        if current_section and not chunk_body.startswith(f"## {current_section}"):
            chunk_body = f"## {current_section}\n\n{chunk_body}"
        chunks.append(
            Chunk(
                text=chunk_body,
                chunk_index=len(chunks),
                section=current_section,
                metadata=metadata,
            )
        )

    return chunks
