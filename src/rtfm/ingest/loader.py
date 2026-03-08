"""File loaders for different document types."""

from pathlib import Path


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

        text = pymupdf4llm.to_markdown(str(path))
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
