"""Unit tests for rtfm.ingest.loader module."""

from pathlib import Path

import pytest

from rtfm.ingest.loader import (
    _clean_pdf_text,
    _is_page_footer,
    _promote_pdf_headings,
    _strip_toc,
    discover_files,
    load_file,
)


class TestLoadFile:
    """Tests for load_file()."""

    def test_load_markdown_file(self, tmp_path: Path):
        md = tmp_path / "readme.md"
        md.write_text("# Hello\n\nWorld", encoding="utf-8")

        text, metadata = load_file(md)

        assert text == "# Hello\n\nWorld"
        assert metadata["source_file"] == "readme.md"

    def test_load_text_file(self, tmp_path: Path):
        txt = tmp_path / "notes.txt"
        txt.write_text("Some plain text content", encoding="utf-8")

        text, metadata = load_file(txt)

        assert text == "Some plain text content"
        assert metadata["source_file"] == "notes.txt"

    def test_unsupported_file_type(self, tmp_path: Path):
        docx = tmp_path / "file.docx"
        docx.write_text("fake content", encoding="utf-8")

        with pytest.raises(ValueError, match="Unsupported file type: .docx"):
            load_file(docx)

    def test_load_empty_file(self, tmp_path: Path):
        empty = tmp_path / "empty.md"
        empty.write_text("", encoding="utf-8")

        text, metadata = load_file(empty)

        assert text == ""
        assert metadata["source_file"] == "empty.md"


class TestDiscoverFiles:
    """Tests for discover_files()."""

    def test_discover_files_directory(self, tmp_path: Path):
        (tmp_path / "doc.md").write_text("md", encoding="utf-8")
        (tmp_path / "notes.txt").write_text("txt", encoding="utf-8")
        (tmp_path / "image.png").write_bytes(b"\x89PNG")
        (tmp_path / "style.css").write_text("body{}", encoding="utf-8")

        result = discover_files(tmp_path)
        suffixes = sorted(f.suffix for f in result)

        assert suffixes == [".md", ".txt"]

    def test_discover_files_single_file(self, tmp_path: Path):
        md = tmp_path / "single.md"
        md.write_text("hello", encoding="utf-8")

        result = discover_files(md)

        assert result == [md]

    def test_discover_files_empty_directory(self, tmp_path: Path):
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()

        result = discover_files(empty_dir)

        assert result == []


class TestIsPageFooter:
    """Tests for _is_page_footer()."""

    def test_bold_page_number(self):
        assert _is_page_footer("**49**")
        assert _is_page_footer("**259**")

    def test_footer_with_pipe_and_page_number(self):
        assert _is_page_footer("**RAG** **|** **259**")
        assert _is_page_footer("**260** **|** **Chapter 6: RAG and Agents**")

    def test_not_footer(self):
        assert not _is_page_footer("**Embedding-based retrieval**")
        assert not _is_page_footer("Some regular text")
        assert not _is_page_footer("**Bold text without numbers**")


class TestPromotePdfHeadings:
    """Tests for _promote_pdf_headings()."""

    def test_chapter_with_title(self):
        text = "**CHAPTER 6**\n#### **RAG and Agents**"
        result = _promote_pdf_headings(text)
        assert "# Chapter 6: RAG and Agents" in result

    def test_chapter_without_title(self):
        text = "**CHAPTER 3**\nSome other content"
        result = _promote_pdf_headings(text)
        assert "# Chapter 3" in result

    def test_standalone_bold_becomes_heading(self):
        text = "Some text\n\n**Embedding-based retrieval**\n\nMore text"
        result = _promote_pdf_headings(text)
        assert "## Embedding-based retrieval" in result

    def test_multi_bold_not_promoted(self):
        # Table headers with multiple bold groups should not be promoted
        text = "**Term** **Document count** **Frequency**"
        result = _promote_pdf_headings(text)
        assert not result.startswith("##")

    def test_short_text_not_promoted(self):
        text = "before\n**A**\nafter"
        result = _promote_pdf_headings(text)
        assert "## A" not in result

    def test_page_number_not_promoted(self):
        text = "before\n**259**\nafter"
        result = _promote_pdf_headings(text)
        assert "## 259" not in result

    def test_unicode_icon_not_promoted(self):
        text = "before\n**\uf05a**\nafter"
        result = _promote_pdf_headings(text)
        assert "## \uf05a" not in result


class TestStripToc:
    """Tests for _strip_toc()."""

    def test_strips_toc_with_dotted_leaders(self):
        text = (
            "Some intro text\n\n"
            "### Table of Contents\n"
            "Chapter 1. . . . . . . . . . . . . . . . 1\n"
            "Chapter 2. . . . . . . . . . . . . . . . 15\n"
            "Chapter 3. . . . . . . . . . . . . . . . 30\n"
            "Chapter 4. . . . . . . . . . . . . . . . 45\n"
            "Chapter 5. . . . . . . . . . . . . . . . 60\n"
            "Chapter 6. . . . . . . . . . . . . . . . 75\n"
            "\n"
            "## Real Content Starts Here\n"
            "This is actual book text."
        )
        result = _strip_toc(text)
        assert "Table of Contents" not in result
        assert "Chapter 1. . ." not in result
        assert "Real Content Starts Here" in result
        assert "actual book text" in result

    def test_preserves_non_toc_content(self):
        text = "# Title\n\nThis is regular content.\n\nMore content here."
        result = _strip_toc(text)
        assert result == text


class TestCleanPdfText:
    """Tests for _clean_pdf_text() integration."""

    def test_full_pipeline(self):
        text = (
            "**CHAPTER 1**\n"
            "#### **Introduction**\n\n"
            "Some content here.\n\n"
            "**RAG** **|** **259**\n\n"
            "**Section Title**\n\n"
            "More content.\n\n"
            "**42**\n"
        )
        result = _clean_pdf_text(text)
        assert "# Chapter 1: Introduction" in result
        assert "## Section Title" in result
        assert "**|**" not in result
        assert "Some content here." in result
        assert "More content." in result

    def test_strips_bold_from_existing_headings(self):
        text = "#### **Some Title**\n\nContent"
        result = _clean_pdf_text(text)
        assert "#### Some Title" in result
        assert "**" not in result

    def test_removes_icon_lines(self):
        text = "##### \uf05a\n\nContent after icon"
        result = _clean_pdf_text(text)
        assert "\uf05a" not in result
        assert "Content after icon" in result
