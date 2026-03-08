"""Unit tests for rtfm.ingest.loader module."""

from pathlib import Path

import pytest

from rtfm.ingest.loader import discover_files, load_file


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
