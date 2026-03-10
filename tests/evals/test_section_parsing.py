"""Section parsing tests for the ingest pipeline.

Tests cover:
1. Pipeline integration (loading real PDFs, verifying section extraction)
2. Unit tests for section detection logic (no PDF required)
3. Reference comparison (sections vs saved baseline)

Integration tests are marked @pytest.mark.slow and require PDFs in docs/.
"""

import json
import re
from pathlib import Path

import pytest

from rtfm.ingest.chunker import Chunk, _clean_overlap, chunk_text
from rtfm.ingest.loader import load_file
from rtfm.config import settings

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

DOCS_DIR = Path("docs")
PROGIT_PATH = DOCS_DIR / "progit.pdf"
REFERENCE_PATH = Path("evals/section_reference.json")

# Skip helpers
progit_available = pytest.mark.skipif(
    not PROGIT_PATH.exists(),
    reason=f"PDF not found: {PROGIT_PATH}",
)
reference_available = pytest.mark.skipif(
    not REFERENCE_PATH.exists(),
    reason=f"Reference file not found: {REFERENCE_PATH}",
)

# ---------------------------------------------------------------------------
# Expected chapter names
# ---------------------------------------------------------------------------

PROGIT_CHAPTERS = [
    "Getting Started",
    "Git Basics",
    "Git Branching",
    "Git on the Server",
    "Distributed Git",
    "GitHub",
    "Git Tools",
    "Customizing Git",
    "Git and Other Systems",
    "Git Internals",
]

PROGIT_APPENDICES = [
    "Appendix A",
    "Appendix B",
    "Appendix C",
]

# ---------------------------------------------------------------------------
# Shared fixtures (session-scoped so PDFs are loaded at most once)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def progit_chunks():
    """Load and chunk progit.pdf. Skips if PDF is absent."""
    if not PROGIT_PATH.exists():
        pytest.skip(f"PDF not found: {PROGIT_PATH}")
    text, metadata = load_file(PROGIT_PATH)
    return chunk_text(text, metadata)


@pytest.fixture(scope="session")
def progit_sections(progit_chunks):
    """Unique section names from progit.pdf chunks."""
    return {c.section for c in progit_chunks}


# ===================================================================
# 1. Pipeline Integration Tests
# ===================================================================


class TestProgitIntegration:
    """Integration tests against progit.pdf."""

    @pytest.mark.slow
    @progit_available
    def test_progit_chapter_sections_present(self, progit_sections):
        """All 10 Pro Git chapters should appear as sections."""
        for chapter in PROGIT_CHAPTERS:
            matches = [s for s in progit_sections if chapter in s]
            assert matches, (
                f"Chapter '{chapter}' not found in sections. "
                f"Sample sections: {sorted(progit_sections)[:20]}"
            )

    @pytest.mark.slow
    @progit_available
    def test_progit_appendices_present(self, progit_sections):
        """All 3 appendices should appear as sections."""
        for appendix in PROGIT_APPENDICES:
            matches = [s for s in progit_sections if appendix in s]
            assert matches, (
                f"'{appendix}' not found in sections. "
                f"Sample sections: {sorted(progit_sections)[:20]}"
            )

    @pytest.mark.slow
    @progit_available
    def test_progit_total_chunks_reasonable(self, progit_chunks):
        """Chunk count should be in a reasonable range (700-1000)."""
        count = len(progit_chunks)
        assert 700 <= count <= 1000, (
            f"Expected 700-1000 chunks, got {count}"
        )

    @pytest.mark.slow
    @progit_available
    def test_section_order_matches_book_order_progit(self, progit_chunks):
        """Chapters should appear in order: chapter 1 before chapter 2, etc."""
        chapter_first_index = {}
        for i, chunk in enumerate(progit_chunks):
            for chapter in PROGIT_CHAPTERS:
                if chapter in chunk.section and chapter not in chapter_first_index:
                    chapter_first_index[chapter] = i

        found = [ch for ch in PROGIT_CHAPTERS if ch in chapter_first_index]
        assert len(found) >= 8, (
            f"Expected at least 8 chapters found in order, got {len(found)}: {found}"
        )

        for j in range(len(found) - 1):
            assert chapter_first_index[found[j]] < chapter_first_index[found[j + 1]], (
                f"Chapter '{found[j]}' (chunk {chapter_first_index[found[j]]}) "
                f"should appear before '{found[j + 1]}' "
                f"(chunk {chapter_first_index[found[j + 1]]})"
            )


class TestChunkInvariants:
    """Invariants that should hold for ALL chunked PDFs."""

    @pytest.mark.slow
    @progit_available
    def test_every_chunk_has_section_or_preamble(self, progit_chunks):
        """Every chunk must have a non-None section (empty string OK for preamble)."""
        for chunk in progit_chunks:
            assert chunk.section is not None, (
                f"Chunk {chunk.chunk_index} has None section. "
                f"Text starts: {chunk.text[:80]!r}"
            )

    @pytest.mark.slow
    @progit_available
    def test_no_empty_chunk_text(self, progit_chunks):
        """No chunk should have empty or whitespace-only text."""
        for chunk in progit_chunks:
            assert chunk.text.strip(), (
                f"Chunk {chunk.chunk_index} has empty text (section: {chunk.section!r})"
            )

    @pytest.mark.slow
    @progit_available
    def test_section_prepended_to_chunks(self, progit_chunks):
        """Chunks with a non-empty section should have '## {section}' in their text."""
        for chunk in progit_chunks:
            if chunk.section:
                assert f"## {chunk.section}" in chunk.text, (
                    f"Chunk {chunk.chunk_index} has section {chunk.section!r} "
                    f"but text does not contain '## {chunk.section}'. "
                    f"Text starts: {chunk.text[:120]!r}"
                )


# ===================================================================
# 2. Unit Tests for Section Detection (no PDF needed)
# ===================================================================


class TestHeadingDetection:
    """Unit tests for markdown heading detection in chunk_text."""

    @pytest.mark.parametrize(
        "heading, expected_section",
        [
            ("# Top Level Heading", "Top Level Heading"),
            ("## Second Level", "Second Level"),
            ("### Third Level", "Third Level"),
            ("#### Fourth Level", "Fourth Level"),
            ("##### Fifth Level", "Fifth Level"),
            ("###### Sixth Level", "Sixth Level"),
        ],
    )
    def test_heading_levels_detected(self, heading, expected_section):
        """All heading levels (#, ##, ###, ####, #####, ######) are detected."""
        text = f"Preamble paragraph.\n\n{heading}\n\nContent under heading."
        chunks = chunk_text(text, {"source_file": "test.md"})
        sections = {c.section for c in chunks}
        assert expected_section in sections, (
            f"Heading '{heading}' should yield section '{expected_section}'. "
            f"Got sections: {sections}"
        )

    @pytest.mark.parametrize(
        "heading_line, expected_name",
        [
            ("# Simple Title", "Simple Title"),
            ("## Two Hashes", "Two Hashes"),
            ("### Three Hashes", "Three Hashes"),
            ("# Chapter 1: Getting Started", "Chapter 1: Getting Started"),
            ("###### Deep Heading", "Deep Heading"),
            ("# Title With  Extra  Spaces", "Title With  Extra  Spaces"),
        ],
    )
    def test_section_name_extraction(self, heading_line, expected_name):
        """Verify lstrip('# ') correctly extracts section names."""
        text = f"Some intro.\n\n{heading_line}\n\nBody text here."
        chunks = chunk_text(text, {"source_file": "test.md"})
        heading_chunks = [c for c in chunks if c.section == expected_name]
        assert heading_chunks, (
            f"Expected section name '{expected_name}' from '{heading_line}'. "
            f"Got sections: {[c.section for c in chunks]}"
        )


class TestSectionBoundaries:
    """Unit tests for section boundary behavior."""

    def test_section_boundary_no_overlap(self):
        """When sections change, overlap does NOT carry across sections."""
        # Create two sections with enough text to trigger overlap
        section_a_body = "Word " * 200  # ~1000 chars
        section_b_body = "Data " * 50

        text = (
            f"# Section A\n\n{section_a_body}\n\n"
            f"# Section B\n\n{section_b_body}"
        )

        chunks = chunk_text(text, {"source_file": "test.md"})
        section_b_chunks = [c for c in chunks if c.section == "Section B"]

        assert section_b_chunks, "Section B should produce at least one chunk"

        for chunk in section_b_chunks:
            body = chunk.text
            if "## Section B" in body:
                body_after_heading = body.split("## Section B", 1)[1]
            else:
                body_after_heading = body

            assert "Section A" not in body_after_heading, (
                f"Section B chunk {chunk.chunk_index} contains Section A content: "
                f"{chunk.text[:200]!r}"
            )

    def test_chunks_assigned_to_correct_section(self):
        """Chunks after a heading should be assigned to that section."""
        text = (
            "# Introduction\n\n"
            "This is the intro.\n\n"
            "# Methods\n\n"
            "This is the methods section.\n\n"
            "# Results\n\n"
            "This is the results section."
        )

        chunks = chunk_text(text, {"source_file": "test.md"})

        for chunk in chunks:
            if "intro" in chunk.text.lower() and "## Introduction" in chunk.text:
                assert chunk.section == "Introduction"
            if "methods section" in chunk.text.lower():
                assert chunk.section == "Methods"
            if "results section" in chunk.text.lower():
                assert chunk.section == "Results"


class TestCleanOverlap:
    """Unit tests for _clean_overlap."""

    def test_clean_overlap_removes_previous_section(self):
        """_clean_overlap strips content before the last heading."""
        overlap = "old content from previous section\n\n## New Section\n\nNew content"
        result = _clean_overlap(overlap)
        assert result.startswith("## New Section"), (
            f"Expected overlap to start with '## New Section', got: {result!r}"
        )
        assert "old content" not in result

    def test_clean_overlap_no_heading_passthrough(self):
        """_clean_overlap returns text unchanged when no heading is present."""
        overlap = "just some regular text with no headings at all"
        result = _clean_overlap(overlap)
        assert result == overlap

    def test_clean_overlap_multiple_headings_keeps_last(self):
        """When multiple headings exist, only content from the last one is kept."""
        overlap = "## First\n\nContent A\n\n## Second\n\nContent B"
        result = _clean_overlap(overlap)
        assert result.startswith("## Second"), (
            f"Expected to start with '## Second', got: {result!r}"
        )
        assert "Content A" not in result
        assert "Content B" in result

    def test_clean_overlap_heading_at_start(self):
        """When the heading is at position 0, the full text is returned."""
        overlap = "## Only Section\n\nSome content"
        result = _clean_overlap(overlap)
        assert result == overlap

    def test_clean_overlap_various_heading_levels(self):
        """_clean_overlap handles all heading levels."""
        for level in range(1, 7):
            hashes = "#" * level
            overlap = f"old stuff\n\n{hashes} Heading\n\nNew stuff"
            result = _clean_overlap(overlap)
            assert result.startswith(f"{hashes} Heading"), (
                f"Failed for heading level {level}: {result!r}"
            )


class TestCodeCommentFalsePositives:
    """Test that code comments and non-heading '#' lines are filtered out."""

    def test_lowercase_code_comment_filtered(self):
        """Lowercase code comments starting with '# ' are filtered out.

        The heading validator rejects single-# lines where the text starts
        with a lowercase character, which catches most code comments.
        """
        text = (
            "# Real Heading\n\n"
            "Some explanation.\n\n"
            "# this is a code comment\n\n"
            "More text after the comment."
        )

        chunks = chunk_text(text, {"source_file": "test.md"})
        sections = {c.section for c in chunks}

        assert "this is a code comment" not in sections
        assert "Real Heading" in sections

    def test_indented_code_comment_filtered(self):
        """Indented lowercase code comments are also filtered out.

        Even though paragraph splitting + strip() turns '    # comment'
        into '# comment', the lowercase check catches it.
        """
        text = (
            "# Real Heading\n\n"
            "Some code example:\n\n"
            "    # indented comment\n"
            "    x = 1\n\n"
            "More text."
        )

        chunks = chunk_text(text, {"source_file": "test.md"})
        sections = {c.section for c in chunks}

        assert "indented comment" not in sections
        assert "Real Heading" in sections

    def test_inline_code_not_separated(self):
        """Code on the same line as text, without blank-line separation, is safe.

        When the code comment is part of the same paragraph (no \\n\\n before it),
        it won't be split into its own paragraph and thus won't be detected.
        """
        text = (
            "# Real Heading\n\n"
            "Some code example:\n"
            "    # inline comment\n"
            "    x = 1\n\n"
            "More text."
        )

        chunks = chunk_text(text, {"source_file": "test.md"})
        sections = {c.section for c in chunks}

        # Because the comment is in the same paragraph, it's not detected
        assert "inline comment" not in sections

    def test_shebang_filtered(self):
        """Shebang lines like #!/usr/bin/env are not treated as headings."""
        text = (
            "# Real Heading\n\n"
            "Script below:\n\n"
            "#!/usr/bin/env ruby\n"
            "puts 'hello'\n\n"
            "More text."
        )
        chunks = chunk_text(text, {"source_file": "test.md"})
        sections = {c.section for c in chunks}
        assert "/usr/bin/env ruby" not in sections
        assert "Real Heading" in sections

    def test_long_single_hash_filtered(self):
        """Long single-# lines (>40 chars) are filtered as likely code output."""
        text = (
            "# Real Heading\n\n"
            "Git output:\n\n"
            "# Please enter the commit message for your changes. Lines starting\n\n"
            "More text."
        )
        chunks = chunk_text(text, {"source_file": "test.md"})
        sections = {c.section for c in chunks}
        assert "Please enter the commit message" not in sections

    def test_fenced_code_block_filtered(self):
        """Headings inside fenced code blocks are not detected."""
        text = (
            "# Real Heading\n\n"
            "Example:\n\n"
            "```\n# Not a heading\ncode here\n```\n\n"
            "More text."
        )
        chunks = chunk_text(text, {"source_file": "test.md"})
        sections = {c.section for c in chunks}
        assert "Not a heading" not in sections
        assert "Real Heading" in sections

    def test_uppercase_code_comment_still_detected(self):
        """Uppercase short single-# lines are still detected (ambiguous case).

        Short capitalized lines like '# TODO' or '# Note' can't easily be
        distinguished from real headings, so they're kept.
        """
        text = (
            "# Real Heading\n\n"
            "Some text.\n\n"
            "# Short Note\n\n"
            "More text."
        )
        chunks = chunk_text(text, {"source_file": "test.md"})
        sections = {c.section for c in chunks}
        assert "Short Note" in sections


# ===================================================================
# 3. Reference Comparison Tests
# ===================================================================


class TestReferenceComparison:
    """Compare current pipeline output against a saved reference baseline."""

    @staticmethod
    def _load_reference() -> dict:
        """Load and return the reference JSON."""
        with open(REFERENCE_PATH, encoding="utf-8") as f:
            return json.load(f)

    @staticmethod
    def _section_counts(chunks: list[Chunk]) -> dict[str, int]:
        """Count chunks per section."""
        counts: dict[str, int] = {}
        for c in chunks:
            counts[c.section] = counts.get(c.section, 0) + 1
        return counts

    @pytest.mark.slow
    @progit_available
    @reference_available
    def test_progit_sections_match_reference(self, progit_chunks):
        """Sections from progit.pdf should match the reference baseline."""
        ref = self._load_reference()
        assert "progit.pdf" in ref, "Reference file missing 'progit.pdf' key"

        ref_sections = set(ref["progit.pdf"]["sections"])
        current_sections = {c.section for c in progit_chunks}

        missing = ref_sections - current_sections
        added = current_sections - ref_sections

        for chapter in PROGIT_CHAPTERS:
            chapter_in_ref = any(chapter in s for s in ref_sections)
            chapter_in_current = any(chapter in s for s in current_sections)
            if chapter_in_ref:
                assert chapter_in_current, (
                    f"Chapter '{chapter}' was in reference but missing from current output"
                )

        if missing or added:
            drift_pct = (len(missing) + len(added)) / max(len(ref_sections), 1) * 100
            assert drift_pct < 20, (
                f"Section drift too large ({drift_pct:.1f}%). "
                f"Missing: {sorted(missing)[:10]}, Added: {sorted(added)[:10]}"
            )

    @pytest.mark.slow
    @progit_available
    @reference_available
    def test_section_chunk_counts_stable(self, progit_chunks):
        """Per-section chunk counts should not drift more than 10% from reference."""
        ref = self._load_reference()
        if "progit.pdf" not in ref:
            pytest.skip("Reference file missing 'progit.pdf'")

        ref_counts = ref["progit.pdf"]["section_chunk_counts"]
        current_counts = self._section_counts(progit_chunks)

        drifted = []
        for section, ref_count in ref_counts.items():
            cur_count = current_counts.get(section, 0)
            if ref_count == 0:
                continue
            drift = abs(cur_count - ref_count) / ref_count
            if drift > 0.10:
                drifted.append(
                    f"  {section!r}: ref={ref_count}, current={cur_count} "
                    f"(drift={drift:.0%})"
                )

        assert not drifted, (
            f"{len(drifted)} sections drifted >10% from reference:\n"
            + "\n".join(drifted[:15])
        )
