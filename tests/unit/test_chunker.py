"""Unit tests for rtfm.ingest.chunker module."""

from rtfm.ingest.chunker import Chunk, chunk_text


class TestChunkText:
    """Tests for chunk_text()."""

    def test_chunk_single_paragraph(self):
        text = "A short paragraph."
        chunks = chunk_text(text, metadata={}, chunk_size=50, chunk_overlap=0)

        assert len(chunks) == 1
        assert chunks[0].text == "A short paragraph."

    def test_chunk_multiple_paragraphs_fit(self):
        text = "First paragraph.\n\nSecond paragraph."
        # chunk_size=50 tokens => 200 chars, plenty of room
        chunks = chunk_text(text, metadata={}, chunk_size=50, chunk_overlap=0)

        assert len(chunks) == 1
        assert "First paragraph." in chunks[0].text
        assert "Second paragraph." in chunks[0].text

    def test_chunk_exceeds_size(self):
        # chunk_size=5 tokens => 20 chars max per chunk
        paragraphs = [f"Paragraph number {i} here." for i in range(10)]
        text = "\n\n".join(paragraphs)

        chunks = chunk_text(text, metadata={}, chunk_size=5, chunk_overlap=0)

        assert len(chunks) > 1

    def test_chunk_overlap(self):
        # chunk_size=10 tokens => 40 chars, overlap=5 => 20 chars overlap
        p1 = "Alpha bravo charlie delta echo."  # 31 chars
        p2 = "Foxtrot golf hotel india juliet."  # 31 chars
        p3 = "Kilo lima mike november oscar."  # 30 chars
        text = f"{p1}\n\n{p2}\n\n{p3}"

        chunks = chunk_text(text, metadata={}, chunk_size=10, chunk_overlap=5)

        assert len(chunks) >= 2
        # The overlap means the tail of chunk N appears at the start of chunk N+1
        # Use a shorter slice to avoid boundary whitespace mismatches
        tail_of_first = chunks[0].text[-15:]
        assert tail_of_first.strip() in chunks[1].text

    def test_chunk_heading_tracking(self):
        text = "# Introduction\n\nSome intro text.\n\n## Details\n\nDetail content here."
        chunks = chunk_text(text, metadata={}, chunk_size=50, chunk_overlap=0)

        # With a large chunk_size, everything fits in one chunk.
        # The last heading seen should be the section.
        assert chunks[-1].section == "Details"

    def test_chunk_indices_sequential(self):
        paragraphs = [f"Paragraph {i} with some text." for i in range(8)]
        text = "\n\n".join(paragraphs)

        chunks = chunk_text(text, metadata={}, chunk_size=10, chunk_overlap=0)

        indices = [c.chunk_index for c in chunks]
        assert indices == list(range(len(chunks)))

    def test_chunk_metadata_preserved(self):
        text = "Para one content.\n\nPara two content.\n\nPara three content."
        meta = {"source_file": "test.md", "author": "tester"}

        chunks = chunk_text(text, metadata=meta, chunk_size=10, chunk_overlap=0)

        assert len(chunks) >= 1
        for chunk in chunks:
            assert chunk.metadata is meta
            assert chunk.metadata["source_file"] == "test.md"
            assert chunk.metadata["author"] == "tester"

    def test_chunk_empty_text(self):
        chunks = chunk_text("", metadata={}, chunk_size=50, chunk_overlap=0)

        assert chunks == []

    def test_chunk_custom_sizes(self):
        # Verify that custom chunk_size and overlap are respected over defaults
        word = "abcdefghij"  # 10 chars
        # 5 paragraphs of 10 chars each; chunk_size=3 tokens => 12 chars max
        text = "\n\n".join([word] * 5)

        chunks_small = chunk_text(text, metadata={}, chunk_size=3, chunk_overlap=0)
        chunks_large = chunk_text(text, metadata={}, chunk_size=500, chunk_overlap=0)

        assert len(chunks_small) > len(chunks_large)
        assert len(chunks_large) == 1
