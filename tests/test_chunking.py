import pytest
from app.services.chunker import RecursiveTextChunker
from app.services.document_parser import ParsedPage, DocumentParser

def test_recursive_chunker_basic():
    """Verify that chunker splits text to within chunk_size limits."""
    chunker = RecursiveTextChunker(chunk_size=100, chunk_overlap=20)
    pages = [
        ParsedPage(
            page_number=1,
            text="Sentence one. Sentence two. Sentence three. Sentence four. Sentence five. Sentence six.",
            total_pages=1,
        )
    ]
    chunks = chunker.chunk_document("test_doc.md", pages)
    assert len(chunks) > 0
    for chunk in chunks:
        assert len(chunk.content) <= 120  # Allows slight margin for word boundary preservation
        assert chunk.document_name == "test_doc.md"
        assert chunk.page_number == 1
        assert "document_name" in chunk.metadata

def test_chunker_overlap_preservation():
    """Verify that adjacent chunks share overlapping text context."""
    chunker = RecursiveTextChunker(chunk_size=80, chunk_overlap=30)
    text = (
        "Technical University of Munich examination rules require medical certificates "
        "to be submitted within three business days after the exam date."
    )
    pages = [ParsedPage(page_number=1, text=text, total_pages=1)]
    chunks = chunker.chunk_document("medical_rules.txt", pages)

    assert len(chunks) >= 2
    # Verify some token/character overlap between chunk 0 and chunk 1
    chunk0_words = set(chunks[0].content.split()[-3:])
    chunk1_words = set(chunks[1].content.split()[:5])
    assert len(chunk0_words & chunk1_words) > 0

def test_chunker_multi_page_attribution():
    """Verify that chunks accurately retain their respective source page numbers."""
    chunker = RecursiveTextChunker(chunk_size=150, chunk_overlap=20)
    pages = [
        ParsedPage(page_number=1, text="Page 1: General provisions and scope of the program.", total_pages=2),
        ParsedPage(page_number=2, text="Page 2: Master thesis requirements and submission deadlines.", total_pages=2),
    ]
    chunks = chunker.chunk_document("curriculum.md", pages)
    page_numbers = {c.page_number for c in chunks}
    assert 1 in page_numbers
    assert 2 in page_numbers

def test_document_parser_markdown_headers():
    """Verify that markdown parser breaks sections into logical pages."""
    md_text = (
        "# Section 1: Overview\nIntroduction to the degree.\n\n"
        "# Section 2: ECTS Requirements\nTotal 120 ECTS required."
    )
    pages = DocumentParser._parse_markdown(md_text, "syllabus.md")
    assert len(pages) == 2
    assert pages[0].page_number == 1
    assert "Overview" in pages[0].text
    assert pages[1].page_number == 2
    assert "ECTS Requirements" in pages[1].text

def test_chunker_empty_input():
    """Verify that empty pages produce zero chunks gracefully."""
    chunker = RecursiveTextChunker(chunk_size=100, chunk_overlap=20)
    pages = [ParsedPage(page_number=1, text="", total_pages=1)]
    chunks = chunker.chunk_document("empty.txt", pages)
    assert len(chunks) == 0
