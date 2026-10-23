import os
import re
from io import BytesIO
from typing import List, Tuple, BinaryIO, Union
from pydantic import BaseModel
from pypdf import PdfReader
from app.core.logging import logger

class ParsedPage(BaseModel):
    """Represents text extracted from a single page or logical section."""
    page_number: int
    text: str
    total_pages: int

class DocumentParser:
    """
    Multi-format document ingestion parser for campus files.
    Supports PDF, Markdown (.md), and Plain Text (.txt) formats,
    preserving page boundaries and structural sections for precise citation.
    """

    @classmethod
    def parse_file(cls, filename: str, content: Union[bytes, BinaryIO]) -> List[ParsedPage]:
        """
        Parses document bytes into a sequence of ParsedPages based on file extension.
        """
        ext = os.path.splitext(filename)[1].lower()
        if isinstance(content, bytes):
            stream = BytesIO(content)
        else:
            stream = content

        logger.info(f"Parsing document '{filename}' (type: {ext})")

        if ext == ".pdf":
            return cls._parse_pdf(stream, filename)
        elif ext in [".md", ".markdown"]:
            text = stream.read().decode("utf-8", errors="replace")
            return cls._parse_markdown(text, filename)
        elif ext in [".txt", ".text"]:
            text = stream.read().decode("utf-8", errors="replace")
            return cls._parse_text(text, filename)
        else:
            # Fallback text decoding for any other text-based campus document
            text = stream.read().decode("utf-8", errors="replace")
            return cls._parse_text(text, filename)

    @staticmethod
    def _parse_pdf(stream: BinaryIO, filename: str) -> List[ParsedPage]:
        """Extracts text per page from PDF using pypdf."""
        reader = PdfReader(stream)
        total_pages = len(reader.pages)
        pages: List[ParsedPage] = []

        for idx, page in enumerate(reader.pages, start=1):
            extracted = page.extract_text() or ""
            # Clean up whitespace while preserving paragraph breaks
            cleaned = re.sub(r"[ \t]+", " ", extracted).strip()
            if cleaned:
                pages.append(ParsedPage(page_number=idx, text=cleaned, total_pages=total_pages))
            else:
                logger.warning(f"Empty or non-extractable text on page {idx} of PDF '{filename}'")

        if not pages:
            # Create a placeholder if PDF was completely scanned/empty
            pages.append(ParsedPage(page_number=1, text="[Scanned or empty document]", total_pages=1))

        return pages

    @staticmethod
    def _parse_markdown(text: str, filename: str) -> List[ParsedPage]:
        """
        Splits Markdown files into logical pages based on structural dividers:
        horizontal rules (---), top-level headers (# ), or ~1500 char blocks.
        """
        # Check if author used explicit horizontal rule page breaks
        divider_pattern = r"(?m)^(?:\*{3,}|-{3,}|_{3,})$"
        parts = [p.strip() for p in re.split(divider_pattern, text) if p.strip()]

        if len(parts) > 1:
            total_pages = len(parts)
            return [ParsedPage(page_number=i, text=part, total_pages=total_pages) for i, part in enumerate(parts, start=1)]

        # Otherwise, partition by major section headers (# Title)
        header_parts = re.split(r"(?m)(?=^# )", text)
        header_parts = [hp.strip() for hp in header_parts if hp.strip()]
        if len(header_parts) > 1:
            total_pages = len(header_parts)
            return [ParsedPage(page_number=i, text=hp, total_pages=total_pages) for i, hp in enumerate(header_parts, start=1)]

        # Fallback: Partition by paragraph chunks of ~1500 characters
        return DocumentParser._split_text_into_pages(text, page_char_limit=1500)

    @staticmethod
    def _parse_text(text: str, filename: str) -> List[ParsedPage]:
        """Splits raw text files into readable pages (~1500 characters per page)."""
        return DocumentParser._split_text_into_pages(text, page_char_limit=1500)

    @staticmethod
    def _split_text_into_pages(text: str, page_char_limit: int = 1500) -> List[ParsedPage]:
        """Helper to break contiguous text into logical pages along paragraph boundaries."""
        paragraphs = text.split("\n\n")
        pages_content = []
        current_page = []
        current_len = 0

        for para in paragraphs:
            para_len = len(para)
            if current_len + para_len > page_char_limit and current_page:
                pages_content.append("\n\n".join(current_page))
                current_page = [para]
                current_len = para_len
            else:
                current_page.append(para)
                current_len += para_len + 2

        if current_page:
            pages_content.append("\n\n".join(current_page))

        if not pages_content:
            pages_content = [text.strip() or ""]

        total_pages = len(pages_content)
        return [
            ParsedPage(page_number=i, text=content, total_pages=total_pages)
            for i, content in enumerate(pages_content, start=1)
        ]
