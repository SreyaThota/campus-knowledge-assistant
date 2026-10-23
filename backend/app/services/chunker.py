from datetime import datetime, timezone
from typing import List, Optional, Dict, Any
from app.models.schemas import DocumentChunk
from app.services.document_parser import ParsedPage
from app.core.logging import logger

class RecursiveTextChunker:
    """
    Intelligent Recursive Character Chunking with overlap.
    Preserves semantic paragraph boundaries, markdown headers, and sentence integrity.
    Maintains exact document name and page number attribution for each chunk.
    """

    def __init__(
        self,
        chunk_size: int = 750,
        chunk_overlap: int = 150,
        separators: Optional[List[str]] = None,
    ):
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be strictly smaller than chunk_size")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or [
            "\n\n### ",
            "\n\n## ",
            "\n\n# ",
            "\n\n",
            "\n",
            ". ",
            "? ",
            "! ",
            "; ",
            " ",
            "",
        ]

    def chunk_document(
        self,
        document_name: str,
        pages: List[ParsedPage],
    ) -> List[DocumentChunk]:
        """
        Chunks an entire document page by page to guarantee that no chunk
        spans across non-contiguous page boundaries without tracking.
        """
        all_chunks: List[DocumentChunk] = []
        global_chunk_idx = 0
        timestamp = datetime.now(timezone.utc).isoformat()

        for page in pages:
            raw_splits = self._split_text(page.text, self.separators)
            merged_splits = self._merge_splits(raw_splits)

            for split_text in merged_splits:
                cleaned_text = split_text.strip()
                if not cleaned_text:
                    continue

                chunk_id = f"{document_name}_p{page.page_number}_c{global_chunk_idx}"
                metadata = {
                    "document_name": document_name,
                    "page_number": page.page_number,
                    "chunk_index": global_chunk_idx,
                    "total_pages": page.total_pages,
                    "char_count": len(cleaned_text),
                    "timestamp": timestamp,
                }

                chunk = DocumentChunk(
                    chunk_id=chunk_id,
                    document_name=document_name,
                    content=cleaned_text,
                    page_number=page.page_number,
                    chunk_index=global_chunk_idx,
                    metadata=metadata,
                )
                all_chunks.append(chunk)
                global_chunk_idx += 1

        logger.info(
            f"Chunked document '{document_name}' into {len(all_chunks)} semantic chunks "
            f"(chunk_size={self.chunk_size}, overlap={self.chunk_overlap})"
        )
        return all_chunks

    def _split_text(self, text: str, separators: List[str]) -> List[str]:
        """Recursively splits text using the highest-priority separator available."""
        final_chunks: List[str] = []
        if not separators:
            return [text]

        separator = separators[0]
        remaining_separators = separators[1:]

        if separator == "":
            return list(text)

        splits = text.split(separator)
        for s in splits:
            if not s:
                continue
            if len(s) <= self.chunk_size:
                final_chunks.append(s)
            else:
                if remaining_separators:
                    sub_splits = self._split_text(s, remaining_separators)
                    final_chunks.extend(sub_splits)
                else:
                    # Hard cut if no more separators exist
                    for i in range(0, len(s), self.chunk_size - self.chunk_overlap):
                        final_chunks.append(s[i:i + self.chunk_size])

        return final_chunks

    def _merge_splits(self, splits: List[str]) -> List[str]:
        """
        Combines small split fragments into chunks of size <= chunk_size,
        with an overlap of chunk_overlap between consecutive chunks.
        """
        chunks: List[str] = []
        current_doc: List[str] = []
        current_len = 0

        for split in splits:
            split_len = len(split)
            if current_len + split_len > self.chunk_size and current_doc:
                merged = " ".join(current_doc).strip()
                chunks.append(merged)

                # Keep overlap from the end of current_doc for the next chunk
                overlap_doc = []
                overlap_len = 0
                for part in reversed(current_doc):
                    if overlap_len + len(part) <= self.chunk_overlap:
                        overlap_doc.insert(0, part)
                        overlap_len += len(part)
                    else:
                        break

                current_doc = overlap_doc
                current_len = sum(len(p) for p in current_doc)

            current_doc.append(split)
            current_len += split_len

        if current_doc:
            merged = " ".join(current_doc).strip()
            if merged:
                chunks.append(merged)

        return chunks
