from datetime import datetime, timezone
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field

class SourceCitation(BaseModel):
    """Specific verifiable source citation for an answer."""
    document_name: str = Field(..., description="Filename of the source campus document")
    page_number: int = Field(..., description="Exact page or section number in document (1-indexed)")
    similarity_score: float = Field(..., description="Cosine similarity score (0.0 to 1.0)")
    excerpt: str = Field(..., description="Direct verbatim text excerpt supporting the answer")
    chunk_id: str = Field(..., description="Unique chunk identifier in vector store")

class QueryRequest(BaseModel):
    """Payload for RAG query request."""
    query: str = Field(..., min_length=2, max_length=1000, description="User question regarding campus matters")
    top_k: Optional[int] = Field(default=4, ge=1, le=10, description="Maximum number of chunks to retrieve")
    filter_doc: Optional[str] = Field(default=None, description="Optional filter by exact document name")
    similarity_threshold: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="Override cosine similarity cutoff")

class QueryResponse(BaseModel):
    """Production response schema with strict citations and guardrails."""
    query: str
    answer: str = Field(..., description="Grounded response or explicit refusal")
    sources: List[SourceCitation] = Field(default_factory=list, description="Verified source citations")
    confidence_score: float = Field(..., description="Calibrated confidence score between 0.0 and 1.0")
    guardrail_status: str = Field(..., description="Status: 'PASS_GROUNDED', 'REFUSAL_NOT_IN_CONTEXT', or 'PARTIAL_RELEVANCE'")
    latency_ms: float = Field(..., description="Total end-to-end RAG latency in milliseconds")
    model_used: str = Field(..., description="LLM model and provider utilized")
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

class DocumentMetadata(BaseModel):
    """Rich metadata preserved during chunking and vector indexing."""
    document_name: str
    page_number: int
    chunk_index: int
    total_pages: int
    file_type: str
    timestamp: str

class DocumentChunk(BaseModel):
    """Atomic text chunk with embedding metadata."""
    chunk_id: str
    document_name: str
    content: str
    page_number: int
    chunk_index: int
    metadata: Dict[str, Any]

class IngestionResult(BaseModel):
    """Summary of document ingestion and indexing pipeline."""
    filename: str
    chunks_created: int
    pages_detected: int
    status: str
    message: str

class DocumentInfo(BaseModel):
    """Summary information for an indexed campus document."""
    document_name: str
    chunks_count: int
    pages_count: int
    ingestion_timestamp: str

class HealthResponse(BaseModel):
    """System health check and operational telemetry."""
    status: str
    version: str
    documents_indexed: int
    total_chunks: int
    embedding_provider: str
    embedding_model: str
    llm_provider: str
    active_llm_model: str

class StatsResponse(BaseModel):
    """Detailed vector collection statistics."""
    collection_name: str
    total_chunks: int
    unique_documents: int
    documents: List[DocumentInfo]
    persist_directory: str
