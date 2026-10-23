from typing import List, Optional
from fastapi import APIRouter, UploadFile, File, HTTPException, status
from app.models.schemas import (
    QueryRequest,
    QueryResponse,
    IngestionResult,
    DocumentInfo,
    HealthResponse,
    StatsResponse,
)
from app.services.rag_engine import rag_engine
from app.services.vector_store import vector_store
from app.services.llm_provider import LLMFactory
from app.core.config import settings
from app.core.logging import logger

router = APIRouter()

ALLOWED_EXTENSIONS = {".pdf", ".md", ".markdown", ".txt", ".text"}

@router.post(
    "/query",
    response_model=QueryResponse,
    summary="Query Campus Knowledge Base",
    description="Executes a dense RAG retrieval with cosine similarity thresholding, returning grounded answers with verifiable source citations.",
)
async def query_knowledge_base(request: QueryRequest) -> QueryResponse:
    try:
        response = rag_engine.query(request)
        return response
    except Exception as e:
        logger.error(f"Error querying knowledge base: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred while processing the RAG query: {str(e)}",
        )

@router.post(
    "/upload",
    response_model=IngestionResult,
    summary="Upload & Ingest Campus Document",
    description="Accepts multi-format campus documents (PDF, Markdown, TXT), performs recursive chunking, generates embeddings, and indexes vectors into ChromaDB.",
)
async def upload_document(file: UploadFile = File(...)) -> IngestionResult:
    filename = file.filename or "uploaded_document.txt"
    ext = "." + filename.split(".")[-1].lower() if "." in filename else ""

    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file format '{ext}'. Allowed formats: {list(ALLOWED_EXTENSIONS)}",
        )

    try:
        content = await file.read()
        if len(content) == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Uploaded file is empty.",
            )

        result = rag_engine.ingest_file(filename, content)
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error ingesting document '{filename}': {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to ingest document '{filename}': {str(e)}",
        )

@router.post(
    "/ingest-sample-docs",
    response_model=List[IngestionResult],
    summary="Ingest Sample Campus Documents",
    description="Indexes all pre-bundled German university documents (FPSO exam regulations, library rules, AI curriculum, enrollment guidelines).",
)
async def ingest_sample_documents() -> List[IngestionResult]:
    try:
        results = rag_engine.ingest_sample_docs()
        return results
    except Exception as e:
        logger.error(f"Error ingesting sample documents: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to ingest sample documents: {str(e)}",
        )

@router.get(
    "/documents",
    response_model=List[DocumentInfo],
    summary="List Indexed Documents",
    description="Returns metadata for all campus documents currently indexed in the vector store.",
)
async def list_indexed_documents() -> List[DocumentInfo]:
    try:
        return vector_store.list_documents()
    except Exception as e:
        logger.error(f"Error listing documents: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve document list.",
        )

@router.delete(
    "/documents/{document_name}",
    summary="Delete Document from Knowledge Base",
    description="Removes all vector embeddings and chunks associated with the specified document.",
)
async def delete_document(document_name: str):
    success = vector_store.delete_document(document_name)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document '{document_name}' not found in vector store.",
        )
    return {"message": f"Successfully deleted '{document_name}' from knowledge base."}

@router.get(
    "/health",
    response_model=HealthResponse,
    summary="System Health & Readiness Check",
    description="Returns system operational status, collection size, active embedding model, and LLM provider.",
)
async def health_check() -> HealthResponse:
    docs = vector_store.list_documents()
    total_chunks = sum(d.chunks_count for d in docs)
    llm = LLMFactory.get_provider()

    return HealthResponse(
        status="healthy",
        version="1.0.0",
        documents_indexed=len(docs),
        total_chunks=total_chunks,
        embedding_provider=settings.EMBEDDING_PROVIDER,
        embedding_model=settings.EMBEDDING_MODEL_NAME,
        llm_provider=settings.LLM_PROVIDER,
        active_llm_model=llm.get_model_name(),
    )

@router.get(
    "/stats",
    response_model=StatsResponse,
    summary="Vector Store Statistics",
    description="Returns collection details, storage path, and breakdown of indexed documents.",
)
async def get_vector_stats() -> StatsResponse:
    stats = vector_store.get_stats()
    docs = [
        DocumentInfo(
            document_name=d["document_name"],
            chunks_count=d["chunks_count"],
            pages_count=d["pages_count"],
            ingestion_timestamp=d["ingestion_timestamp"],
        )
        for d in stats["documents"]
    ]
    return StatsResponse(
        collection_name=stats["collection_name"],
        total_chunks=stats["total_chunks"],
        unique_documents=stats["unique_documents"],
        documents=docs,
        persist_directory=stats["persist_directory"],
    )
