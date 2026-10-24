import os
import time
from pathlib import Path
from typing import List, Optional
from app.core.config import settings
from app.core.logging import logger
from app.models.schemas import (
    QueryRequest,
    QueryResponse,
    IngestionResult,
    SourceCitation,
)
from app.services.document_parser import DocumentParser
from app.services.chunker import RecursiveTextChunker
from app.services.embedding_service import embedding_service
from app.services.vector_store import vector_store
from app.services.guardrails import GuardrailEngine, EXACT_REFUSAL_PHRASE
from app.services.llm_provider import LLMFactory

class RAGEngine:
    """
    Core RAG Pipeline Engine.
    Orchestrates ingestion, embedding generation, dense vector retrieval,
    cosine threshold filtering, prompt engineering, and guardrail validation.
    """

    def __init__(self):
        self.chunker = RecursiveTextChunker(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP,
        )
        self.embedding_service = embedding_service
        self.vector_store = vector_store

    def query(self, request: QueryRequest) -> QueryResponse:
        """Executes a full RAG query with strict source citations and guardrails."""
        start_time = time.perf_counter()
        query_text = request.query.strip()
        top_k = request.top_k or settings.TOP_K
        threshold = (
            request.similarity_threshold
            if request.similarity_threshold is not None
            else settings.SIMILARITY_THRESHOLD
        )

        logger.info(f"Processing query: '{query_text}' (top_k={top_k}, threshold={threshold})")

        # 0. Pre-retrieval Scope Gate Verification (Refuse Out-of-Scope queries early)
        scope_check = GuardrailEngine.check_query_scope(query_text)
        if not scope_check["is_in_scope"]:
            elapsed_ms = round((time.perf_counter() - start_time) * 1000, 2)
            logger.info(
                f"Query rejected by Learned Scope Gate (OOS score={scope_check['oos_score']:.4f} >= threshold={scope_check['threshold']:.4f}). "
                "Returning immediate deterministic refusal without retrieval."
            )
            return QueryResponse(
                query=query_text,
                answer=EXACT_REFUSAL_PHRASE,
                sources=[],
                confidence_score=0.0,
                guardrail_status="REFUSAL_NOT_IN_CONTEXT",
                latency_ms=elapsed_ms,
                model_used=f"Learned Scope Gate (Pre-retrieval refusal, score={scope_check['oos_score']:.3f})",
            )

        # 1. Embed query vector
        query_embedding = self.embedding_service.embed_query(query_text)

        # 2. Retrieve candidate chunks with cosine similarity filtering
        retrieved_chunks = self.vector_store.similarity_search(
            query_embedding=query_embedding,
            top_k=top_k,
            threshold=threshold,
            filter_doc=request.filter_doc,
        )

        # 3. Guardrail check: if no chunks matched threshold, refuse immediately
        if not retrieved_chunks:
            elapsed_ms = round((time.perf_counter() - start_time) * 1000, 2)
            logger.info("No documents passed similarity threshold. Returning strict guardrail refusal.")
            return QueryResponse(
                query=query_text,
                answer=EXACT_REFUSAL_PHRASE,
                sources=[],
                confidence_score=0.0,
                guardrail_status="REFUSAL_NOT_IN_CONTEXT",
                latency_ms=elapsed_ms,
                model_used="Guardrail Filter (Zero-match cutoff)",
            )

        # 4. Construct grounded prompts
        system_prompt = GuardrailEngine.construct_system_prompt()
        user_prompt = GuardrailEngine.format_context_prompt(query_text, retrieved_chunks)

        # 5. Invoke LLM generation
        llm = LLMFactory.get_provider()
        model_name = llm.get_model_name()

        try:
            raw_answer = llm.generate(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                retrieved_chunks=retrieved_chunks,
            )
        except Exception as e:
            logger.error(f"Error during LLM generation ({e}). Falling back to deterministic extraction.")
            from app.services.llm_provider import DeterministicGroundedProvider
            fallback_llm = DeterministicGroundedProvider()
            raw_answer = fallback_llm.generate(system_prompt, user_prompt, retrieved_chunks)
            model_name = "Fallback Extractive Synthesizer"

        # 6. Post-generation guardrail evaluation & citation verification
        eval_result = GuardrailEngine.evaluate_response(
            query=query_text,
            raw_answer=raw_answer,
            retrieved_chunks=retrieved_chunks,
        )

        elapsed_ms = round((time.perf_counter() - start_time) * 1000, 2)

        return QueryResponse(
            query=query_text,
            answer=eval_result["final_answer"],
            sources=eval_result["sources"],
            confidence_score=eval_result["confidence_score"],
            guardrail_status=eval_result["guardrail_status"],
            latency_ms=elapsed_ms,
            model_used=model_name,
        )

    def ingest_file(self, filename: str, content: bytes) -> IngestionResult:
        """Ingests a single campus document, performs chunking, generates embeddings, and indexes vectors."""
        logger.info(f"Starting ingestion pipeline for '{filename}' ({len(content)} bytes)")

        # 1. Parse document into pages
        pages = DocumentParser.parse_file(filename, content)
        if not pages:
            return IngestionResult(
                filename=filename,
                chunks_created=0,
                pages_detected=0,
                status="FAILED",
                message="No extractable text found in file.",
            )

        # 2. Semantic recursive chunking
        chunks = self.chunker.chunk_document(filename, pages)
        if not chunks:
            return IngestionResult(
                filename=filename,
                chunks_created=0,
                pages_detected=len(pages),
                status="FAILED",
                message="Document was parsed but produced 0 text chunks.",
            )

        # 3. Generate embeddings
        texts = [c.content for c in chunks]
        embeddings = self.embedding_service.embed_documents(texts)

        # 4. Store in vector store
        upserted = self.vector_store.add_chunks(chunks, embeddings)

        return IngestionResult(
            filename=filename,
            chunks_created=upserted,
            pages_detected=len(pages),
            status="SUCCESS",
            message=f"Successfully indexed {upserted} chunks across {len(pages)} pages.",
        )

    def ingest_sample_docs(self, sample_dir: Optional[str] = None) -> List[IngestionResult]:
        """Loads and indexes all sample campus documents from data/sample_docs/."""
        target_dir = Path(sample_dir or settings.SAMPLE_DOCS_DIRECTORY)
        results: List[IngestionResult] = []

        if not target_dir.exists():
            logger.warning(f"Sample documents directory does not exist: {target_dir}")
            return results

        for file_path in target_dir.glob("*.*"):
            if file_path.suffix.lower() in [".pdf", ".md", ".txt"]:
                with open(file_path, "rb") as f:
                    content = f.read()
                res = self.ingest_file(file_path.name, content)
                results.append(res)

        return results

# Singleton instance
rag_engine = RAGEngine()
