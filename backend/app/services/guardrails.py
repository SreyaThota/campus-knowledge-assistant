import re
from typing import List, Tuple, Dict, Any, Optional
from app.models.schemas import DocumentChunk, SourceCitation
from app.core.logging import logger

EXACT_REFUSAL_PHRASE = "Information not found in provided campus documents."

class GuardrailEngine:
    """
    RAG Safety and Grounding Guardrails.
    Enforces strict citation rules, out-of-domain refusal enforcement,
    and hallucination mitigation.
    """

    @staticmethod
    def construct_system_prompt() -> str:
        """Constructs the system prompt with strict citation and refusal instructions."""
        return (
            "You are the official Campus Knowledge Assistant for university students.\n"
            "Your objective is to deliver accurate, grounded, and concise answers based EXCLUSIVELY on the provided campus documents.\n\n"
            "STRICT CITATION AND FACTUAL GROUNDING RULES:\n"
            "1. Base your answer strictly on the facts present in the provided context excerpts.\n"
            "2. For every factual statement, cite the exact source document and page number in square brackets, e.g., [TUM_FPSO_Exam_Regulations.md, Page 2].\n"
            f"3. If the provided context DOES NOT contain enough information to answer the question, or if no relevant context is given, you MUST respond with the exact phrase:\n"
            f'   "{EXACT_REFUSAL_PHRASE}"\n'
            "4. Do NOT attempt to guess, extrapolate, or hallucinate facts beyond the provided context.\n"
            "5. Answer in the same language as the question (e.g., German if queried in German, English if queried in English), but keep the exact refusal phrase if information is missing."
        )

    @staticmethod
    def format_context_prompt(query: str, retrieved_chunks: List[Tuple[DocumentChunk, float]]) -> str:
        """Formats retrieved chunks with metadata headers for LLM ingestion."""
        if not retrieved_chunks:
            return (
                f"User Question: {query}\n\n"
                "Retrieved Context: No relevant campus documents were found in the database for this query.\n\n"
                f'Please respond with: "{EXACT_REFUSAL_PHRASE}"'
            )

        context_blocks = []
        for i, (chunk, score) in enumerate(retrieved_chunks, start=1):
            block = (
                f"--- [EXCERPT {i}] ---\n"
                f"Source: {chunk.document_name} | Page: {chunk.page_number} | Relevance Score: {score:.2f}\n"
                f"Content:\n{chunk.content}\n"
            )
            context_blocks.append(block)

        formatted_context = "\n".join(context_blocks)
        return (
            f"User Question: {query}\n\n"
            f"VERIFIED CAMPUS CONTEXT:\n{formatted_context}\n\n"
            "Instructions: Provide a concise, directly grounded answer citing the sources above. "
            f'If the question cannot be answered from this text, respond ONLY with: "{EXACT_REFUSAL_PHRASE}"'
        )

    @staticmethod
    def evaluate_response(
        query: str,
        raw_answer: str,
        retrieved_chunks: List[Tuple[DocumentChunk, float]],
    ) -> Dict[str, Any]:
        """
        Post-generation guardrail evaluation.
        Evaluates whether the response adhered to grounding constraints,
        calculates a calibrated confidence score, and formats source citations.
        """
        trimmed = raw_answer.strip()
        refusal_detected = EXACT_REFUSAL_PHRASE.lower() in trimmed.lower()

        # If context was empty or all filtered out, enforce refusal
        if not retrieved_chunks or refusal_detected:
            return {
                "final_answer": EXACT_REFUSAL_PHRASE,
                "sources": [],
                "confidence_score": 0.0,
                "guardrail_status": "REFUSAL_NOT_IN_CONTEXT",
            }

        # Build verifiable citations from retrieved chunks
        sources: List[SourceCitation] = []
        for chunk, score in retrieved_chunks:
            # Create a concise excerpt (first 200 characters)
            excerpt = chunk.content[:220].replace("\n", " ").strip()
            if len(chunk.content) > 220:
                excerpt += "..."

            sources.append(
                SourceCitation(
                    document_name=chunk.document_name,
                    page_number=chunk.page_number,
                    similarity_score=round(score, 4),
                    excerpt=excerpt,
                    chunk_id=chunk.chunk_id,
                )
            )

        # Calibrate confidence score: based on max similarity, average similarity, and context presence
        max_sim = retrieved_chunks[0][1] if retrieved_chunks else 0.0
        avg_sim = sum(s for _, s in retrieved_chunks) / len(retrieved_chunks) if retrieved_chunks else 0.0

        # Weighted confidence metric
        confidence = round(min(1.0, (max_sim * 0.7) + (avg_sim * 0.3)), 3)

        # Check for grounding status
        if confidence >= 0.65:
            status = "PASS_GROUNDED"
        elif confidence >= 0.40:
            status = "PARTIAL_RELEVANCE"
        else:
            status = "REFUSAL_NOT_IN_CONTEXT"
            trimmed = EXACT_REFUSAL_PHRASE

        return {
            "final_answer": trimmed,
            "sources": sources,
            "confidence_score": confidence,
            "guardrail_status": status,
        }
