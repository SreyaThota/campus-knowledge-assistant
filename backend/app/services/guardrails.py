import re
import sys
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional
from app.models.schemas import DocumentChunk, SourceCitation
from app.core.config import settings, PROJECT_ROOT
from app.core.logging import logger

# Ensure project root is on sys.path for gate module imports
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

EXACT_REFUSAL_PHRASE = "Information not found in provided campus documents."

class GuardrailEngine:
    """
    RAG Safety, Grounding Guardrails, and Learned Scope Gate.
    Enforces strict citation rules, out-of-domain refusal enforcement,
    early scope classification, and hallucination mitigation.
    """

    _scope_gate_predictor = None
    _scope_gate_load_attempted = False

    @classmethod
    def get_scope_gate_predictor(cls):
        """Returns the loaded ScopeGatePredictor singleton or None if disabled/unavailable."""
        if not settings.ENABLE_SCOPE_GATE:
            return None

        if cls._scope_gate_predictor is None and not cls._scope_gate_load_attempted:
            cls._scope_gate_load_attempted = True
            ckpt_path = Path(settings.SCOPE_GATE_CHECKPOINT)
            if not ckpt_path.is_absolute():
                ckpt_path = PROJECT_ROOT / ckpt_path

            if ckpt_path.exists():
                try:
                    from gate.predict import ScopeGatePredictor
                    cls._scope_gate_predictor = ScopeGatePredictor(
                        checkpoint_path=ckpt_path,
                        device="cpu",
                    )
                    logger.info(
                        f"Loaded Learned Scope Gate predictor from {ckpt_path} "
                        f"(threshold={cls._scope_gate_predictor.threshold:.4f})"
                    )
                except Exception as e:
                    logger.warning(f"Failed to load Scope Gate checkpoint ({e}). Operating in fallback mode.")
            else:
                logger.warning(f"Scope Gate checkpoint not found at {ckpt_path}. Operating in fallback mode.")

        return cls._scope_gate_predictor

    @classmethod
    def check_query_scope(cls, query: str) -> Dict[str, Any]:
        """
        Runs the learned scope gate classifier on an incoming query.
        Returns:
            Dict containing:
                - is_in_scope: bool
                - oos_score: float
                - threshold: float
                - status: str ('IN_SCOPE', 'REFUSAL_OUT_OF_SCOPE', or 'GATE_BYPASSED')
        """
        predictor = cls.get_scope_gate_predictor()
        if predictor is None:
            return {
                "is_in_scope": True,
                "oos_score": 0.0,
                "threshold": settings.SCOPE_GATE_THRESHOLD,
                "status": "GATE_BYPASSED",
            }

        try:
            res = predictor.predict_single(query, threshold=settings.SCOPE_GATE_THRESHOLD)
            is_in_scope = bool(res["is_in_scope"])
            return {
                "is_in_scope": is_in_scope,
                "oos_score": float(res["oos_score"]),
                "threshold": float(res["decision_threshold"]),
                "status": "IN_SCOPE" if is_in_scope else "REFUSAL_OUT_OF_SCOPE",
            }
        except Exception as e:
            logger.warning(f"Scope gate inference error ({e}). Allowing query to proceed to retrieval.")
            return {
                "is_in_scope": True,
                "oos_score": 0.0,
                "threshold": settings.SCOPE_GATE_THRESHOLD,
                "status": "GATE_BYPASSED",
            }

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
