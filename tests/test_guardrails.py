import pytest
from app.services.guardrails import GuardrailEngine, EXACT_REFUSAL_PHRASE
from app.models.schemas import DocumentChunk, QueryRequest
from app.services.rag_engine import rag_engine

def test_guardrail_refusal_when_no_chunks():
    """Verify that an empty context returns the exact refusal phrase."""
    eval_result = GuardrailEngine.evaluate_response(
        query="What is the recipe for chocolate cake?",
        raw_answer="Some hallucinated recipe",
        retrieved_chunks=[],
    )
    assert eval_result["final_answer"] == EXACT_REFUSAL_PHRASE
    assert eval_result["confidence_score"] == 0.0
    assert eval_result["guardrail_status"] == "REFUSAL_NOT_IN_CONTEXT"
    assert len(eval_result["sources"]) == 0

def test_guardrail_refusal_when_model_states_refusal():
    """Verify that if raw answer triggers the refusal phrase, it is cleanly standardized."""
    dummy_chunk = DocumentChunk(
        chunk_id="c1",
        document_name="doc.txt",
        content="Some irrelevant text.",
        page_number=1,
        chunk_index=0,
        metadata={},
    )
    eval_result = GuardrailEngine.evaluate_response(
        query="Where can I rent a spaceship?",
        raw_answer=f"Sorry, {EXACT_REFUSAL_PHRASE}",
        retrieved_chunks=[(dummy_chunk, 0.45)],
    )
    assert eval_result["final_answer"] == EXACT_REFUSAL_PHRASE
    assert eval_result["guardrail_status"] == "REFUSAL_NOT_IN_CONTEXT"

def test_guardrail_pass_grounded_citations():
    """Verify that high-scoring context produces verifiable sources and high confidence."""
    chunk = DocumentChunk(
        chunk_id="exam_p3_c1",
        document_name="TUM_Informatik_FPSO_Exam_Regulations.md",
        content="Die Masterarbeit umfasst einen Arbeitsumfang von 30 ECTS-Credits.",
        page_number=3,
        chunk_index=1,
        metadata={"document_name": "TUM_Informatik_FPSO_Exam_Regulations.md", "page_number": 3},
    )
    eval_result = GuardrailEngine.evaluate_response(
        query="Wie viele ECTS hat die Masterarbeit?",
        raw_answer="Die Masterarbeit umfasst 30 ECTS-Credits [TUM_Informatik_FPSO_Exam_Regulations.md, Page 3].",
        retrieved_chunks=[(chunk, 0.92)],
    )
    assert eval_result["guardrail_status"] == "PASS_GROUNDED"
    assert eval_result["confidence_score"] >= 0.65
    assert len(eval_result["sources"]) == 1
    source = eval_result["sources"][0]
    assert source.document_name == "TUM_Informatik_FPSO_Exam_Regulations.md"
    assert source.page_number == 3
    assert source.similarity_score == 0.92

def test_end_to_end_out_of_domain_query_refusal():
    """End-to-end integration test verifying that irrelevant questions receive exact refusal."""
    request = QueryRequest(
        query="How do I change the oil in a 1998 Toyota Corolla?",
        similarity_threshold=0.75,  # High threshold for unrelated query
    )
    response = rag_engine.query(request)
    assert response.answer == EXACT_REFUSAL_PHRASE
    assert response.guardrail_status == "REFUSAL_NOT_IN_CONTEXT"
    assert response.confidence_score == 0.0
