import pytest
from fastapi.testclient import TestClient
from app.main import app
from app.services.guardrails import EXACT_REFUSAL_PHRASE

@pytest.fixture(scope="module")
def client():
    """Provides a TestClient with startup events executed."""
    with TestClient(app) as test_client:
        yield test_client

def test_api_health_endpoint(client):
    """Verify health endpoint returns 200 and valid schema."""
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "total_chunks" in data
    assert "embedding_model" in data
    assert "active_llm_model" in data

def test_api_list_documents(client):
    """Verify documents endpoint returns indexed list."""
    response = client.get("/api/v1/documents")
    assert response.status_code == 200
    docs = response.json()
    assert isinstance(docs, list)

def test_api_query_grounded_answer(client):
    """Verify that a valid campus query produces answer and citations."""
    payload = {
        "query": "What are the Central Library opening hours on weekends?",
        "top_k": 3,
        "similarity_threshold": 0.15,
    }
    response = client.post("/api/v1/query", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "answer" in data
    assert "sources" in data
    assert "confidence_score" in data
    assert "latency_ms" in data
    assert data["guardrail_status"] in ["PASS_GROUNDED", "PARTIAL_RELEVANCE"]

def test_api_query_out_of_domain_refusal(client):
    """Verify that an irrelevant query triggers strict refusal."""
    payload = {
        "query": "What is the capital city of planet Neptune?",
        "top_k": 3,
        "similarity_threshold": 0.80,
    }
    response = client.post("/api/v1/query", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["answer"] == EXACT_REFUSAL_PHRASE
    assert data["guardrail_status"] == "REFUSAL_NOT_IN_CONTEXT"
    assert data["confidence_score"] == 0.0

def test_api_upload_document(client):
    """Verify document upload endpoint indexes a new document."""
    file_content = b"# Department Guidelines\nAll campus laboratories close at 20:00 on Fridays."
    files = {"file": ("test_guideline.md", file_content, "text/markdown")}
    response = client.post("/api/v1/upload", files=files)
    assert response.status_code == 200
    data = response.json()
    assert data["filename"] == "test_guideline.md"
    assert data["chunks_created"] > 0
    assert data["status"] == "SUCCESS"
