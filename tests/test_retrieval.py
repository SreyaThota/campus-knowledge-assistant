import pytest
import numpy as np
from app.services.embedding_service import EmbeddingService
from app.services.vector_store import VectorStore
from app.models.schemas import DocumentChunk

@pytest.fixture
def test_embedding_service():
    """Provides an isolated embedding service instance."""
    return EmbeddingService(provider="fallback")

@pytest.fixture
def temp_vector_store(tmp_path):
    """Provides an isolated temporary vector store instance."""
    return VectorStore(
        persist_directory=str(tmp_path / "test_chroma"),
        collection_name="test_collection",
    )

def test_embedding_dimensionality(test_embedding_service):
    """Verify that embeddings match expected dimensionality and are normalized."""
    text = "Artificial Intelligence Master Curriculum at Technical University of Munich"
    emb = test_embedding_service.embed_query(text)
    assert len(emb) == test_embedding_service.dimension
    norm = np.linalg.norm(emb)
    assert pytest.approx(norm, abs=1e-3) == 1.0

def test_vector_store_add_and_retrieve(temp_vector_store, test_embedding_service):
    """Verify storing chunks and retrieving them by semantic similarity."""
    chunk1 = DocumentChunk(
        chunk_id="doc1_p1_c0",
        document_name="library_rules.txt",
        content="Eating food is prohibited in all study and reading halls.",
        page_number=1,
        chunk_index=0,
        metadata={"document_name": "library_rules.txt", "page_number": 1},
    )
    chunk2 = DocumentChunk(
        chunk_id="doc2_p1_c0",
        document_name="exam_rules.md",
        content="Students must submit a medical certificate within 3 days of missing an exam.",
        page_number=1,
        chunk_index=0,
        metadata={"document_name": "exam_rules.md", "page_number": 1},
    )

    chunks = [chunk1, chunk2]
    embeddings = test_embedding_service.embed_documents([c.content for c in chunks])
    upserted = temp_vector_store.add_chunks(chunks, embeddings)
    assert upserted == 2

    # Query specifically for medical illness
    q_emb = test_embedding_service.embed_query("medical certificate illness deadline")
    results = temp_vector_store.similarity_search(q_emb, top_k=2, threshold=0.1)

    assert len(results) > 0
    top_chunk, score = results[0]
    assert top_chunk.document_name == "exam_rules.md"
    assert "medical certificate" in top_chunk.content
    assert 0.0 <= score <= 1.0

def test_vector_store_similarity_threshold_filtering(temp_vector_store, test_embedding_service):
    """Verify that chunks below similarity threshold are strictly filtered out."""
    chunk = DocumentChunk(
        chunk_id="doc1_c0",
        document_name="exam_rules.md",
        content="The master thesis duration is exactly six months.",
        page_number=1,
        chunk_index=0,
        metadata={"document_name": "exam_rules.md", "page_number": 1},
    )
    temp_vector_store.add_chunks([chunk], test_embedding_service.embed_documents([chunk.content]))

    # Query with high threshold for completely irrelevant topic
    irrelevant_q = test_embedding_service.embed_query("quantum submarine baking pasta recipe")
    results = temp_vector_store.similarity_search(irrelevant_q, top_k=2, threshold=0.85)
    assert len(results) == 0

def test_vector_store_delete_document(temp_vector_store, test_embedding_service):
    """Verify deleting a document removes all its chunks from the store."""
    chunk = DocumentChunk(
        chunk_id="doc_del_c0",
        document_name="to_delete.txt",
        content="Temporary campus announcement.",
        page_number=1,
        chunk_index=0,
        metadata={"document_name": "to_delete.txt", "page_number": 1},
    )
    temp_vector_store.add_chunks([chunk], test_embedding_service.embed_documents([chunk.content]))
    assert len(temp_vector_store.list_documents()) == 1

    temp_vector_store.delete_document("to_delete.txt")
    assert len(temp_vector_store.list_documents()) == 0
