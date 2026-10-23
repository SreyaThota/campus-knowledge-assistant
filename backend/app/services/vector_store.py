import os
import json
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
import numpy as np
from app.core.config import settings
from app.core.logging import logger
from app.models.schemas import DocumentChunk, DocumentInfo

class VectorStore:
    """
    Production ChromaDB vector store client with persistent on-disk storage,
    rich metadata indexing, cosine similarity ranking, and document lifecycle management.
    Includes an in-memory fallback store if chromadb binary wheels encounter platform hurdles.
    """

    def __init__(
        self,
        persist_directory: Optional[str] = None,
        collection_name: Optional[str] = None,
    ):
        self.persist_directory = persist_directory or settings.CHROMA_PERSIST_DIRECTORY
        self.collection_name = collection_name or settings.CHROMA_COLLECTION_NAME
        Path(self.persist_directory).mkdir(parents=True, exist_ok=True)

        self._chroma_client = None
        self._collection = None
        self._fallback_mode = False
        self._fallback_records: Dict[str, Dict[str, Any]] = {}

        self._initialize_store()

    def _initialize_store(self):
        """Initializes ChromaDB with persistent HNSW cosine indexing."""
        try:
            import chromadb
            from chromadb.config import Settings as ChromaSettings

            logger.info(f"Connecting to persistent ChromaDB at '{self.persist_directory}'...")
            self._chroma_client = chromadb.PersistentClient(
                path=self.persist_directory,
                settings=ChromaSettings(anonymized_telemetry=False),
            )
            self._collection = self._chroma_client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"},
            )
            logger.info(
                f"ChromaDB collection '{self.collection_name}' ready. "
                f"Total indexed vectors: {self._collection.count()}"
            )
        except Exception as e:
            logger.warning(
                f"Failed to initialize ChromaDB ({e}). "
                f"Switching to resilient persistent JSON/Vector in-memory store."
            )
            self._fallback_mode = True
            self._load_fallback_data()

    def _load_fallback_data(self):
        """Loads fallback persistence from disk if available."""
        fallback_file = Path(self.persist_directory) / "fallback_store.json"
        if fallback_file.exists():
            try:
                with open(fallback_file, "r", encoding="utf-8") as f:
                    self._fallback_records = json.load(f)
                logger.info(f"Loaded {len(self._fallback_records)} vectors from fallback cache.")
            except Exception as e:
                logger.warning(f"Could not load fallback store ({e})")
                self._fallback_records = {}

    def _save_fallback_data(self):
        """Saves fallback records to disk."""
        fallback_file = Path(self.persist_directory) / "fallback_store.json"
        try:
            with open(fallback_file, "w", encoding="utf-8") as f:
                json.dump(self._fallback_records, f)
        except Exception as e:
            logger.warning(f"Could not save fallback store: {e}")

    def add_chunks(self, chunks: List[DocumentChunk], embeddings: List[List[float]]) -> int:
        """
        Inserts or updates document chunks and their embeddings into vector store.
        """
        if not chunks:
            return 0

        ids = [c.chunk_id for c in chunks]
        documents = [c.content for c in chunks]
        metadatas = [c.metadata for c in chunks]

        if not self._fallback_mode and self._collection is not None:
            try:
                self._collection.upsert(
                    ids=ids,
                    embeddings=embeddings,
                    documents=documents,
                    metadatas=metadatas,
                )
                logger.info(f"Successfully upserted {len(chunks)} chunks into ChromaDB.")
                return len(chunks)
            except Exception as e:
                logger.error(f"Error upserting to ChromaDB ({e}). Storing in fallback cache.")

        # Fallback storage path
        for chunk, emb in zip(chunks, embeddings):
            self._fallback_records[chunk.chunk_id] = {
                "chunk": chunk.model_dump(),
                "embedding": emb,
            }
        self._save_fallback_data()
        logger.info(f"Stored {len(chunks)} chunks in fallback vector store.")
        return len(chunks)

    def similarity_search(
        self,
        query_embedding: List[float],
        top_k: int = 4,
        threshold: float = 0.35,
        filter_doc: Optional[str] = None,
    ) -> List[Tuple[DocumentChunk, float]]:
        """
        Dense vector retrieval with Cosine Similarity filtering.
        Returns matched DocumentChunks and their cosine similarity score (0.0 to 1.0).
        """
        results: List[Tuple[DocumentChunk, float]] = []

        if not self._fallback_mode and self._collection is not None:
            try:
                where_filter = {"document_name": filter_doc} if filter_doc else None
                chroma_results = self._collection.query(
                    query_embeddings=[query_embedding],
                    n_results=top_k * 2,  # Query wider pool for threshold filtering
                    where=where_filter,
                    include=["documents", "metadatas", "distances"],
                )

                if chroma_results and chroma_results["ids"] and chroma_results["ids"][0]:
                    ids = chroma_results["ids"][0]
                    docs = chroma_results["documents"][0]
                    metadatas = chroma_results["metadatas"][0]
                    distances = chroma_results["distances"][0]

                    for cid, doc_text, meta, dist in zip(ids, docs, metadatas, distances):
                        # For ChromaDB cosine distance d in [0, 2], cosine_sim = 1 - d
                        cosine_sim = max(0.0, min(1.0, 1.0 - dist))
                        if cosine_sim >= threshold:
                            chunk = DocumentChunk(
                                chunk_id=cid,
                                document_name=meta.get("document_name", "unknown"),
                                content=doc_text,
                                page_number=int(meta.get("page_number", 1)),
                                chunk_index=int(meta.get("chunk_index", 0)),
                                metadata=meta,
                            )
                            results.append((chunk, round(cosine_sim, 4)))

                    results.sort(key=lambda x: x[1], reverse=True)
                    return results[:top_k]
            except Exception as e:
                logger.error(f"ChromaDB search query failed ({e}). Attempting fallback search.")

        # Fallback Vector Search (Brute force exact Cosine similarity)
        q_vec = np.array(query_embedding, dtype=np.float32)
        q_norm = np.linalg.norm(q_vec)
        if q_norm == 0:
            return []

        scored: List[Tuple[DocumentChunk, float]] = []
        for cid, record in self._fallback_records.items():
            meta = record["chunk"]["metadata"]
            if filter_doc and meta.get("document_name") != filter_doc:
                continue

            c_vec = np.array(record["embedding"], dtype=np.float32)
            c_norm = np.linalg.norm(c_vec)
            if c_norm > 0:
                similarity = float(np.dot(q_vec, c_vec) / (q_norm * c_norm))
            else:
                similarity = 0.0

            if similarity >= threshold:
                chunk = DocumentChunk(**record["chunk"])
                scored.append((chunk, round(similarity, 4)))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

    def list_documents(self) -> List[DocumentInfo]:
        """Aggregates and returns metadata for all unique indexed campus documents."""
        doc_stats: Dict[str, Dict[str, Any]] = {}

        if not self._fallback_mode and self._collection is not None:
            try:
                data = self._collection.get(include=["metadatas"])
                if data and data["metadatas"]:
                    for meta in data["metadatas"]:
                        doc_name = meta.get("document_name", "unknown")
                        if doc_name not in doc_stats:
                            doc_stats[doc_name] = {
                                "chunks_count": 0,
                                "pages": set(),
                                "timestamp": meta.get("timestamp", ""),
                            }
                        doc_stats[doc_name]["chunks_count"] += 1
                        doc_stats[doc_name]["pages"].add(int(meta.get("page_number", 1)))

                    return [
                        DocumentInfo(
                            document_name=name,
                            chunks_count=info["chunks_count"],
                            pages_count=len(info["pages"]),
                            ingestion_timestamp=info["timestamp"],
                        )
                        for name, info in doc_stats.items()
                    ]
            except Exception as e:
                logger.error(f"Error fetching document list from ChromaDB: {e}")

        # Fallback store listing
        for cid, record in self._fallback_records.items():
            meta = record["chunk"]["metadata"]
            doc_name = meta.get("document_name", "unknown")
            if doc_name not in doc_stats:
                doc_stats[doc_name] = {
                    "chunks_count": 0,
                    "pages": set(),
                    "timestamp": meta.get("timestamp", ""),
                }
            doc_stats[doc_name]["chunks_count"] += 1
            doc_stats[doc_name]["pages"].add(int(meta.get("page_number", 1)))

        return [
            DocumentInfo(
                document_name=name,
                chunks_count=info["chunks_count"],
                pages_count=len(info["pages"]),
                ingestion_timestamp=info["timestamp"],
            )
            for name, info in doc_stats.items()
        ]

    def delete_document(self, document_name: str) -> bool:
        """Deletes all chunks and vectors associated with a document name."""
        deleted = False
        if not self._fallback_mode and self._collection is not None:
            try:
                self._collection.delete(where={"document_name": document_name})
                logger.info(f"Deleted document '{document_name}' from ChromaDB.")
                deleted = True
            except Exception as e:
                logger.error(f"Failed to delete '{document_name}' from ChromaDB: {e}")

        # Clean from fallback
        keys_to_del = [
            cid for cid, rec in self._fallback_records.items()
            if rec["chunk"]["metadata"].get("document_name") == document_name
        ]
        for k in keys_to_del:
            del self._fallback_records[k]
            deleted = True
        if keys_to_del:
            self._save_fallback_data()

        return deleted

    def get_stats(self) -> Dict[str, Any]:
        """Returns statistics on stored vectors."""
        docs = self.list_documents()
        total_chunks = sum(d.chunks_count for d in docs)
        return {
            "collection_name": self.collection_name,
            "total_chunks": total_chunks,
            "unique_documents": len(docs),
            "documents": [d.model_dump() for d in docs],
            "persist_directory": self.persist_directory,
            "backend": "ChromaDB" if not self._fallback_mode else "Persistent JSON Vector Store",
        }

# Singleton instance
vector_store = VectorStore()
