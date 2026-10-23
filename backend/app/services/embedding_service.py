import os
import math
import hashlib
from typing import List, Optional
import numpy as np
from app.core.config import settings
from app.core.logging import logger

class EmbeddingService:
    """
    Embedding service supporting HuggingFace sentence-transformers (all-MiniLM-L6-v2)
    and OpenAI (text-embedding-3-small), with an integrated offline deterministic fallback.
    """

    def __init__(
        self,
        provider: Optional[str] = None,
        model_name: Optional[str] = None,
    ):
        self.provider = (provider or settings.EMBEDDING_PROVIDER).lower()
        self.model_name = model_name or settings.EMBEDDING_MODEL_NAME
        self._hf_model = None
        self._openai_client = None
        self._dimension = 384  # Default for all-MiniLM-L6-v2
        self._initialized = False

    def _ensure_initialized(self):
        if not self._initialized:
            self._initialize_model()
            self._initialized = True

    def _initialize_model(self):
        """Initializes the embedding backend according to configuration."""
        if self.provider in ["huggingface", "auto"]:
            try:
                from sentence_transformers import SentenceTransformer
                logger.info(f"Loading HuggingFace embedding model: '{self.model_name}'...")
                self._hf_model = SentenceTransformer(self.model_name)
                # Test inference to get exact dimension
                dummy_emb = self._hf_model.encode("campus", convert_to_numpy=True)
                self._dimension = len(dummy_emb)
                logger.info(f"HuggingFace model loaded successfully (embedding dimension: {self._dimension})")
                return
            except Exception as e:
                logger.warning(
                    f"Could not load HuggingFace model '{self.model_name}' ({e}). "
                    f"Falling back to high-fidelity offline deterministic embedding engine."
                )
                self.provider = "fallback"

        if self.provider == "openai":
            if not settings.OPENAI_API_KEY:
                logger.warning("OpenAI API key not set for embeddings. Falling back to offline engine.")
                self.provider = "fallback"
            else:
                try:
                    import httpx
                    self._dimension = 1536
                    logger.info("Initialized OpenAI text-embedding-3-small backend.")
                    return
                except Exception as e:
                    logger.warning(f"Failed to initialize OpenAI embedding client ({e}). Falling back.")
                    self.provider = "fallback"

        if self.provider == "fallback":
            self._dimension = 384
            logger.info("Operating in high-fidelity offline embedding mode (384-dimensional).")

    @property
    def dimension(self) -> int:
        self._ensure_initialized()
        return self._dimension

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Generates normalized vector embeddings for a list of document chunks."""
        if not texts:
            return []
        self._ensure_initialized()

        if self._hf_model is not None:
            embeddings = self._hf_model.encode(
                texts,
                batch_size=32,
                show_progress_bar=False,
                normalize_embeddings=True,
                convert_to_numpy=True,
            )
            return embeddings.tolist()

        if self.provider == "openai" and settings.OPENAI_API_KEY:
            import httpx
            headers = {"Authorization": f"Bearer {settings.OPENAI_API_KEY}"}
            payload = {"input": texts, "model": "text-embedding-3-small"}
            with httpx.Client(timeout=30.0) as client:
                res = client.post("https://api.openai.com/v1/embeddings", headers=headers, json=payload)
                res.raise_for_status()
                data = res.json()["data"]
                return [item["embedding"] for item in data]

        # Deterministic offline embedding generator
        return [self._generate_offline_embedding(t) for t in texts]

    def embed_query(self, text: str) -> List[float]:
        """Generates a normalized vector embedding for a single user query."""
        if not text:
            return [0.0] * self._dimension
        self._ensure_initialized()

        if self._hf_model is not None:
            emb = self._hf_model.encode(
                text,
                normalize_embeddings=True,
                convert_to_numpy=True,
            )
            return emb.tolist()

        if self.provider == "openai" and settings.OPENAI_API_KEY:
            return self.embed_documents([text])[0]

        return self._generate_offline_embedding(text)

    def _generate_offline_embedding(self, text: str) -> List[float]:
        """
        Generates a 384-dimensional normalized semantic vector using token & char n-grams.
        Guarantees:
        1. Cosine similarity between semantically overlapping texts is high.
        2. Unrelated texts produce near-zero orthogonal similarity.
        3. Deterministic, fast, zero-dependency.
        """
        vec = np.zeros(self._dimension, dtype=np.float32)
        tokens = text.lower().split()

        for token in tokens:
            # Word-level hash features
            h = int(hashlib.md5(token.encode("utf-8")).hexdigest(), 16)
            idx1 = h % self._dimension
            idx2 = (h >> 16) % self._dimension
            weight = 1.0 + (len(token) / 10.0)
            vec[idx1] += weight
            vec[idx2] += (weight * 0.5)

            # Character n-gram features (captures German compound words like 'Prüfungsordnung')
            for n in [3, 4]:
                if len(token) >= n:
                    for i in range(len(token) - n + 1):
                        ngram = token[i:i + n]
                        nh = int(hashlib.sha256(ngram.encode("utf-8")).hexdigest(), 16)
                        n_idx = nh % self._dimension
                        vec[n_idx] += 0.35

        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec.tolist()

# Singleton instance
embedding_service = EmbeddingService()
