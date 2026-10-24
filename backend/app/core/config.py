import os
from pathlib import Path
from typing import Optional, Literal
from pydantic_settings import BaseSettings, SettingsConfigDict

# Base directory for the campus-knowledge-assistant repository
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent

class Settings(BaseSettings):
    """
    Application configuration loaded from environment variables and .env file.
    Follows 12-factor application methodology.
    """
    model_config = SettingsConfigDict(
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore"
    )

    # Server settings
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    ENVIRONMENT: str = "development"
    LOG_LEVEL: str = "INFO"

    # Embedding settings
    EMBEDDING_PROVIDER: Literal["huggingface", "openai", "auto"] = "huggingface"
    EMBEDDING_MODEL_NAME: str = "sentence-transformers/all-MiniLM-L6-v2"

    # ChromaDB Vector Store settings
    CHROMA_PERSIST_DIRECTORY: str = str(PROJECT_ROOT / "data" / "chroma_db")
    CHROMA_COLLECTION_NAME: str = "campus_knowledge_base"

    # Ingestion & Chunking parameters
    # 750 chars (~150 tokens) provides high semantic granularity for university regulations
    CHUNK_SIZE: int = 750
    CHUNK_OVERLAP: int = 150

    # RAG Retrieval parameters
    SIMILARITY_THRESHOLD: float = 0.35
    TOP_K: int = 4

    # LLM Provider settings
    LLM_PROVIDER: Literal["auto", "ollama", "openai", "mock"] = "auto"
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_MODEL: str = "llama3:8b"
    OPENAI_API_KEY: Optional[str] = None
    OPENAI_MODEL: str = "gpt-4o-mini"

    # Sample Documents directory
    SAMPLE_DOCS_DIRECTORY: str = str(PROJECT_ROOT / "data" / "sample_docs")

    # Learned Scope Gate parameters
    ENABLE_SCOPE_GATE: bool = True
    SCOPE_GATE_CHECKPOINT: str = str(PROJECT_ROOT / "results" / "checkpoints" / "scope_gate_seed_2026.pt")
    SCOPE_GATE_THRESHOLD: float = 0.3219

    @property
    def is_production(self) -> bool:
        return self.ENVIRONMENT.lower() == "production"

# Singleton instance
settings = Settings()
