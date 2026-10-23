from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from app.core.config import settings
from app.core.logging import logger
from app.api.routes import router as api_router
from app.services.vector_store import vector_store
from app.services.rag_engine import rag_engine

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan context manager for startup and shutdown routines."""
    logger.info("=" * 60)
    logger.info("Starting Campus Knowledge Assistant (Production RAG API)")
    logger.info(f"Environment: {settings.ENVIRONMENT}")
    logger.info(f"Embedding Model: {settings.EMBEDDING_MODEL_NAME}")
    logger.info(f"LLM Provider: {settings.LLM_PROVIDER}")
    logger.info("=" * 60)

    # Automatically index sample docs on initial startup if knowledge base is empty
    docs = vector_store.list_documents()
    if len(docs) == 0:
        logger.info("Empty vector store detected on boot. Auto-indexing sample campus documents...")
        results = rag_engine.ingest_sample_docs()
        logger.info(f"Auto-ingested {len(results)} sample documents.")
    else:
        logger.info(f"Vector store already contains {len(docs)} documents.")

    yield

    logger.info("Shutting down Campus Knowledge Assistant API.")

app = FastAPI(
    title="Campus Knowledge Assistant API",
    version="1.0.0",
    description=(
        "Production-grade Enterprise Retrieval-Augmented Generation (RAG) System "
        "indexing German university documents with strict source citations, "
        "cosine similarity thresholding, and anti-hallucination guardrails."
    ),
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# Enable CORS for Streamlit / React frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register API v1 routes
app.include_router(api_router, prefix="/api/v1")

@app.get("/", include_in_schema=False)
async def root_redirect():
    """Redirects base path to interactive OpenAPI documentation."""
    return RedirectResponse(url="/docs")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host=settings.HOST, port=settings.PORT, reload=True)
