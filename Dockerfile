# Production multi-stage Dockerfile for Campus Knowledge Assistant (CampusRAG)
FROM python:3.11-slim as base

# Set environment flags
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy dependency requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Copy application source code
COPY backend ./backend
COPY frontend ./frontend
COPY data ./data

# Expose default ports for FastAPI (8000) and Streamlit (8501)
EXPOSE 8000 8501

# Default command starts backend
CMD ["uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
