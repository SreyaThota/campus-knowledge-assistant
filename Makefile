.PHONY: help install test test-unit test-api run-backend run-frontend docker-up docker-down clean

help:
	@echo "Campus Knowledge Assistant - Developer Commands"
	@echo "  make install       Install all dependencies into active venv"
	@echo "  make test          Run full pytest test suite"
	@echo "  make test-unit     Run chunking and retrieval unit tests"
	@echo "  make run-backend   Start FastAPI backend on port 8000"
	@echo "  make run-frontend  Start Streamlit frontend on port 8501"
	@echo "  make docker-up     Start all services via docker-compose"
	@echo "  make docker-down   Stop docker-compose containers"
	@echo "  make clean         Remove cache files and temporary test artifacts"

install:
	pip install --upgrade pip
	pip install -r requirements.txt

test:
	python -m pytest tests/ -v --tb=short

test-unit:
	python -m pytest tests/test_chunking.py tests/test_retrieval.py -v

run-backend:
	PYTHONPATH=backend python -m uvicorn backend.app.main:app --host 0.0.0.0 --port 8000 --reload

run-frontend:
	BACKEND_API_URL=http://localhost:8000/api/v1 streamlit run frontend/app.py

docker-up:
	docker-compose up --build -d

docker-down:
	docker-compose down

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	rm -rf data/chroma_db
