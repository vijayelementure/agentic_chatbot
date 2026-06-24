.PHONY: install install-dev ingest chat serve test lint format docker-build docker-up clean

install:
	pip install -r requirements.txt

install-dev:
	pip install -r requirements-dev.txt

ingest:
	python -m agentic_rag.ingest

chat:
	python -m agentic_rag.cli

serve:
	uvicorn agentic_rag.api.main:app --host 0.0.0.0 --port 8000 --reload

test:
	pytest --cov=agentic_rag --cov-report=term-missing

lint:
	ruff check agentic_rag tests

format:
	ruff check --fix agentic_rag tests

docker-build:
	docker compose build

docker-up:
	docker compose up

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	rm -rf .pytest_cache .mypy_cache .ruff_cache .coverage htmlcov
