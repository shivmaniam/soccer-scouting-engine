.PHONY: help install lint test ingest features train embed index api app docker-build docker-up clean

PYTHON      := python
SRC         := src
COMP_ID     ?= 11    # La Liga
SEASON_ID   ?= 90    # 2020/21
MAX_MATCHES ?= 10    # override for quick smoke tests

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

# ── setup ─────────────────────────────────────────────────────────────────────

install: ## Install Python dependencies
	pip install -r requirements.txt

# ── code quality ──────────────────────────────────────────────────────────────

lint: ## Run ruff linter
	ruff check $(SRC) api app

format: ## Auto-fix formatting with ruff
	ruff format $(SRC) api app

test: ## Run test suite with coverage
	pytest tests/ -v --cov=$(SRC) --cov-report=term-missing

# ── pipeline ──────────────────────────────────────────────────────────────────

ingest: ## Pull StatsBomb open data  (COMP_ID=11 SEASON_ID=90 MAX_MATCHES=10)
	$(PYTHON) -m src.ingest \
		--competition-id $(COMP_ID) \
		--season-id $(SEASON_ID) \
		--max-matches $(MAX_MATCHES)

ingest-all: ## Ingest all available StatsBomb open-data competitions
	$(PYTHON) -m src.ingest

features: ## Build per-90 player feature matrix
	$(PYTHON) -m src.features \
		--competition-id $(COMP_ID) \
		--season-id $(SEASON_ID) \
		--min-minutes 90

train: ## Train the autoencoder embedding model (logs to MLflow)
	$(PYTHON) -m src.model

embed: ## Generate player embeddings from the trained model
	$(PYTHON) -m src.embed

index: ## Build FAISS / Chroma vector index
	$(PYTHON) -m src.search --build

evaluate: ## Evaluate embedding quality
	$(PYTHON) -m src.evaluate

# ── full pipeline shortcut ────────────────────────────────────────────────────

pipeline: ingest features train embed index ## Run the complete end-to-end pipeline

# ── services ──────────────────────────────────────────────────────────────────

api: ## Start FastAPI server (development mode)
	uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

app: ## Start Streamlit frontend
	streamlit run app/streamlit_app.py --server.port 8501

mlflow-ui: ## Open MLflow experiment tracking UI
	mlflow ui --backend-store-uri mlflow/ --host 0.0.0.0 --port 5000

# ── docker ────────────────────────────────────────────────────────────────────

docker-build: ## Build all Docker images
	docker compose -f docker/docker-compose.yml build

docker-up: ## Start all services via Docker Compose
	docker compose -f docker/docker-compose.yml up

docker-down: ## Stop all Docker Compose services
	docker compose -f docker/docker-compose.yml down

# ── housekeeping ──────────────────────────────────────────────────────────────

clean: ## Remove cached data, build artefacts and __pycache__
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache .ruff_cache htmlcov .coverage
