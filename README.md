# Soccer Scouting Similarity Engine

> *"Which players across global leagues play most like [Player X]?"*

A portfolio project that answers this question using player embeddings built from StatsBomb open event data, a PyTorch autoencoder, FAISS vector search, a FastAPI backend, and a Streamlit UI.

---

## Quick Start

```bash
# 1. Install dependencies
make install

# 2. Pull data (La Liga 2020/21, first 10 matches)
make ingest COMP_ID=11 SEASON_ID=90 MAX_MATCHES=10

# 3. Build per-90 feature vectors
make features

# 4. Train the autoencoder (logs to MLflow)
make train

# 5. Generate embeddings + build FAISS index
make embed
make index

# 6. Start the API + UI
make api        # http://localhost:8000
make app        # http://localhost:8501
make mlflow-ui  # http://localhost:5000
```

Or run the full pipeline in one command:
```bash
make pipeline
```

---

## Architecture

See [docs/architecture.md](docs/architecture.md).

## Project Charter

See [docs/project_charter.md](docs/project_charter.md).

---

## Repo Structure

```
soccer-scouting-engine/
├── data/
│   ├── raw/                    # Parquet dumps from StatsBomb
│   ├── player_features.parquet
│   ├── embeddings.parquet
│   └── autoencoder.pt
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_features.ipynb
│   └── 03_embedding.ipynb
├── src/
│   ├── ingest.py       # StatsBomb data ingestion
│   ├── features.py     # Per-90 feature engineering
│   ├── model.py        # PyTorch autoencoder
│   ├── embed.py        # Generate embeddings
│   ├── search.py       # FAISS index + similarity search
│   └── evaluate.py     # Embedding quality metrics
├── api/
│   └── main.py         # FastAPI REST endpoints
├── app/
│   └── streamlit_app.py
├── docker/
│   ├── Dockerfile
│   └── docker-compose.yml
├── .github/workflows/ci.yml
├── docs/
│   ├── architecture.md
│   └── project_charter.md
├── requirements.txt
├── Makefile
└── README.md
```

---

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Health check |
| `GET` | `/players` | List indexed players |
| `GET` | `/players/{id}` | Player metadata |
| `GET` | `/similar/{id}` | Top-k similar players by ID |
| `POST` | `/similar/by-name` | Top-k similar players by name |

---

## Docker

```bash
make docker-build
make docker-up
```

Services: `api` (8000), `streamlit` (8501), `mlflow` (5000).

---

## CI/CD

GitHub Actions runs on every push to `main` / `develop`:
1. **Lint** — ruff
2. **Test** — pytest with coverage
3. **Smoke pipeline** — ingest 3 matches → features → train → embed → index
4. **Docker build** — API + Streamlit images
