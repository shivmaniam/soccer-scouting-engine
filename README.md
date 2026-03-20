# Soccer Scouting Similarity Engine

> *"Which players across global leagues play most like [Player X]?"*

A project that answers this question using player embeddings built from StatsBomb open event data, a PyTorch autoencoder, sklearn nearest-neighbor search, and a Streamlit UI.

---

## About This Project

This is an **experiment in AI-assisted development** — specifically, using [Claude Code](https://claude.ai/claude-code) to take a project from initial idea all the way to a working product. The architecture, feature engineering, model design, test suite, and documentation were all developed collaboratively with Claude Code, with a human acting as product owner and decision-maker.

The goal: see how far you can get building a real ML system when the AI handles implementation details and you stay focused on the "what" and "why."

**Live demo:** [similar-soccer-player-search.streamlit.app](https://similar-soccer-player-search.streamlit.app/)

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

# 5. Generate embeddings + build nearest-neighbor index
make embed
make index

# 6. Launch the UI
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
│   ├── nn_index.pkl            # sklearn NearestNeighbors index
│   ├── nn_id_map.json
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
│   ├── search.py       # sklearn index + similarity search
│   └── evaluate.py     # Embedding quality metrics
├── app/
│   └── streamlit_app.py
├── .github/workflows/ci.yml
├── docs/
│   ├── architecture.md
│   └── project_charter.md
├── tests/
│   ├── test_model.py
│   ├── test_features.py
│   ├── test_search.py
│   └── test_evaluate.py
├── requirements.txt
├── Makefile
└── README.md
```

---

## Tests

```bash
make test
```

38 tests covering the model, feature pipeline, search index, and evaluation metrics. All synthetic — no real data required to run the suite.

---

## CI/CD

GitHub Actions runs on every push to `main` / `develop`:
1. **Lint** — ruff
2. **Test** — pytest with coverage
3. **Smoke pipeline** — ingest 3 matches → features → train → embed → index
