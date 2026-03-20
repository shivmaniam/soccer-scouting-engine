# Project Charter: Soccer Scouting Similarity Engine

## Problem Statement

Scouts and analysts lack a fast, data-driven way to find "stylistically similar" players across different leagues and contexts. This project answers: *"Which players across global leagues play most like [Player X]?"*

## Goals

1. Ingest StatsBomb open event data for multiple competitions
2. Build per-90 normalised statistical profiles per player
3. Learn compact (8-dim) player embeddings via an autoencoder
4. Enable sub-second similarity search over thousands of players
5. Expose results via a REST API and an interactive Streamlit dashboard

## Non-Goals

- Proprietary or paid data sources
- Real-time (live match) ingestion
- Player valuation or transfer recommendations

## Success Metrics

| Metric | Target |
|--------|--------|
| Reconstruction MSE | < 0.05 on held-out set |
| Position purity @5 | > 0.70 |
| API p95 latency | < 200 ms |
| CI pipeline pass rate | 100 % on main |

## Stack

| Layer | Technology |
|-------|-----------|
| Data | statsbombpy (free open data) |
| Features | pandas, numpy, scikit-learn |
| Model | PyTorch autoencoder |
| Vector search | scikit-learn NearestNeighbors |
| Experiment tracking | MLflow |
| Frontend | Streamlit + Plotly |
| Containerisation | Docker + Docker Compose |
| CI/CD | GitHub Actions |

## Potential Improvements

### Vector Search

The current implementation uses scikit-learn's `NearestNeighbors` with exact Euclidean search, which is appropriate for the scale of StatsBomb open data (a few thousand players). If the dataset grows significantly, the following alternatives are worth considering:

- **FAISS** (Facebook AI Similarity Search) — highly optimised for large-scale exact and approximate nearest-neighbour search. Supports GPU acceleration. A good next step if the player pool grows into the tens of thousands.
- **hnswlib** — approximate nearest-neighbour search using Hierarchical Navigable Small World graphs. Often faster than FAISS for approximate search with a simpler API.
- **Annoy** (Spotify) — approximate search using random projection trees. Lightweight, read-optimised, and easy to deploy. Index is static once built, which suits this use case.
- **Vector databases** (Chroma, Qdrant, Weaviate, Pinecone) — full persistence, filtering, and cloud hosting. Overkill at current scale but relevant if the project moves toward a hosted, multi-user product.

### Other Areas

- **Richer feature set** — incorporate positional heatmaps, pass networks, or on-ball action sequences to capture style beyond per-90 counting stats.
- **Paid / licensed data** — StatsBomb 360 or Opta data would significantly expand player and competition coverage.
- **Position-aware search** — optionally constrain results to the same positional group to improve practical utility for scouts.
- **REST API** — exposing `SimilarityIndex` behind a FastAPI service would enable programmatic access and decouple the search backend from the Streamlit frontend.

## Milestones

1. **Data pipeline** — ingest + feature engineering complete
2. **Model** — autoencoder training + MLflow tracking
3. **Search** — sklearn NearestNeighbors index built + `SimilarityIndex` tested
4. **UI** — Streamlit: search, radar chart, UMAP scatter (loads data directly from disk, no API layer)
5. **Containerisation** — Dockerfile + docker-compose
6. **CI/CD** — GitHub Actions: lint → test → smoke pipeline → Docker build
