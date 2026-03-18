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
| Vector search | FAISS |
| Experiment tracking | MLflow |
| API | FastAPI |
| Frontend | Streamlit + Plotly |
| Containerisation | Docker + Docker Compose |
| CI/CD | GitHub Actions |

## Milestones

1. **Data pipeline** — ingest + feature engineering complete
2. **Model** — autoencoder training + MLflow tracking
3. **Search** — FAISS index built + `SimilarityIndex` tested
4. **API** — FastAPI endpoints, Pydantic schemas
5. **UI** — Streamlit: search, radar chart, UMAP scatter
6. **Containerisation** — Dockerfile + docker-compose
7. **CI/CD** — GitHub Actions: lint → test → smoke pipeline → Docker build
