# Architecture

## System Overview

```
StatsBomb Open Data
       │
       ▼
┌──────────────┐    parquet     ┌──────────────────┐    parquet
│  ingest.py   │ ────────────▶ │   features.py    │ ──────────▶  data/player_features.parquet
└──────────────┘  data/raw/    └──────────────────┘
                                       │
                                       ▼
                               ┌──────────────┐   autoencoder.pt
                               │   model.py   │ ──────────────────▶  MLflow
                               └──────────────┘
                                       │
                                       ▼
                               ┌──────────────┐   embeddings.parquet
                               │   embed.py   │ ───────────────────▶  data/embeddings.parquet
                               └──────────────┘
                                       │
                                       ▼
                               ┌──────────────┐   nn_index.pkl
                               │   search.py  │ ───────────────────▶  data/nn_index.pkl
                               └──────────────┘
                                       │
                                       ▼
                               ┌──────────────┐
                               │  Streamlit   │
                               │  app/        │
                               └──────────────┘
```

## Component Descriptions

| Component | File | Responsibility |
|-----------|------|----------------|
| Ingest | `src/ingest.py` | Pull StatsBomb open data → Parquet |
| Features | `src/features.py` | Per-90 feature vectors (30 dims) |
| Model | `src/model.py` | PyTorch autoencoder training + MLflow |
| Embed | `src/embed.py` | Encode all players → 8-dim embeddings |
| Search | `src/search.py` | sklearn NearestNeighbors index + `SimilarityIndex` class |
| Evaluate | `src/evaluate.py` | Reconstruction loss, position purity |
| UI | `app/streamlit_app.py` | Radar chart, UMAP scatter, results table (loads data directly from disk) |

## Feature Groups (30 features)

- **Attacking (8):** goals, shots, xG, xA, key passes, dribbles completed, touches in box, progressive carries
- **Passing (8):** pass completion %, total passes, forward pass %, long ball %, crosses, through balls, switches of play, progressive passes
- **Defending (7):** tackles won, interceptions, clearances, blocks, pressures, pressure success %, duel win %
- **Style (7):** receptions, carries, dispossessed, fouls won, fouls committed, yellow cards, completed passes

## Model Architecture

```
Input (30) → Linear(64) → BN → ReLU → Linear(32) → ReLU → Linear(8)   [encoder]
         (8) → Linear(32) → ReLU → Linear(64) → ReLU → Linear(30)     [decoder]
```

Loss: MSE reconstruction. Optimizer: Adam (lr=1e-3). Trained for 100 epochs.
