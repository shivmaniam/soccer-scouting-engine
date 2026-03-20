"""
Tests for src/evaluate.py — embedding quality metrics.
Uses synthetic embeddings written to tmp_path; no real model or data required.
"""

import math

import numpy as np
import pandas as pd
import pytest


# ── helpers ────────────────────────────────────────────────────────────────────

def make_embeddings(n: int = 20, with_position: bool = True, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    emb_cols = {f"emb_{i}": rng.standard_normal(n) for i in range(8)}
    df = pd.DataFrame(
        {
            "player_name": [f"Player {i}" for i in range(n)],
            "team": "Team A",
            "total_minutes": 900.0,
            **emb_cols,
        }
    )
    if with_position:
        df["position"] = ["FWD"] * (n // 2) + ["DEF"] * (n // 2)
    df.index.name = "player_id"
    return df


# ── position_purity ────────────────────────────────────────────────────────────

def test_position_purity_returns_nan_without_position_column(tmp_path):
    from src.evaluate import position_purity

    embeddings = make_embeddings(with_position=False)
    path = tmp_path / "embeddings.parquet"
    embeddings.to_parquet(path)

    result = position_purity(embeddings_path=path)
    assert math.isnan(result)


def test_position_purity_returns_float(tmp_path):
    from src.evaluate import position_purity

    # Make two clearly separated clusters so purity is high
    rng = np.random.default_rng(0)
    n = 20
    # FWD cluster centred at +10, DEF cluster at -10
    emb_fwd = {f"emb_{i}": rng.standard_normal(n // 2) + 10 for i in range(8)}
    emb_def = {f"emb_{i}": rng.standard_normal(n // 2) - 10 for i in range(8)}
    emb_cols = {k: np.concatenate([emb_fwd[k], emb_def[k]]) for k in emb_fwd}

    embeddings = pd.DataFrame(
        {
            "player_name": [f"P{i}" for i in range(n)],
            "team": "X",
            "total_minutes": 900.0,
            "position": ["FWD"] * (n // 2) + ["DEF"] * (n // 2),
            **emb_cols,
        }
    )
    embeddings.index.name = "player_id"

    path = tmp_path / "embeddings.parquet"
    embeddings.to_parquet(path)

    result = position_purity(embeddings_path=path)
    assert isinstance(result, float)
    assert 0.0 <= result <= 1.0


def test_position_purity_high_for_well_separated_clusters(tmp_path):
    from src.evaluate import position_purity

    rng = np.random.default_rng(1)
    n = 30
    emb_fwd = {f"emb_{i}": rng.standard_normal(n // 2) + 100 for i in range(8)}
    emb_def = {f"emb_{i}": rng.standard_normal(n // 2) - 100 for i in range(8)}
    emb_cols = {k: np.concatenate([emb_fwd[k], emb_def[k]]) for k in emb_fwd}

    embeddings = pd.DataFrame(
        {
            "player_name": [f"P{i}" for i in range(n)],
            "team": "X",
            "total_minutes": 900.0,
            "position": ["FWD"] * (n // 2) + ["DEF"] * (n // 2),
            **emb_cols,
        }
    )
    embeddings.index.name = "player_id"
    path = tmp_path / "embeddings.parquet"
    embeddings.to_parquet(path)

    result = position_purity(embeddings_path=path)
    assert result > 0.9, f"Expected high purity for well-separated clusters, got {result:.3f}"
