"""
Tests for src/search.py — index building, persistence, and similarity queries.
All inputs are synthetic embeddings DataFrames; no real data required.
"""

import numpy as np
import pandas as pd
import pytest

from src.search import (
    SimilarityIndex,
    build_sklearn_index,
    load_sklearn_index,
    search_sklearn,
)


# ── helpers ────────────────────────────────────────────────────────────────────

def make_embeddings(n: int = 20, seed: int = 42) -> pd.DataFrame:
    """Return a synthetic embeddings DataFrame with the expected schema."""
    rng = np.random.default_rng(seed)
    emb_cols = {f"emb_{i}": rng.standard_normal(n) for i in range(8)}
    df = pd.DataFrame(
        {
            "player_name": [f"Player {i}" for i in range(n)],
            "team": ["Team A"] * (n // 2) + ["Team B"] * (n // 2),
            "total_minutes": rng.integers(90, 3000, n).astype(float),
            **emb_cols,
        }
    )
    df.index.name = "player_id"
    return df


# ── index build / load ─────────────────────────────────────────────────────────

def test_build_sklearn_index(tmp_path):
    embeddings = make_embeddings()
    nn, player_ids = build_sklearn_index(
        embeddings,
        index_path=tmp_path / "idx.pkl",
        id_map_path=tmp_path / "ids.json",
    )
    assert len(player_ids) == 20
    assert nn is not None


def test_load_sklearn_index_roundtrip(tmp_path):
    embeddings = make_embeddings()
    build_sklearn_index(
        embeddings,
        index_path=tmp_path / "idx.pkl",
        id_map_path=tmp_path / "ids.json",
    )
    nn, player_ids = load_sklearn_index(
        index_path=tmp_path / "idx.pkl",
        id_map_path=tmp_path / "ids.json",
    )
    assert len(player_ids) == 20


def test_player_ids_order_preserved(tmp_path):
    embeddings = make_embeddings()
    _, player_ids = build_sklearn_index(
        embeddings,
        index_path=tmp_path / "idx.pkl",
        id_map_path=tmp_path / "ids.json",
    )
    assert player_ids == list(embeddings.index)


# ── search_sklearn ─────────────────────────────────────────────────────────────

def test_search_returns_top_k_plus_one(tmp_path):
    """search_sklearn returns top_k + 1 results (caller skips self)."""
    embeddings = make_embeddings()
    nn, player_ids = build_sklearn_index(
        embeddings,
        index_path=tmp_path / "idx.pkl",
        id_map_path=tmp_path / "ids.json",
    )
    query = embeddings[[f"emb_{i}" for i in range(8)]].iloc[0].values.astype(np.float32)
    results = search_sklearn(query, nn, player_ids, top_k=5)
    assert len(results) == 6  # top_k + 1


def test_search_result_types(tmp_path):
    embeddings = make_embeddings()
    nn, player_ids = build_sklearn_index(
        embeddings,
        index_path=tmp_path / "idx.pkl",
        id_map_path=tmp_path / "ids.json",
    )
    query = embeddings[[f"emb_{i}" for i in range(8)]].iloc[0].values.astype(np.float32)
    results = search_sklearn(query, nn, player_ids, top_k=3)
    for pid, dist in results:
        assert isinstance(pid, int)
        assert isinstance(dist, float)
        assert dist >= 0.0


def test_search_nearest_is_self(tmp_path):
    """The closest match to a player's own vector should be themselves."""
    embeddings = make_embeddings()
    nn, player_ids = build_sklearn_index(
        embeddings,
        index_path=tmp_path / "idx.pkl",
        id_map_path=tmp_path / "ids.json",
    )
    query = embeddings[[f"emb_{i}" for i in range(8)]].iloc[0].values.astype(np.float32)
    results = search_sklearn(query, nn, player_ids, top_k=5)
    assert results[0][0] == player_ids[0]  # first result is self
    assert results[0][1] < 1e-6            # distance to self ≈ 0


# ── SimilarityIndex ────────────────────────────────────────────────────────────

def make_index(tmp_path, n=20):
    embeddings = make_embeddings(n)
    nn, player_ids = build_sklearn_index(
        embeddings,
        index_path=tmp_path / "idx.pkl",
        id_map_path=tmp_path / "ids.json",
    )
    return SimilarityIndex(embeddings, nn, player_ids)


def test_find_similar_by_id(tmp_path):
    idx = make_index(tmp_path)
    results = idx.find_similar(0, top_k=5)
    assert len(results) == 5
    assert 0 not in results["player_id"].values  # self excluded


def test_find_similar_by_name(tmp_path):
    idx = make_index(tmp_path)
    results = idx.find_similar("Player 0", top_k=3)
    assert len(results) == 3


def test_find_similar_case_insensitive(tmp_path):
    idx = make_index(tmp_path)
    results = idx.find_similar("player 0", top_k=3)
    assert len(results) == 3


def test_find_similar_partial_name(tmp_path):
    idx = make_index(tmp_path)
    results = idx.find_similar("Player 1", top_k=3)  # partial could match "Player 10" etc.
    assert len(results) == 3


def test_find_similar_unknown_player_raises(tmp_path):
    idx = make_index(tmp_path)
    with pytest.raises(ValueError, match="not found"):
        idx.find_similar("Zlatan Ibrahimovic", top_k=3)


def test_find_similar_result_columns(tmp_path):
    idx = make_index(tmp_path)
    results = idx.find_similar(0, top_k=5)
    for col in ("player_id", "player_name", "team", "total_minutes", "distance"):
        assert col in results.columns


def test_find_similar_distances_ascending(tmp_path):
    idx = make_index(tmp_path)
    results = idx.find_similar(0, top_k=5)
    distances = results["distance"].tolist()
    assert distances == sorted(distances)


def test_get_embedding_shape(tmp_path):
    idx = make_index(tmp_path)
    emb = idx.get_embedding(0)
    assert emb.shape == (8,)
    assert emb.dtype == np.float32
