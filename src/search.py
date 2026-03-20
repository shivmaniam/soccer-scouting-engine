"""
search.py
---------
Build a scikit-learn NearestNeighbors index over player embeddings and expose
nearest-neighbour search.
"""

from __future__ import annotations

import json
import logging
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

from src.embed import load_embeddings

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"

EMB_COLS_PREFIX = "emb_"


def _get_emb_matrix(embeddings: pd.DataFrame) -> np.ndarray:
    """Extract the float32 embedding matrix from the embeddings DataFrame."""
    emb_cols = [c for c in embeddings.columns if c.startswith(EMB_COLS_PREFIX)]
    return embeddings[emb_cols].values.astype(np.float32)


# ── sklearn backend ────────────────────────────────────────────────────────────

def build_sklearn_index(
    embeddings: pd.DataFrame,
    index_path: Optional[Path] = None,
    id_map_path: Optional[Path] = None,
) -> tuple[NearestNeighbors, list[int]]:
    """Fit and persist a sklearn NearestNeighbors index.

    Parameters
    ----------
    embeddings:
        Output of :func:`~src.embed.load_embeddings`.
    index_path:
        Where to write the pickled NearestNeighbors object.
    id_map_path:
        Where to write the player_id list (JSON), needed to map integer
        positions back to player IDs.

    Returns
    -------
    tuple[NearestNeighbors, list[int]]
        The fitted index and the ordered list of player IDs.
    """
    index_path = index_path or DATA_DIR / "nn_index.pkl"
    id_map_path = id_map_path or DATA_DIR / "nn_id_map.json"

    X = _get_emb_matrix(embeddings)

    nn = NearestNeighbors(metric="euclidean", algorithm="auto")
    nn.fit(X)

    with open(index_path, "wb") as f:
        pickle.dump(nn, f)

    player_ids = [int(pid) for pid in embeddings.index]
    with open(id_map_path, "w") as f:
        json.dump(player_ids, f)

    logger.info("Built sklearn index (%d vectors, dim=%d) → %s", len(player_ids), X.shape[1], index_path)
    return nn, player_ids


def load_sklearn_index(
    index_path: Optional[Path] = None,
    id_map_path: Optional[Path] = None,
) -> tuple[NearestNeighbors, list[int]]:
    """Load a persisted sklearn NearestNeighbors index."""
    index_path = index_path or DATA_DIR / "nn_index.pkl"
    id_map_path = id_map_path or DATA_DIR / "nn_id_map.json"

    with open(index_path, "rb") as f:
        nn = pickle.load(f)
    with open(id_map_path, "r") as f:
        player_ids = json.load(f)
    return nn, player_ids


def search_sklearn(
    query_vector: np.ndarray,
    nn: NearestNeighbors,
    player_ids: list[int],
    top_k: int = 10,
) -> list[tuple[int, float]]:
    """Return top-k nearest neighbours from a sklearn index.

    Parameters
    ----------
    query_vector:
        1-D float32 array of shape (latent_dim,).
    nn:
        A fitted NearestNeighbors instance.
    player_ids:
        Ordered player ID list (same order as index was built).
    top_k:
        Number of results to return.

    Returns
    -------
    list[tuple[int, float]]
        (player_id, euclidean_distance) pairs, sorted by ascending distance.
    """
    q = query_vector.reshape(1, -1).astype(np.float32)
    distances, indices = nn.kneighbors(q, n_neighbors=top_k + 1)  # +1 to skip self

    results = []
    for dist, idx in zip(distances[0], indices[0]):
        results.append((player_ids[idx], float(dist)))
    return results


# ── high-level API ────────────────────────────────────────────────────────────

class SimilarityIndex:
    """Wraps sklearn NearestNeighbors search with player metadata for easy lookup.

    Example
    -------
    >>> idx = SimilarityIndex.from_disk()
    >>> idx.find_similar("Lionel Messi", top_k=5)
    """

    def __init__(
        self,
        embeddings: pd.DataFrame,
        nn: NearestNeighbors,
        player_ids: list[int],
    ) -> None:
        self._embeddings = embeddings
        self._nn = nn
        self._player_ids = player_ids
        # Build reverse lookup: player_name (lower) → player_id
        self._name_to_id: dict[str, int] = {
            str(row["player_name"]).lower(): pid
            for pid, row in embeddings.iterrows()
            if pd.notna(row.get("player_name"))
        }

    @classmethod
    def build(cls, embeddings: Optional[pd.DataFrame] = None) -> "SimilarityIndex":
        """Build the index from embeddings (and persist to disk)."""
        if embeddings is None:
            embeddings = load_embeddings()
        nn, player_ids = build_sklearn_index(embeddings)
        return cls(embeddings, nn, player_ids)

    @classmethod
    def from_disk(cls) -> "SimilarityIndex":
        """Load a pre-built index from disk."""
        embeddings = load_embeddings()
        nn, player_ids = load_sklearn_index()
        return cls(embeddings, nn, player_ids)

    def find_similar(
        self,
        player: int | str,
        top_k: int = 10,
    ) -> pd.DataFrame:
        """Find the most similar players to a given player.

        Parameters
        ----------
        player:
            Player ID (int) or player name (str, case-insensitive).
        top_k:
            Number of similar players to return (excluding the query player).

        Returns
        -------
        pd.DataFrame
            Columns: player_id, player_name, team, total_minutes, distance.
        """
        if isinstance(player, str):
            pid = self._name_to_id.get(player.lower())
            if pid is None:
                # Fuzzy fallback: partial name match
                matches = [k for k in self._name_to_id if player.lower() in k]
                if not matches:
                    raise ValueError(f"Player '{player}' not found in index.")
                pid = self._name_to_id[matches[0]]
                logger.info("Resolved '%s' → '%s' (pid=%d)", player, matches[0], pid)
        else:
            pid = player

        if pid not in self._embeddings.index:
            raise ValueError(f"player_id={pid} not in index.")

        emb_cols = [c for c in self._embeddings.columns if c.startswith(EMB_COLS_PREFIX)]
        query_vec = self._embeddings.loc[pid, emb_cols].values.astype(np.float32)

        raw = search_sklearn(query_vec, self._nn, self._player_ids, top_k=top_k + 1)

        rows = []
        for result_pid, dist in raw:
            if result_pid == pid:
                continue  # skip self
            meta = self._embeddings.loc[result_pid]
            rows.append(
                {
                    "player_id": result_pid,
                    "player_name": meta.get("player_name"),
                    "team": meta.get("team"),
                    "total_minutes": meta.get("total_minutes"),
                    "distance": dist,
                }
            )
            if len(rows) == top_k:
                break

        return pd.DataFrame(rows)

    def get_embedding(self, player_id: int) -> np.ndarray:
        """Return the raw embedding vector for a player."""
        emb_cols = [c for c in self._embeddings.columns if c.startswith(EMB_COLS_PREFIX)]
        return self._embeddings.loc[player_id, emb_cols].values.astype(np.float32)


# ── CLI entry-point ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    parser = argparse.ArgumentParser(description="Build or query the similarity index.")
    parser.add_argument("--build", action="store_true", help="Build the sklearn index.")
    parser.add_argument("--query", type=str, default=None, help="Player name to query.")
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args()

    if args.build:
        SimilarityIndex.build()
    elif args.query:
        idx = SimilarityIndex.from_disk()
        results = idx.find_similar(args.query, top_k=args.top_k)
        print(results.to_string(index=False))
    else:
        parser.print_help()
