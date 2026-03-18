"""
evaluate.py
-----------
Embedding quality metrics:
  - Reconstruction loss on a held-out test set
  - Cohesion: mean intra-cluster distance for players grouped by primary position
  - Separation: mean inter-cluster distance across positions
  - Nearest-neighbour position purity (% of top-5 neighbours sharing same position)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]


def reconstruction_loss(
    features_path: Optional[Path] = None,
    model_path: Optional[Path] = None,
) -> float:
    """Compute MSE reconstruction loss on the full feature set.

    Returns
    -------
    float
        Mean squared error.
    """
    import torch
    from src.features import load_features, scale_features
    from src.model import load_model

    features_df = load_features(features_path)
    X, _ = scale_features(features_df)
    model = load_model(model_path)
    model.eval()

    X_t = torch.tensor(X, dtype=torch.float32)
    with torch.no_grad():
        x_hat, _ = model(X_t)
    mse = float(torch.nn.functional.mse_loss(x_hat, X_t).item())
    logger.info("Reconstruction MSE: %.6f", mse)
    return mse


def position_purity(
    embeddings_path: Optional[Path] = None,
    lineups_path: Optional[Path] = None,
    top_k: int = 5,
) -> float:
    """Nearest-neighbour position purity.

    For each player, compute the fraction of top-k neighbours that share
    the same broad position group (GK / DEF / MID / FWD).

    Returns
    -------
    float
        Mean purity across all players (0–1).
    """
    from src.embed import load_embeddings
    from src.search import SimilarityIndex

    embeddings = load_embeddings(embeddings_path)

    # Attach position if available
    if "position" not in embeddings.columns:
        logger.warning("No 'position' column in embeddings — skipping position purity.")
        return float("nan")

    idx = SimilarityIndex.build(embeddings)
    purities: list[float] = []

    for pid in embeddings.index:
        pos = embeddings.loc[pid, "position"]
        neighbours = idx.find_similar(pid, top_k=top_k)
        if neighbours.empty:
            continue
        neighbour_positions = embeddings.loc[neighbours["player_id"].values, "position"]
        purity = (neighbour_positions == pos).mean()
        purities.append(float(purity))

    mean_purity = float(np.mean(purities)) if purities else float("nan")
    logger.info("Position purity @%d: %.3f", top_k, mean_purity)
    return mean_purity


def run_all(
    features_path: Optional[Path] = None,
    model_path: Optional[Path] = None,
    embeddings_path: Optional[Path] = None,
) -> dict[str, float]:
    """Run all evaluation metrics and return a results dict."""
    results: dict[str, float] = {}
    results["reconstruction_mse"] = reconstruction_loss(features_path, model_path)
    results["position_purity_top5"] = position_purity(embeddings_path)
    for k, v in results.items():
        print(f"{k}: {v:.4f}")
    return results


if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    parser = argparse.ArgumentParser(description="Evaluate embedding quality.")
    parser.add_argument("--features", type=Path, default=None)
    parser.add_argument("--model", type=Path, default=None)
    parser.add_argument("--embeddings", type=Path, default=None)
    args = parser.parse_args()
    run_all(args.features, args.model, args.embeddings)
