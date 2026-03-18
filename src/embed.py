"""
embed.py
--------
Generate 8-dim player embeddings from the trained autoencoder and persist
them to ``data/embeddings.parquet``.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"


def generate_embeddings(
    features_path: Optional[Path] = None,
    model_path: Optional[Path] = None,
    output_path: Optional[Path] = None,
) -> pd.DataFrame:
    """Run every player through the encoder and save the latent vectors.

    Parameters
    ----------
    features_path:
        Path to the feature Parquet; defaults to ``data/player_features.parquet``.
    model_path:
        Path to the saved model state dict; defaults to ``data/autoencoder.pt``.
    output_path:
        Destination Parquet; defaults to ``data/embeddings.parquet``.

    Returns
    -------
    pd.DataFrame
        Columns: player_id (index), player_name, team, total_minutes,
        emb_0 … emb_7.
    """
    # Lazy imports — torch and the pipeline modules are only needed when
    # re-generating embeddings, not when loading pre-built ones.
    import torch
    from src.features import load_features, scale_features
    from src.model import load_model

    features_df = load_features(features_path)
    X, _ = scale_features(features_df)

    model = load_model(model_path)
    model.eval()

    with torch.no_grad():
        X_t = torch.tensor(X, dtype=torch.float32)
        embeddings = model.encode(X_t).numpy()

    emb_cols = [f"emb_{i}" for i in range(embeddings.shape[1])]
    emb_df = pd.DataFrame(embeddings, index=features_df.index, columns=emb_cols)

    meta = features_df[["player_name", "team", "total_minutes"]]
    result = meta.join(emb_df)

    out = output_path or DATA_DIR / "embeddings.parquet"
    result.to_parquet(out)
    logger.info("Saved %d player embeddings → %s", len(result), out)
    return result


def load_embeddings(path: Optional[Path] = None) -> pd.DataFrame:
    """Load saved embeddings from Parquet."""
    path = path or DATA_DIR / "embeddings.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Embeddings not found: {path}. Run embed.py first.")
    return pd.read_parquet(path)


if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    parser = argparse.ArgumentParser(description="Generate player embeddings.")
    parser.add_argument("--features", type=Path, default=None)
    parser.add_argument("--model", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    df = generate_embeddings(args.features, args.model, args.output)
    print(df.head())
