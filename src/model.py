"""
model.py
--------
PyTorch shallow autoencoder: 30-dim input → 8-dim latent embedding.

Architecture:
    Encoder: Linear(30→64) → BatchNorm → ReLU → Linear(64→32) → ReLU → Linear(32→8)
    Decoder: Linear(8→32)  → ReLU → Linear(32→64) → ReLU → Linear(64→30)

Training is logged to MLflow (experiment: "scouting-autoencoder").
"""

from __future__ import annotations

import argparse
import contextlib
import logging
from pathlib import Path
from typing import Optional

import mlflow
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.features import FEATURE_COLUMNS, load_features, scale_features

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "data"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

INPUT_DIM = 30
LATENT_DIM = 8


# ── model architecture ────────────────────────────────────────────────────────

class Encoder(nn.Module):
    """Maps 30-dim feature vector to 8-dim latent code."""

    def __init__(self, input_dim: int = INPUT_DIM, latent_dim: int = LATENT_DIM) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, latent_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Decoder(nn.Module):
    """Reconstructs 30-dim feature vector from 8-dim latent code."""

    def __init__(self, latent_dim: int = LATENT_DIM, output_dim: int = INPUT_DIM) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class PlayerAutoencoder(nn.Module):
    """Full autoencoder: encode then decode."""

    def __init__(
        self,
        input_dim: int = INPUT_DIM,
        latent_dim: int = LATENT_DIM,
    ) -> None:
        super().__init__()
        self.encoder = Encoder(input_dim, latent_dim)
        self.decoder = Decoder(latent_dim, input_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat, z

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Return latent embedding without reconstruction."""
        return self.encoder(x)


# ── training ──────────────────────────────────────────────────────────────────

def train(
    model: PlayerAutoencoder,
    X_train: np.ndarray,
    X_val: Optional[np.ndarray] = None,
    epochs: int = 100,
    batch_size: int = 64,
    lr: float = 1e-3,
    device: str = "cpu",
    use_mlflow: bool = True,
) -> PlayerAutoencoder:
    """Train the autoencoder and optionally log metrics to MLflow.

    Parameters
    ----------
    model:
        The ``PlayerAutoencoder`` instance to train.
    X_train:
        Scaled training feature array (n_players × input_dim).
    X_val:
        Optional validation array.  If provided, val loss is tracked.
    epochs:
        Number of training epochs.
    batch_size:
        Mini-batch size.
    lr:
        Adam learning rate.
    device:
        ``"cpu"`` or ``"cuda"``.
    use_mlflow:
        Whether to log parameters and metrics to MLflow.

    Returns
    -------
    PlayerAutoencoder
        The trained model (on CPU).
    """
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    X_t = torch.tensor(X_train, dtype=torch.float32)
    loader = DataLoader(TensorDataset(X_t), batch_size=batch_size, shuffle=True)

    X_v = torch.tensor(X_val, dtype=torch.float32).to(device) if X_val is not None else None

    if use_mlflow:
        mlflow.log_params(
            {
                "epochs": epochs,
                "batch_size": batch_size,
                "lr": lr,
                "input_dim": model.encoder.net[0].in_features,
                "latent_dim": model.encoder.net[-1].out_features,
                "n_train": len(X_train),
            }
        )

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for (batch,) in loader:
            batch = batch.to(device)
            x_hat, _ = model(batch)
            loss = criterion(x_hat, batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(batch)

        train_loss = total_loss / len(X_train)

        log_dict: dict[str, float] = {"train_loss": train_loss}
        if X_v is not None:
            model.eval()
            with torch.no_grad():
                x_hat_val, _ = model(X_v)
                val_loss = criterion(x_hat_val, X_v).item()
            log_dict["val_loss"] = val_loss

        if use_mlflow:
            mlflow.log_metrics(log_dict, step=epoch)

        if epoch % 10 == 0 or epoch == epochs:
            val_str = f"  val_loss={log_dict.get('val_loss', float('nan')):.6f}" if X_v is not None else ""
            logger.info("Epoch %d/%d  train_loss=%.6f%s", epoch, epochs, train_loss, val_str)

    return model.to("cpu")


# ── persistence ───────────────────────────────────────────────────────────────

def save_model(model: PlayerAutoencoder, path: Optional[Path] = None) -> Path:
    """Save model state dict to disk."""
    if path is None:
        path = MODEL_DIR / "autoencoder.pt"
    torch.save(model.state_dict(), path)
    logger.info("Saved model → %s", path)
    return path


def load_model(
    path: Optional[Path] = None,
    input_dim: int = INPUT_DIM,
    latent_dim: int = LATENT_DIM,
) -> PlayerAutoencoder:
    """Load a saved model from disk."""
    if path is None:
        path = MODEL_DIR / "autoencoder.pt"
    model = PlayerAutoencoder(input_dim=input_dim, latent_dim=latent_dim)
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    model.eval()
    return model


# ── CLI entry-point ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    parser = argparse.ArgumentParser(description="Train player autoencoder.")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--latent-dim", type=int, default=LATENT_DIM)
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--no-mlflow", action="store_true", help="Disable MLflow logging.")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    features_df = load_features()
    X, scaler = scale_features(features_df)

    # Train / val split
    n_val = max(1, int(len(X) * args.val_split))
    idx = np.random.permutation(len(X))
    X_train, X_val = X[idx[n_val:]], X[idx[:n_val]]

    model = PlayerAutoencoder(input_dim=INPUT_DIM, latent_dim=args.latent_dim)

    tracking_uri = str(ROOT / "mlflow")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("scouting-autoencoder")

    use_mlflow = not args.no_mlflow
    with (mlflow.start_run() if use_mlflow else contextlib.nullcontext()):
        trained = train(
            model,
            X_train,
            X_val=X_val,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            use_mlflow=use_mlflow,
        )
        out_path = save_model(trained, path=args.output)
        if use_mlflow:
            mlflow.log_artifact(str(out_path))
