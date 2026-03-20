"""
Tests for src/model.py — autoencoder architecture, training, and persistence.
No real data required; all inputs are synthetic tensors.
"""

import numpy as np
import pytest
import torch

from src.model import PlayerAutoencoder, load_model, save_model, train


# ── architecture ───────────────────────────────────────────────────────────────

def test_forward_pass_shapes():
    model = PlayerAutoencoder()
    x = torch.randn(8, 30)
    x_hat, z = model(x)
    assert x_hat.shape == (8, 30)
    assert z.shape == (8, 8)


def test_encode_shape():
    model = PlayerAutoencoder()
    x = torch.randn(5, 30)
    z = model.encode(x)
    assert z.shape == (5, 8)


def test_custom_dims():
    model = PlayerAutoencoder(input_dim=20, latent_dim=4)
    x = torch.randn(3, 20)
    x_hat, z = model(x)
    assert x_hat.shape == (3, 20)
    assert z.shape == (3, 4)


def test_single_player_forward():
    """Batch size of 1 should work (BatchNorm eval mode handles this)."""
    model = PlayerAutoencoder()
    model.eval()
    x = torch.randn(1, 30)
    with torch.no_grad():
        x_hat, z = model(x)
    assert x_hat.shape == (1, 30)
    assert z.shape == (1, 8)


# ── persistence ────────────────────────────────────────────────────────────────

def test_save_load_roundtrip(tmp_path):
    model = PlayerAutoencoder()
    model.eval()  # eval mode so BatchNorm uses running stats consistently
    path = save_model(model, tmp_path / "test.pt")
    assert path.exists()

    loaded = load_model(path)
    x = torch.randn(4, 30)
    with torch.no_grad():
        z1 = model.encode(x)
        z2 = loaded.encode(x)
    assert torch.allclose(z1, z2)


def test_load_model_missing_file(tmp_path):
    with pytest.raises(Exception):
        load_model(tmp_path / "nonexistent.pt")


# ── training ───────────────────────────────────────────────────────────────────

def test_train_runs():
    model = PlayerAutoencoder()
    X = np.random.default_rng(0).standard_normal((50, 30)).astype(np.float32)
    trained = train(model, X, epochs=3, use_mlflow=False)
    assert isinstance(trained, PlayerAutoencoder)


def test_train_with_validation():
    model = PlayerAutoencoder()
    rng = np.random.default_rng(1)
    X_train = rng.standard_normal((40, 30)).astype(np.float32)
    X_val = rng.standard_normal((10, 30)).astype(np.float32)
    trained = train(model, X_train, X_val=X_val, epochs=3, use_mlflow=False)
    assert isinstance(trained, PlayerAutoencoder)


def test_train_reduces_loss():
    """Loss after training should be lower than a random baseline."""
    model = PlayerAutoencoder()
    X = np.random.default_rng(2).standard_normal((80, 30)).astype(np.float32)
    criterion = torch.nn.MSELoss()

    X_t = torch.tensor(X)
    with torch.no_grad():
        x_hat_before, _ = model(X_t)
    loss_before = criterion(x_hat_before, X_t).item()

    trained = train(model, X, epochs=20, use_mlflow=False)

    trained.eval()
    with torch.no_grad():
        x_hat_after, _ = trained(X_t)
    loss_after = criterion(x_hat_after, X_t).item()

    assert loss_after < loss_before
