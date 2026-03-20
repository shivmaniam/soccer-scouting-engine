"""
Tests for src/features.py — feature column definitions, scaling, and per-90 normalisation.
Aggregation functions are tested with minimal synthetic event DataFrames.
"""

import numpy as np
import pandas as pd
import pytest

from src.features import (
    FEATURE_COLUMNS,
    _per_90,
    aggregate_defending,
    aggregate_passing,
    aggregate_style,
    scale_features,
)


# ── FEATURE_COLUMNS contract ───────────────────────────────────────────────────

def test_feature_columns_count():
    assert len(FEATURE_COLUMNS) == 30


def test_feature_columns_no_duplicates():
    assert len(FEATURE_COLUMNS) == len(set(FEATURE_COLUMNS))


def test_feature_columns_all_lowercase():
    for col in FEATURE_COLUMNS:
        assert col == col.lower(), f"Column {col!r} contains uppercase characters"


# ── scale_features ─────────────────────────────────────────────────────────────

def make_player_features(n: int = 20) -> pd.DataFrame:
    """Synthetic player features DataFrame matching the expected schema."""
    rng = np.random.default_rng(42)
    data = {col: rng.uniform(0, 5, n) for col in FEATURE_COLUMNS}
    data["player_name"] = [f"Player {i}" for i in range(n)]
    data["team"] = "Team A"
    data["total_minutes"] = rng.integers(200, 2000, n).astype(float)
    df = pd.DataFrame(data)
    df.index.name = "player_id"
    return df


def test_scale_features_output_shape():
    df = make_player_features(15)
    X, scaler = scale_features(df)
    assert X.shape == (15, 30)


def test_scale_features_standardised():
    df = make_player_features(100)
    X, _ = scale_features(df)
    # After StandardScaler each column has mean ≈ 0 and std ≈ 1
    assert np.abs(X.mean(axis=0)).max() < 1e-6
    assert np.abs(X.std(axis=0) - 1).max() < 0.02


def test_scale_features_returns_scaler():
    from sklearn.preprocessing import StandardScaler
    df = make_player_features(10)
    _, scaler = scale_features(df)
    assert isinstance(scaler, StandardScaler)


# ── _per_90 ────────────────────────────────────────────────────────────────────

def test_per_90_halves_stats_for_double_minutes():
    """A player with 180 minutes should have half the p90 rate of one with 90 minutes,
    given the same raw count."""
    df = pd.DataFrame({
        "total_minutes": [90.0, 180.0],
        "goals": [2.0, 2.0],
    })
    result = _per_90(df, minutes_col="total_minutes")
    # goals_p90: player 0 = 2/(90/90)=2.0, player 1 = 2/(180/90)=1.0
    assert "goals_p90" in result.columns
    assert pytest.approx(result["goals_p90"].iloc[0]) == 2.0
    assert pytest.approx(result["goals_p90"].iloc[1]) == 1.0


def test_per_90_drops_original_counting_column():
    df = pd.DataFrame({
        "total_minutes": [90.0],
        "goals": [1.0],
    })
    result = _per_90(df, minutes_col="total_minutes")
    assert "goals" not in result.columns
    assert "goals_p90" in result.columns


# ── aggregate_defending ────────────────────────────────────────────────────────

def make_defending_events(n_players: int = 3, n_matches: int = 2) -> pd.DataFrame:
    """Minimal synthetic event DataFrame for defending aggregation tests."""
    rows = []
    for pid in range(n_players):
        for mid in range(n_matches):
            for event_type in ("Interception", "Clearance", "Block"):
                rows.append({"type_name": event_type, "player_id": pid, "match_id": mid})
    return pd.DataFrame(rows)


def test_aggregate_defending_column_names_are_lowercase():
    events = make_defending_events()
    result = aggregate_defending(events)
    for col in result.columns:
        assert col == col.lower(), f"Column {col!r} is not lowercase"


def test_aggregate_defending_contains_expected_columns():
    events = make_defending_events()
    result = aggregate_defending(events)
    for col in ("interceptions", "clearances", "blocks"):
        assert col in result.columns, f"Expected column {col!r} missing"


# ── aggregate_passing ──────────────────────────────────────────────────────────

def make_passing_events(n_players: int = 3) -> pd.DataFrame:
    """Minimal synthetic passing events."""
    rows = []
    for pid in range(n_players):
        rows.append({
            "type_name": "Pass",
            "player_id": pid,
            "match_id": 0,
            "pass_outcome_name": None,   # None = completed pass
            "pass_length": 10.0,
            "pass_angle": 0.1,
            "pass_switch": False,
            "pass_goal_assist": False,
            "pass_cross": False,
            "pass_technique_name": None,
            "pass_end_location": [50.0, 30.0],
            "location": [40.0, 30.0],
        })
    return pd.DataFrame(rows)


def test_aggregate_passing_returns_dataframe():
    events = make_passing_events()
    result = aggregate_passing(events)
    assert isinstance(result, pd.DataFrame)
    assert "player_id" in result.columns
    assert "match_id" in result.columns


# ── aggregate_style ────────────────────────────────────────────────────────────

def make_style_events(n_players: int = 3) -> pd.DataFrame:
    rows = []
    for pid in range(n_players):
        rows.append({"type_name": "Carry", "player_id": pid, "match_id": 0})
        rows.append({"type_name": "Ball Receipt*", "player_id": pid, "match_id": 0})
    return pd.DataFrame(rows)


def test_aggregate_style_returns_dataframe():
    events = make_style_events()
    result = aggregate_style(events)
    assert isinstance(result, pd.DataFrame)
    assert "carries" in result.columns
    assert "ball_receipts" in result.columns
