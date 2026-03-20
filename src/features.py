"""
features.py
-----------
Build per-90-minute normalised feature vectors for every player from raw
StatsBomb event data.

Feature groups (30 features total):
  Attacking  (8) — goals, shots, xG, xA, key passes, dribbles completed,
                   touches in box, progressive carries
  Passing    (8) — pass completion %, total passes, forward passes %,
                   long ball %, crosses, through balls, switches of play,
                   progressive passes
  Defending  (7) — tackles won, interceptions, clearances, blocks,
                   pressures, pressure success %, duel win %
  Style      (7) — receptions, carries, dispossessed, fouls won,
                   fouls committed, yellow cards, completed passes

All counting stats are normalised to per-90 minutes using actual playing time
estimated from lineups / substitution events.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
FEATURES_DIR = ROOT / "data"
FEATURES_DIR.mkdir(parents=True, exist_ok=True)


# ── playing-time estimation ────────────────────────────────────────────────────

def _extract_substitution_minutes(events: pd.DataFrame) -> pd.DataFrame:
    """Parse substitution events to determine when players left the pitch.

    The player listed on the substitution row is the one going *off*; their
    minute_off equals the event minute.

    Returns
    -------
    pd.DataFrame
        Columns: match_id, player_id, minute_off
    """
    type_col = events["type"].apply(
        lambda t: t.get("name") if isinstance(t, dict) else str(t)
    )
    subs = events[type_col == "Substitution"].copy()

    if subs.empty:
        return pd.DataFrame(columns=["match_id", "player_id", "minute_off"])

    # player going OFF: prefer flat player_id column, fall back to nested dict
    if "player_id" in subs.columns:
        pid_series = subs["player_id"]
    else:
        pid_series = subs["player"].apply(
            lambda p: p.get("id") if isinstance(p, dict) else p
        )

    return pd.DataFrame(
        {
            "match_id": subs["match_id"].values,
            "player_id": pid_series.values,
            "minute_off": subs["minute"].values,
        }
    )


def estimate_minutes_played(
    events: pd.DataFrame,
    lineups: pd.DataFrame,
) -> pd.DataFrame:
    """Estimate minutes played per player per match.

    Logic:
      - Every player in the starting XI plays from minute 0.
      - Substitutes play from the minute they enter.
      - Players subbed off stop at their substitution minute.
      - Otherwise they play until the last recorded event minute (≈ full game).

    Parameters
    ----------
    events:
        Full events DataFrame (all matches).
    lineups:
        Full lineups DataFrame with columns player_id, match_id, team,
        and optionally ``positions`` (list of position dicts).

    Returns
    -------
    pd.DataFrame
        Columns: player_id, player_name, team, match_id, minutes_played
    """
    # Last event minute per match — proxy for match duration
    match_duration = (
        events.groupby("match_id")["minute"].max().rename("duration").reset_index()
    )

    # Substitution minutes-off
    subs_off = _extract_substitution_minutes(events)

    # Build a flat lineup frame: one row per (match_id, player_id)
    lineup_cols = ["match_id", "player_id", "player_name", "team"]
    # statsbombpy may store player info nested; normalise
    if "player_id" not in lineups.columns and "player" in lineups.columns:
        lineups = lineups.copy()
        lineups["player_id"] = lineups["player"].apply(
            lambda p: p.get("id") if isinstance(p, dict) else p
        )
        lineups["player_name"] = lineups["player"].apply(
            lambda p: p.get("name") if isinstance(p, dict) else None
        )

    base = lineups[[c for c in lineup_cols if c in lineups.columns]].drop_duplicates()

    # Merge duration
    base = base.merge(match_duration, on="match_id", how="left")

    # Determine kick-off minute: starters = 0; subs = their entry minute
    # Substitution-on events carry the replacement player id
    sub_on_events = events[events["type"].apply(
        lambda t: t.get("name") == "Substitution" if isinstance(t, dict) else t == "Substitution"
    )].copy()

    entry_minutes: dict[tuple[int, int], int] = {}
    for _, row in sub_on_events.iterrows():
        # With flatten_attrs=True the replacement player id is a flat column;
        # fall back to the nested dict for older statsbombpy versions.
        rep_id = row.get("substitution_replacement_id")
        if rep_id is None:
            sub_detail = row.get("substitution", {})
            if isinstance(sub_detail, dict):
                replacement = sub_detail.get("replacement", {})
                rep_id = replacement.get("id") if isinstance(replacement, dict) else None
        if rep_id is not None:
            entry_minutes[(int(row["match_id"]), int(rep_id))] = int(row["minute"])

    base["minute_on"] = base.apply(
        lambda r: entry_minutes.get((int(r["match_id"]), int(r["player_id"])), 0),
        axis=1,
    )

    # Merge substitution-off minutes
    base = base.merge(subs_off, on=["match_id", "player_id"], how="left")
    base["minute_off"] = base["minute_off"].fillna(base["duration"])

    base["minutes_played"] = (base["minute_off"] - base["minute_on"]).clip(lower=0)

    return base[["player_id", "player_name", "team", "match_id", "minutes_played"]]


# ── raw event-level aggregations ───────────────────────────────────────────────

def _safe_name(val: object) -> str:
    """Extract a name string from a StatsBomb type dict or plain string."""
    if isinstance(val, dict):
        return val.get("name", "")
    return str(val) if val is not None else ""


def _col(df: pd.DataFrame, name: str, fill: object = None) -> pd.Series:
    """Return a DataFrame column or a constant-filled Series if the column is absent.

    statsbombpy with ``flatten_attrs=True`` (default) expands nested event
    dicts (``shot``, ``pass``, ``carry``, …) into flat columns such as
    ``shot_statsbomb_xg``, ``pass_length``, ``carry_end_location``.  This
    helper keeps aggregation code concise and safe when those columns are
    missing (e.g. the events DataFrame contains no shots at all).
    """
    if name in df.columns:
        return df[name]
    dtype = type(fill) if fill is not None else object
    return pd.Series(fill, index=df.index, dtype=dtype)


def _extract_player_id(df: pd.DataFrame) -> pd.Series:
    """Return a player_id Series, handling both flat and nested player columns."""
    if "player_id" in df.columns:
        return df["player_id"]
    return df["player"].apply(lambda p: p.get("id") if isinstance(p, dict) else p)


def _extract_type_name(df: pd.DataFrame) -> pd.Series:
    """Return a type_name Series from the StatsBomb ``type`` column."""
    if "type_name" in df.columns:
        return df["type_name"]
    return df["type"].apply(_safe_name)


def _loc_x(loc: object) -> float:
    """Safely extract the x-coordinate from a StatsBomb location (list/array)."""
    try:
        return float(loc[0])
    except (TypeError, IndexError):
        return 0.0


def _in_box(location: object) -> bool:
    """Return True if the location is inside the penalty box (StatsBomb 120×80 pitch)."""
    try:
        x, y = float(location[0]), float(location[1])
        return x >= 102 and 18 <= y <= 62
    except (TypeError, IndexError):
        return False


def aggregate_attacking(events: pd.DataFrame) -> pd.DataFrame:
    """Aggregate attacking counting stats per (player_id, match_id).

    Uses flat column names from statsbombpy ``flatten_attrs=True``:
    ``shot_statsbomb_xg``, ``shot_outcome`` ("Goal"/"Saved"/…),
    ``pass_goal_assist``, ``pass_shot_assist``, ``dribble_outcome``
    ("Complete"/"Incomplete"), ``carry_end_location``.
    """
    df = events.copy()
    df["_type"] = _extract_type_name(df)
    df["player_id"] = _extract_player_id(df)

    shots = df[df["_type"] == "Shot"].copy()
    passes = df[df["_type"] == "Pass"].copy()
    dribbles = df[df["_type"] == "Dribble"].copy()
    carries = df[df["_type"] == "Carry"].copy()

    # ── shots ──────────────────────────────────────────────────────────────────
    shots["xg_val"] = _col(shots, "shot_statsbomb_xg", 0.0).fillna(0.0)
    # shot_outcome is a plain string: "Goal", "Saved", "Off T", "Blocked", …
    shots["is_goal"] = _col(shots, "shot_outcome", "").fillna("") == "Goal"

    # ── passes ─────────────────────────────────────────────────────────────────
    passes["xa_val"] = _col(passes, "pass_xa", 0.0).fillna(0.0)
    goal_assist = _col(passes, "pass_goal_assist", False).fillna(False).astype(bool)
    shot_assist = _col(passes, "pass_shot_assist", False).fillna(False).astype(bool)
    passes["is_key_pass"] = goal_assist | shot_assist

    # ── dribbles ───────────────────────────────────────────────────────────────
    # dribble_outcome is a plain string: "Complete" or "Incomplete"
    dribbles["dribble_complete"] = (
        _col(dribbles, "dribble_outcome", "").fillna("") == "Complete"
    )

    # ── in-box touches ─────────────────────────────────────────────────────────
    shots["touch_in_box"] = shots["location"].apply(_in_box)
    passes["touch_in_box"] = passes["location"].apply(_in_box)
    carries["touch_in_box"] = _col(carries, "carry_end_location").apply(_in_box)

    # ── progressive carries (end_x ≥ start_x + 10) ───────────────────────────
    def _progressive(row: pd.Series) -> bool:
        try:
            return bool(float(row["carry_end_location"][0]) - float(row["location"][0]) >= 10)
        except (TypeError, IndexError, KeyError):
            return False

    if "carry_end_location" in carries.columns:
        carries["is_progressive"] = carries.apply(_progressive, axis=1)
    else:
        carries["is_progressive"] = False

    # ── aggregate ──────────────────────────────────────────────────────────────
    def _agg(frame: pd.DataFrame, cols: dict[str, str]) -> pd.DataFrame:
        frame = frame.copy()
        for col in cols:
            if col not in frame.columns:
                frame[col] = 0
        return (
            frame.groupby(["player_id", "match_id"])[list(cols.keys())]
            .sum()
            .rename(columns=cols)
            .reset_index()
        )

    shot_agg = _agg(shots, {"is_goal": "goals", "xg_val": "xg", "touch_in_box": "shots_in_box"})
    shot_count = shots.groupby(["player_id", "match_id"]).size().rename("shots").reset_index()
    pass_agg = _agg(passes, {"xa_val": "xa", "is_key_pass": "key_passes", "touch_in_box": "passes_in_box"})
    dribble_agg = _agg(dribbles, {"dribble_complete": "dribbles_completed"})
    carry_agg = _agg(carries, {"is_progressive": "progressive_carries", "touch_in_box": "carries_in_box"})

    merged = (
        shot_agg
        .merge(shot_count, on=["player_id", "match_id"], how="outer")
        .merge(pass_agg, on=["player_id", "match_id"], how="outer")
        .merge(dribble_agg, on=["player_id", "match_id"], how="outer")
        .merge(carry_agg, on=["player_id", "match_id"], how="outer")
        .fillna(0)
    )

    touch_cols = [c for c in ["shots_in_box", "passes_in_box", "carries_in_box"] if c in merged.columns]
    merged["touches_in_box"] = merged[touch_cols].sum(axis=1)
    merged.drop(columns=touch_cols, inplace=True)

    return merged


def aggregate_passing(events: pd.DataFrame) -> pd.DataFrame:
    """Aggregate passing stats per (player_id, match_id).

    Flat column names used: ``pass_length``, ``pass_angle``, ``pass_cross``,
    ``pass_switch``, ``pass_technique_name``, ``pass_outcome_name``,
    ``pass_end_location``.
    """
    df = events.copy()
    df["_type"] = _extract_type_name(df)
    df["player_id"] = _extract_player_id(df)

    passes = df[df["_type"] == "Pass"].copy()

    # Completion: successful passes have NaN outcome; failed ones are "Incomplete"/"Out"/…
    # pass_outcome is a plain string (not a dict)
    outcome = _col(passes, "pass_outcome", "").fillna("")
    passes["is_complete"] = outcome == ""

    length = _col(passes, "pass_length", 0.0).fillna(0.0)
    passes["is_long"] = length >= 32  # ~35 yards

    angle = _col(passes, "pass_angle", 0.0).fillna(0.0)
    passes["is_forward"] = angle.apply(lambda a: abs(a) < (np.pi / 2))

    passes["is_cross"] = _col(passes, "pass_cross", False).fillna(False).astype(bool)
    passes["is_switch"] = _col(passes, "pass_switch", False).fillna(False).astype(bool)

    # Through ball: pass_technique is a plain string, or pass_through_ball is a bool column
    through_ball_flag = _col(passes, "pass_through_ball", False).fillna(False).astype(bool)
    tech = _col(passes, "pass_technique", "").fillna("")
    passes["is_through_ball"] = through_ball_flag | (tech == "Through Ball")

    # Progressive: end_x − start_x ≥ 10 yards (locations are numpy arrays)
    end_x = _col(passes, "pass_end_location").apply(_loc_x)
    start_x = passes["location"].apply(_loc_x)
    passes["is_progressive"] = (end_x - start_x) >= 10

    grp = passes.groupby(["player_id", "match_id"])
    agg = grp.agg(
        total_passes=("is_complete", "count"),
        completed_passes=("is_complete", "sum"),
        long_passes=("is_long", "sum"),
        forward_passes=("is_forward", "sum"),
        crosses=("is_cross", "sum"),
        through_balls=("is_through_ball", "sum"),
        switches=("is_switch", "sum"),
        progressive_passes=("is_progressive", "sum"),
    ).reset_index()

    # NOTE: percentage stats are NOT computed here because this function returns
    # per-match counts that get summed across all a player's matches in
    # build_player_features().  Computing ratios before the sum would give
    # nonsensical values (e.g. 10 matches × 80% = 800%).  The rates are
    # computed from the career totals in build_player_features() instead.

    return agg


def aggregate_defending(events: pd.DataFrame) -> pd.DataFrame:
    """Aggregate defensive stats per (player_id, match_id).

    Flat column names used: ``duel_outcome_name``,
    ``counterpressure`` (pressure success proxy).
    """
    df = events.copy()
    df["_type"] = _extract_type_name(df)
    df["player_id"] = _extract_player_id(df)

    def _count(type_name: str) -> pd.DataFrame:
        sub = df[df["_type"] == type_name]
        return sub.groupby(["player_id", "match_id"]).size().rename(type_name).reset_index()

    # Tackles: StatsBomb uses Duel events with duel_type == "Tackle".
    # duel_type is a plain string (e.g. "Tackle", "Aerial Lost").
    tackle_mask = (df["_type"] == "Tackle") | (
        (df["_type"] == "Duel")
        & (_col(df, "duel_type", "").fillna("") == "Tackle")
    )
    tackles = df[tackle_mask].copy()
    # duel_outcome is a plain string: "Won", "Lost", "Success In Play", etc.
    tackle_outcome = _col(tackles, "duel_outcome", "").fillna("")
    tackles["won"] = tackle_outcome.isin(["Won", "Success In Play", "Success Out"])
    tackle_agg = tackles.groupby(["player_id", "match_id"]).agg(
        tackles_total=("won", "count"),
        tackles_won=("won", "sum"),
    ).reset_index()

    # Pressures: count all; use counterpressure flag as a "won" proxy
    pressures = df[df["_type"] == "Pressure"].copy()
    pressures["success"] = _col(pressures, "counterpressure", False).fillna(False).astype(bool)
    pressure_agg = pressures.groupby(["player_id", "match_id"]).agg(
        pressures=("success", "count"),
        pressures_success=("success", "sum"),
    ).reset_index()

    # Duels (aerial + ground)
    duel = df[df["_type"] == "Duel"].copy()
    duel_outcome = _col(duel, "duel_outcome", "").fillna("")
    duel["won"] = duel_outcome.isin(["Won", "Success In Play", "Success Out"])
    duel_agg = duel.groupby(["player_id", "match_id"]).agg(
        duels=("won", "count"),
        duels_won=("won", "sum"),
    ).reset_index()

    interceptions = _count("Interception").rename(columns={"Interception": "interceptions"})
    clearances = _count("Clearance").rename(columns={"Clearance": "clearances"})
    blocks = _count("Block").rename(columns={"Block": "blocks"})

    merged = (
        tackle_agg
        .merge(pressure_agg, on=["player_id", "match_id"], how="outer")
        .merge(duel_agg, on=["player_id", "match_id"], how="outer")
        .merge(interceptions, on=["player_id", "match_id"], how="outer")
        .merge(clearances, on=["player_id", "match_id"], how="outer")
        .merge(blocks, on=["player_id", "match_id"], how="outer")
        .fillna(0)
    )

    # NOTE: pressure_success_pct and duel_win_pct are computed from career
    # totals in build_player_features() — not here — to avoid the sum-of-ratios bug.

    return merged


def aggregate_style(events: pd.DataFrame) -> pd.DataFrame:
    """Aggregate style/misc stats per (player_id, match_id).

    Flat column name used: ``foul_committed_card_name``.
    """
    df = events.copy()
    df["_type"] = _extract_type_name(df)
    df["player_id"] = _extract_player_id(df)

    def _count(type_name: str, col_name: str) -> pd.DataFrame:
        sub = df[df["_type"] == type_name]
        return (
            sub.groupby(["player_id", "match_id"])
            .size()
            .rename(col_name)
            .reset_index()
        )

    receipts = _count("Ball Receipt*", "ball_receipts")
    carries = _count("Carry", "carries")

    misc_agg = (
        df[df["_type"] == "Miscontrol"]
        .groupby(["player_id", "match_id"]).size()
        .rename("dispossessed").reset_index()
    )

    fouls = df[df["_type"] == "Foul Committed"].copy()
    fouls_won = df[df["_type"] == "Foul Won"].copy()

    # Yellow cards: foul_committed_card is a plain string: "Yellow Card", "Red Card"
    card_name = _col(fouls, "foul_committed_card", "").fillna("")
    yellow = fouls[card_name == "Yellow Card"]

    foul_agg = fouls.groupby(["player_id", "match_id"]).size().rename("fouls_committed").reset_index()
    foul_won_agg = fouls_won.groupby(["player_id", "match_id"]).size().rename("fouls_won").reset_index()
    yellow_agg = yellow.groupby(["player_id", "match_id"]).size().rename("yellow_cards").reset_index()

    merged = (
        receipts
        .merge(carries, on=["player_id", "match_id"], how="outer")
        .merge(misc_agg, on=["player_id", "match_id"], how="outer")
        .merge(foul_agg, on=["player_id", "match_id"], how="outer")
        .merge(foul_won_agg, on=["player_id", "match_id"], how="outer")
        .merge(yellow_agg, on=["player_id", "match_id"], how="outer")
        .fillna(0)
    )

    return merged


# ── per-90 normalisation ───────────────────────────────────────────────────────

# Columns that should be expressed as rates (%), not per-90 counting stats
RATE_COLUMNS = {
    "pass_completion_pct",
    "long_ball_pct",
    "forward_pass_pct",
    "pressure_success_pct",
    "duel_win_pct",
}

# Final ordered list of 30 feature columns
FEATURE_COLUMNS = [
    # Attacking (8)
    "goals_p90",
    "shots_p90",
    "xg_p90",
    "xa_p90",
    "key_passes_p90",
    "dribbles_completed_p90",
    "touches_in_box_p90",
    "progressive_carries_p90",
    # Passing (8)
    "pass_completion_pct",
    "total_passes_p90",
    "forward_pass_pct",
    "long_ball_pct",
    "crosses_p90",
    "through_balls_p90",
    "switches_p90",
    "progressive_passes_p90",
    # Defending (7)
    "tackles_won_p90",
    "interceptions_p90",
    "clearances_p90",
    "blocks_p90",
    "pressures_p90",
    "pressure_success_pct",
    "duel_win_pct",
    # Style (7)
    "ball_receipts_p90",
    "carries_p90",
    "dispossessed_p90",
    "fouls_won_p90",
    "fouls_committed_p90",
    "yellow_cards_p90",
    # (30th feature) total passes kept for completeness — swap if needed
    "completed_passes_p90",
]


def _per_90(df: pd.DataFrame, minutes_col: str = "total_minutes") -> pd.DataFrame:
    """Convert counting stats to per-90 values in-place.

    Columns in ``RATE_COLUMNS`` are left untouched.
    All other numeric columns (except ids / minutes) are divided by
    (minutes / 90).

    Parameters
    ----------
    df:
        DataFrame with a ``total_minutes`` column and counting stats.
    minutes_col:
        Name of the playing-time column.

    Returns
    -------
    pd.DataFrame
        Same shape with counting columns replaced by per-90 equivalents.
    """
    result = df.copy()
    non_stat_cols = {"player_id", "player_name", "team", minutes_col}
    numeric_cols = result.select_dtypes(include=[np.number]).columns.tolist()

    for col in numeric_cols:
        if col in non_stat_cols or col in RATE_COLUMNS:
            continue
        per90_name = col if col.endswith("_p90") else f"{col}_p90"
        result[per90_name] = np.where(
            result[minutes_col] > 0,
            result[col] / (result[minutes_col] / 90),
            0.0,
        )
        if not col.endswith("_p90"):
            result.drop(columns=[col], inplace=True)

    return result


# ── main pipeline ──────────────────────────────────────────────────────────────

def build_player_features(
    events: pd.DataFrame,
    lineups: pd.DataFrame,
    min_minutes: int = 180,
) -> pd.DataFrame:
    """Build the final per-90 normalised feature matrix.

    Parameters
    ----------
    events:
        Raw events DataFrame (one or more competitions merged together).
    lineups:
        Raw lineups DataFrame corresponding to the same matches.
    min_minutes:
        Minimum total playing time (minutes) to include a player.
        Players with fewer minutes are dropped (too small a sample).

    Returns
    -------
    pd.DataFrame
        Index: player_id.  Columns: ``FEATURE_COLUMNS`` + metadata columns
        (player_name, team, total_minutes).
    """
    logger.info("Estimating minutes played …")
    minutes_df = estimate_minutes_played(events, lineups)

    # Aggregate per (player, match) across all feature groups
    logger.info("Aggregating attacking features …")
    atk = aggregate_attacking(events)

    logger.info("Aggregating passing features …")
    pas = aggregate_passing(events)

    logger.info("Aggregating defending features …")
    dfs = aggregate_defending(events)

    logger.info("Aggregating style features …")
    sty = aggregate_style(events)

    # Merge all feature groups on (player_id, match_id)
    merged = (
        atk
        .merge(pas, on=["player_id", "match_id"], how="outer")
        .merge(dfs, on=["player_id", "match_id"], how="outer")
        .merge(sty, on=["player_id", "match_id"], how="outer")
        .fillna(0)
    )

    # Merge playing time
    merged = merged.merge(
        minutes_df[["player_id", "player_name", "team", "match_id", "minutes_played"]],
        on=["player_id", "match_id"],
        how="left",
    )
    merged["minutes_played"] = merged["minutes_played"].fillna(0)

    # Sum across all matches to get career totals
    meta_cols = ["player_id", "player_name", "team"]
    stat_cols = [c for c in merged.columns if c not in meta_cols + ["match_id"]]

    player_totals = (
        merged.groupby(meta_cols, as_index=False)[stat_cols].sum()
    )
    player_totals.rename(columns={"minutes_played": "total_minutes"}, inplace=True)

    # Compute rate stats from career totals (must happen AFTER summing across matches
    # so the denominators reflect total volume, not a sum of per-match percentages).
    player_totals["pass_completion_pct"] = np.where(
        player_totals.get("total_passes", pd.Series(0)) > 0,
        player_totals["completed_passes"] / player_totals["total_passes"] * 100,
        0.0,
    )
    player_totals["long_ball_pct"] = np.where(
        player_totals.get("total_passes", pd.Series(0)) > 0,
        player_totals.get("long_passes", 0) / player_totals["total_passes"] * 100,
        0.0,
    )
    player_totals["forward_pass_pct"] = np.where(
        player_totals.get("total_passes", pd.Series(0)) > 0,
        player_totals.get("forward_passes", 0) / player_totals["total_passes"] * 100,
        0.0,
    )
    player_totals["pressure_success_pct"] = np.where(
        player_totals.get("pressures", pd.Series(0)) > 0,
        player_totals.get("pressures_success", 0) / player_totals["pressures"] * 100,
        0.0,
    )
    player_totals["duel_win_pct"] = np.where(
        player_totals.get("duels", pd.Series(0)) > 0,
        player_totals.get("duels_won", 0) / player_totals["duels"] * 100,
        0.0,
    )

    # Drop players below minimum minute threshold
    before = len(player_totals)
    player_totals = player_totals[player_totals["total_minutes"] >= min_minutes]
    logger.info(
        "Dropped %d players with < %d minutes; %d remain.",
        before - len(player_totals),
        min_minutes,
        len(player_totals),
    )

    # Convert counting stats to per-90
    logger.info("Normalising to per-90 …")
    player_features = _per_90(player_totals, minutes_col="total_minutes")

    # Keep only the defined feature columns (fill missing with 0)
    for col in FEATURE_COLUMNS:
        if col not in player_features.columns:
            player_features[col] = 0.0

    output_cols = meta_cols + ["total_minutes"] + FEATURE_COLUMNS
    player_features = player_features[[c for c in output_cols if c in player_features.columns]]

    logger.info("Feature matrix shape: %s", player_features.shape)
    return player_features.set_index("player_id")


def scale_features(
    player_features: pd.DataFrame,
    scaler: Optional[StandardScaler] = None,
) -> tuple[np.ndarray, StandardScaler]:
    """Z-score normalise the feature matrix.

    Parameters
    ----------
    player_features:
        Output of ``build_player_features``.
    scaler:
        A pre-fitted ``StandardScaler`` to apply (e.g. loaded from disk).
        If *None*, a new scaler is fitted on ``player_features``.

    Returns
    -------
    tuple[np.ndarray, StandardScaler]
        Scaled feature array (n_players × 30) and the fitted scaler.
    """
    X = player_features[FEATURE_COLUMNS].values.astype(np.float32)
    if scaler is None:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
    else:
        X_scaled = scaler.transform(X)
    return X_scaled.astype(np.float32), scaler


def save_features(
    player_features: pd.DataFrame,
    path: Optional[Path] = None,
) -> Path:
    """Persist the feature matrix to Parquet.

    Parameters
    ----------
    player_features:
        Output of ``build_player_features``.
    path:
        Destination path.  Defaults to ``data/player_features.parquet``.

    Returns
    -------
    Path
        The written file path.
    """
    if path is None:
        path = FEATURES_DIR / "player_features.parquet"
    player_features.to_parquet(path)
    logger.info("Saved feature matrix (%d players) → %s", len(player_features), path)
    return path


def load_features(path: Optional[Path] = None) -> pd.DataFrame:
    """Load a previously saved feature matrix.

    Parameters
    ----------
    path:
        Source path.  Defaults to ``data/player_features.parquet``.

    Returns
    -------
    pd.DataFrame

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    """
    if path is None:
        path = FEATURES_DIR / "player_features.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"Feature file not found: {path}\n"
            "Run build_player_features() and save_features() first."
        )
    return pd.read_parquet(path)


# ── CLI entry-point ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    from src.ingest import load_events, load_lineups

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    parser = argparse.ArgumentParser(description="Build player feature vectors.")
    parser.add_argument("--competition-id", type=int, required=True)
    parser.add_argument("--season-id", type=int, required=True)
    parser.add_argument(
        "--min-minutes",
        type=int,
        default=180,
        help="Minimum playing time to include a player (default: 180).",
    )
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    events = load_events(args.competition_id, args.season_id)
    lineups = load_lineups(args.competition_id, args.season_id)

    features = build_player_features(events, lineups, min_minutes=args.min_minutes)
    save_features(features, path=args.output)
    print(features[FEATURE_COLUMNS].describe().T.to_string())
