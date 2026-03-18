"""
ingest.py
---------
Pull StatsBomb open-data using statsbombpy and persist raw event + lineup
DataFrames to data/raw/.

StatsBomb open data covers:
  - La Liga (competition_id=11) — multiple seasons
  - Champions League (competition_id=16)
  - Women's World Cup (competition_id=72)
  - NWSL, FA Women's Super League, and more

All data is free; no API key is required.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

import pandas as pd
from statsbombpy import sb

logger = logging.getLogger(__name__)

# ── paths ──────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)


# ── low-level helpers ──────────────────────────────────────────────────────────

def get_competitions() -> pd.DataFrame:
    """Return the full StatsBomb competition catalogue.

    Returns
    -------
    pd.DataFrame
        One row per (competition, season) pair with columns:
        competition_id, season_id, competition_name, season_name, …
    """
    comps = sb.competitions()
    logger.info("Fetched %d competition-season pairs.", len(comps))
    return comps


def get_matches(competition_id: int, season_id: int) -> pd.DataFrame:
    """Fetch match-level metadata for one competition-season.

    Parameters
    ----------
    competition_id:
        StatsBomb competition identifier (e.g. 11 for La Liga).
    season_id:
        StatsBomb season identifier (e.g. 90 for 2020/21).

    Returns
    -------
    pd.DataFrame
        One row per match with match_id, home/away team, score, etc.
    """
    matches = sb.matches(competition_id=competition_id, season_id=season_id)
    logger.info(
        "Fetched %d matches for competition=%d season=%d.",
        len(matches),
        competition_id,
        season_id,
    )
    return matches


def get_events(match_id: int) -> pd.DataFrame:
    """Fetch all on-ball events for a single match.

    Parameters
    ----------
    match_id:
        StatsBomb match identifier.

    Returns
    -------
    pd.DataFrame
        One row per event (pass, shot, dribble, etc.) with full metadata.
    """
    events = sb.events(match_id=match_id)
    logger.info("Fetched %d events for match_id=%d.", len(events), match_id)
    return events


def get_lineups(match_id: int) -> pd.DataFrame:
    """Fetch lineup data (players + positions) for a single match.

    Parameters
    ----------
    match_id:
        StatsBomb match identifier.

    Returns
    -------
    pd.DataFrame
        One row per player appearance with player_id, player_name,
        team, position, etc.
    """
    raw = sb.lineups(match_id=match_id)
    # statsbombpy returns a dict {team_name: DataFrame}; normalise to one DF.
    frames = []
    for team_name, df in raw.items():
        df = df.copy()
        df["team"] = team_name
        df["match_id"] = match_id
        frames.append(df)
    lineups = pd.concat(frames, ignore_index=True)
    logger.info("Fetched lineups for match_id=%d (%d entries).", match_id, len(lineups))
    return lineups


# ── high-level pipeline ────────────────────────────────────────────────────────

def ingest_competition(
    competition_id: int,
    season_id: int,
    max_matches: Optional[int] = None,
    overwrite: bool = False,
) -> tuple[Path, Path]:
    """Download all events and lineups for a competition-season and save to disk.

    Files are written as Parquet to ``data/raw/``:
      - ``events_{competition_id}_{season_id}.parquet``
      - ``lineups_{competition_id}_{season_id}.parquet``

    Parameters
    ----------
    competition_id:
        StatsBomb competition identifier.
    season_id:
        StatsBomb season identifier.
    max_matches:
        If set, only the first *max_matches* matches are ingested (useful for
        quick smoke-tests).
    overwrite:
        Re-download even if the Parquet files already exist.

    Returns
    -------
    tuple[Path, Path]
        Paths to the saved events and lineups Parquet files.
    """
    events_path = RAW_DIR / f"events_{competition_id}_{season_id}.parquet"
    lineups_path = RAW_DIR / f"lineups_{competition_id}_{season_id}.parquet"

    if not overwrite and events_path.exists() and lineups_path.exists():
        logger.info(
            "Cache hit — skipping ingest for competition=%d season=%d.",
            competition_id,
            season_id,
        )
        return events_path, lineups_path

    matches = get_matches(competition_id, season_id)
    if max_matches is not None:
        matches = matches.head(max_matches)
        logger.info("Limiting ingest to %d matches.", max_matches)

    all_events: list[pd.DataFrame] = []
    all_lineups: list[pd.DataFrame] = []

    for _, row in matches.iterrows():
        mid = int(row["match_id"])
        try:
            all_events.append(get_events(mid))
            all_lineups.append(get_lineups(mid))
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to fetch match_id=%d: %s", mid, exc)

    events_df = pd.concat(all_events, ignore_index=True)
    lineups_df = pd.concat(all_lineups, ignore_index=True)

    events_df.to_parquet(events_path, index=False)
    lineups_df.to_parquet(lineups_path, index=False)

    logger.info(
        "Saved %d events → %s", len(events_df), events_path
    )
    logger.info(
        "Saved %d lineup entries → %s", len(lineups_df), lineups_path
    )
    return events_path, lineups_path


def ingest_all_open_data(
    max_matches_per_season: Optional[int] = None,
    overwrite: bool = False,
) -> list[tuple[Path, Path]]:
    """Ingest every competition-season available in StatsBomb open data.

    Parameters
    ----------
    max_matches_per_season:
        Cap per season (useful for CI / smoke-tests).
    overwrite:
        Force re-download of existing files.

    Returns
    -------
    list[tuple[Path, Path]]
        One (events_path, lineups_path) pair per competition-season processed.
    """
    comps = get_competitions()
    results: list[tuple[Path, Path]] = []

    for _, row in comps.iterrows():
        cid = int(row["competition_id"])
        sid = int(row["season_id"])
        logger.info(
            "Ingesting %s — %s (competition=%d season=%d).",
            row["competition_name"],
            row["season_name"],
            cid,
            sid,
        )
        paths = ingest_competition(
            cid,
            sid,
            max_matches=max_matches_per_season,
            overwrite=overwrite,
        )
        results.append(paths)

    return results


def load_events(competition_id: int, season_id: int) -> pd.DataFrame:
    """Load a previously-ingested events Parquet file.

    Parameters
    ----------
    competition_id:
        StatsBomb competition identifier.
    season_id:
        StatsBomb season identifier.

    Returns
    -------
    pd.DataFrame

    Raises
    ------
    FileNotFoundError
        If the Parquet file does not exist (run ingest first).
    """
    path = RAW_DIR / f"events_{competition_id}_{season_id}.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"Events file not found: {path}\n"
            "Run ingest_competition() first."
        )
    return pd.read_parquet(path)


def load_lineups(competition_id: int, season_id: int) -> pd.DataFrame:
    """Load a previously-ingested lineups Parquet file.

    Parameters
    ----------
    competition_id:
        StatsBomb competition identifier.
    season_id:
        StatsBomb season identifier.

    Returns
    -------
    pd.DataFrame

    Raises
    ------
    FileNotFoundError
        If the Parquet file does not exist (run ingest first).
    """
    path = RAW_DIR / f"lineups_{competition_id}_{season_id}.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"Lineups file not found: {path}\n"
            "Run ingest_competition() first."
        )
    return pd.read_parquet(path)


# ── CLI entry-point ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    parser = argparse.ArgumentParser(description="Ingest StatsBomb open data.")
    parser.add_argument(
        "--competition-id",
        type=int,
        default=None,
        help="Single competition ID to ingest (omit to ingest all).",
    )
    parser.add_argument(
        "--season-id",
        type=int,
        default=None,
        help="Season ID (required when --competition-id is set).",
    )
    parser.add_argument(
        "--max-matches",
        type=int,
        default=None,
        help="Limit matches per season (useful for quick tests).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Force re-download even if files already exist.",
    )
    args = parser.parse_args()

    if args.competition_id is not None:
        if args.season_id is None:
            parser.error("--season-id is required when --competition-id is set.")
        ingest_competition(
            args.competition_id,
            args.season_id,
            max_matches=args.max_matches,
            overwrite=args.overwrite,
        )
    else:
        ingest_all_open_data(
            max_matches_per_season=args.max_matches,
            overwrite=args.overwrite,
        )
