"""
api/main.py
-----------
FastAPI service exposing player similarity search.

Endpoints:
  GET  /health
  GET  /players                  — list all indexed players
  GET  /players/{player_id}      — player metadata + feature vector
  GET  /similar/{player_id}      — top-k similar players
  POST /similar/by-name          — same, look up by name
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import Optional

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Soccer Scouting Similarity Engine",
    version="0.1.0",
    description="Find players who play most like a given player, using learned embeddings.",
)


# ── lazy-loaded index ──────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def _get_index():  # type: ignore[return]
    """Load the similarity index once and cache it for the lifetime of the process."""
    from src.search import SimilarityIndex

    return SimilarityIndex.from_disk()


# ── response models ────────────────────────────────────────────────────────────

class PlayerSummary(BaseModel):
    player_id: int
    player_name: str
    team: str
    total_minutes: float


class SimilarPlayer(PlayerSummary):
    distance: float


class SimilarityResponse(BaseModel):
    query_player: PlayerSummary
    similar_players: list[SimilarPlayer]


# ── endpoints ──────────────────────────────────────────────────────────────────

@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/players", response_model=list[PlayerSummary])
def list_players(limit: int = Query(default=50, le=500)) -> list[PlayerSummary]:
    """Return a sample of indexed players."""
    idx = _get_index()
    sample = idx._embeddings[["player_name", "team", "total_minutes"]].head(limit)
    return [
        PlayerSummary(
            player_id=int(pid),
            player_name=str(row["player_name"]),
            team=str(row["team"]),
            total_minutes=float(row["total_minutes"]),
        )
        for pid, row in sample.iterrows()
    ]


@app.get("/players/{player_id}", response_model=PlayerSummary)
def get_player(player_id: int) -> PlayerSummary:
    """Return metadata for a specific player."""
    idx = _get_index()
    if player_id not in idx._embeddings.index:
        raise HTTPException(status_code=404, detail=f"player_id={player_id} not found.")
    row = idx._embeddings.loc[player_id]
    return PlayerSummary(
        player_id=player_id,
        player_name=str(row["player_name"]),
        team=str(row["team"]),
        total_minutes=float(row["total_minutes"]),
    )


@app.get("/similar/{player_id}", response_model=SimilarityResponse)
def similar_by_id(
    player_id: int,
    top_k: int = Query(default=10, ge=1, le=50),
) -> SimilarityResponse:
    """Find the top-k most similar players to the given player_id."""
    idx = _get_index()
    if player_id not in idx._embeddings.index:
        raise HTTPException(status_code=404, detail=f"player_id={player_id} not found.")

    query_row = idx._embeddings.loc[player_id]
    query_player = PlayerSummary(
        player_id=player_id,
        player_name=str(query_row["player_name"]),
        team=str(query_row["team"]),
        total_minutes=float(query_row["total_minutes"]),
    )

    results_df = idx.find_similar(player_id, top_k=top_k)
    similar = [
        SimilarPlayer(
            player_id=int(r["player_id"]),
            player_name=str(r["player_name"]),
            team=str(r["team"]),
            total_minutes=float(r["total_minutes"]),
            distance=float(r["distance"]),
        )
        for _, r in results_df.iterrows()
    ]
    return SimilarityResponse(query_player=query_player, similar_players=similar)


class NameQuery(BaseModel):
    name: str
    top_k: int = 10


@app.post("/similar/by-name", response_model=SimilarityResponse)
def similar_by_name(body: NameQuery) -> SimilarityResponse:
    """Find similar players by player name (case-insensitive partial match)."""
    idx = _get_index()
    try:
        results_df = idx.find_similar(body.name, top_k=body.top_k)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    # Resolve the matched player ID from the name map
    matched_name = body.name.lower()
    pid = idx._name_to_id.get(
        matched_name,
        next((v for k, v in idx._name_to_id.items() if matched_name in k), None),
    )
    if pid is None:
        raise HTTPException(status_code=404, detail=f"Player '{body.name}' not found.")

    query_row = idx._embeddings.loc[pid]
    query_player = PlayerSummary(
        player_id=int(pid),
        player_name=str(query_row["player_name"]),
        team=str(query_row["team"]),
        total_minutes=float(query_row["total_minutes"]),
    )
    similar = [
        SimilarPlayer(
            player_id=int(r["player_id"]),
            player_name=str(r["player_name"]),
            team=str(r["team"]),
            total_minutes=float(r["total_minutes"]),
            distance=float(r["distance"]),
        )
        for _, r in results_df.iterrows()
    ]
    return SimilarityResponse(query_player=query_player, similar_players=similar)
